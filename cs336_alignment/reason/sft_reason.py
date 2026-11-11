from pathlib import Path

import json
import os
import numpy as np

import torch
from torch.optim import AdamW

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig

from cs336_alignment.reason.hw5 import tokenize_prompt_and_output, get_response_log_probs, sft_microbatch_train_step

class SFTDataLoader:
    def __init__(self, data_file, prmpt_file, batch_size, seed=42):
        self.data_file = data_file
        self.prmpt_file = prmpt_file
        self.batch_size = batch_size
        self.rng = np.random.RandomState(seed)
        '''load & shuffle'''
        self._load_data()
        self.reset()
    
    def _load_data(self):
        with open(self.data_file) as f: data = json.load(f)
        template = Path(self.prmpt_file).read_text()
        self.data_list = [
            {
                'prompt':template.format(question = e['problem']),
                'completion':e['reasoning_trace'],
                'ground_truth':e['expected_answer']
             }
             for e in data]

    def get_batch(self):
        batch_data = self.data_list[self.ptr: self.ptr + self.batch_size]
        self.ptr += self.batch_size
        if self.ptr >= len(self.data_list): self.reset()
        return batch_data
    
    def reset(self):
        self.rng.shuffle(self.data_list)
        self.ptr = 0
    

if __name__ == "__main__":

    # ── Hyperparameters ──────────────────────────────────────────────────────
    
    data_file = 'cs336_alignment/reason/data/sft/sft_gpt-oss-120b_filtered.jsonl'
    prmpt_file = 'cs336_alignment/reason/prompts/r1_zero.prompt'
    model_name = 'gpt2'
    batch_size = 64
    micro_batch_size = 2

    # LoRA
    use_lora = True
    use_qlora = True  # 4-bit NF4 base model + LoRA adapters (needs CUDA + bitsandbytes)
    lora_r = 16
    lora_alpha = 32
    lora_dropout = 0.05
    is_gpt2 = model_name.startswith('gpt2')
    # QLoRA paper: adapt all linear layers. GPT-2 uses Conv1D, so name its modules explicitly.
    lora_target_modules = ["c_attn", "c_proj"] if is_gpt2 else "all-linear"
    max_lr = 2e-4 if use_lora else 0.001  # LoRA typically tolerates a higher LR than full FT; 1e-3 is high for full FT
    betas = (0.9, 0.999)
    grad_norm_clip = 1.0

    # ── Data ─────────────────────────────────────────────────────────────────
    train_data_loader = SFTDataLoader(data_file, prmpt_file, micro_batch_size)
    
    
    # ── Model + Tokenizer + Optimizer ────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if use_qlora:
        assert use_lora and device == 'cuda', "QLoRA needs use_lora=True and a CUDA GPU"
        if is_gpt2:
            print("WARNING: bitsandbytes only quantizes nn.Linear; GPT-2 uses Conv1D, so its "
                  "weights will NOT be 4-bit. Use a Llama/Qwen-style model to see QLoRA savings.")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb_config, device_map={"": 0}, torch_dtype=torch.bfloat16)
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    max_len = getattr(model.config, 'n_positions', None) or model.config.max_position_embeddings

    if use_lora:
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=lora_target_modules,
            fan_in_fan_out=is_gpt2,  # GPT-2 uses Conv1D (transposed weights)
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if use_qlora:  # paged 8-bit AdamW as in the QLoRA paper
        import bitsandbytes as bnb
        optimizer = bnb.optim.PagedAdamW8bit(trainable_params, lr=max_lr, betas=betas, eps=1e-08, weight_decay=0.01)
    else:
        optimizer = AdamW(trainable_params, lr=max_lr, betas=betas, eps=1e-08, weight_decay=0.01)
    
    # ── Training loop ────────────────────────────────────────────────────────
    steps = len(train_data_loader.data_list)//batch_size
    if os.environ.get('MAX_STEPS'): steps = min(steps, int(os.environ['MAX_STEPS']))  # quick smoke test
    grad_acc_steps =  batch_size //micro_batch_size
    for step in range(1, steps+1):
        optimizer.zero_grad() #optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for micro_step in range(grad_acc_steps):
            batch = train_data_loader.get_batch()
            prompts = [e['prompt'] for e in batch]
            completions  = [e['completion'] for e in batch]
            tokenized_batch = tokenize_prompt_and_output(prompts, completions, tokenizer)
            for key in tokenized_batch: tokenized_batch[key] = tokenized_batch[key][:, :max_len]
            input_ids, labels, completion_mask = (tokenized_batch[k].to(device) for k in ("input_ids", "labels", "response_mask"))
            response_log_probs = get_response_log_probs(model, input_ids, labels, True)
            normalize_constant = completion_mask.sum(dim=1).clamp(min=1)
            loss, loss_metrics = sft_microbatch_train_step(response_log_probs["log_probs"], completion_mask, grad_acc_steps, normalize_constant=normalize_constant)
            loss_accum += loss.detach()
            print(loss)
        print('***', loss_accum )
        norm = torch.nn.utils.clip_grad_norm_(trainable_params, grad_norm_clip)
        optimizer.step()




