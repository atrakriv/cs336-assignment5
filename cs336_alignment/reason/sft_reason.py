from pathlib import Path

import json
import numpy as np

import torch
from torch.optim import AdamW

from transformers import AutoTokenizer, AutoModelForCausalLM

from hw5 import tokenize_prompt_and_output, get_response_log_probs, sft_microbatch_train_step

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
    
    data_file = 'data/sft/sft_gpt-oss-120b_filtered.jsonl'
    prmpt_file = 'cs336_alignment/prompts/r1_zero.prompt'
    model_name = 'gpt2'
    batch_size = 64
    micro_batch_size = 2

    max_lr = 0.001
    betas = (0.9, 0.999)
    grad_norm_clip = 1.0

    # ── Data ─────────────────────────────────────────────────────────────────
    train_data_loader = SFTDataLoader(data_file, prmpt_file, micro_batch_size)
    
    
    # ── Model + Tokenizer + Optimizer ────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    optimizer =  AdamW (model.parameters(), lr=max_lr, betas=betas, eps=1e-08, weight_decay=0.01)
    
    # ── Training loop ────────────────────────────────────────────────────────
    steps = len(train_data_loader.data_list)//batch_size
    grad_acc_steps =  batch_size //micro_batch_size
    for step in range(1, steps+1):
        optimizer.zero_grad() #optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for micro_step in range(grad_acc_steps):
            batch = train_data_loader.get_batch()
            prompts = [e['prompt'] for e in batch]
            completions  = [e['completion'] for e in batch]
            tokenized_batch = tokenize_prompt_and_output(prompts, completions, tokenizer)
            max_len = model.config.n_positions  # 1024 for GPT-2
            for key in tokenized_batch: tokenized_batch[key] = tokenized_batch[key][:, :max_len]
            input_ids, labels,  completion_mask = tokenized_batch["input_ids"], tokenized_batch["labels"], tokenized_batch["response_mask"]
            # labels, response_mask = tokenized_batch["input_ids"].to(device), tokenized_batch["labels"].to(device), tokenized_batch["response_mask"].to(device)
            response_log_probs = get_response_log_probs(model, input_ids, labels, True)
            normalize_constant = completion_mask.sum(dim=1).clamp(min=1)
            loss, loss_metrics = sft_microbatch_train_step(response_log_probs["log_probs"], completion_mask, grad_acc_steps, normalize_constant=normalize_constant)
            loss_accum += loss.detach()
            print(loss)
        print('***', loss_accum )
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm_clip)
        optimizer.step()




