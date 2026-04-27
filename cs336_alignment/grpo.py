from pathlib import Path
import argparse, random

import torch
from torch.optim import AdamW

from transformers import AutoTokenizer, AutoModelForCausalLM

from vllm import SamplingParams

from omegaconf import OmegaConf
from configs.defaults import Config

from drgrpo_grader import r1_zero_reward_fn
from utils import dtype_mapping, load_dataset
from inference import init_vllm, load_policy_into_vllm_instance, evaluate_vllm
from hw5 import tokenize_prompt_and_output, get_response_log_probs, grpo_microbatch_train_step, compute_group_normalized_rewards


if __name__ == "__main__":

    # ── Load the config ──────────────────────────────────────────────────────
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to a yaml config file (overrides defaults)")
    args = parser.parse_args()

    default_config = OmegaConf.structured(Config)

    if args.config is not None:
        config = OmegaConf.merge(default_config, OmegaConf.load(args.config))
    else:
        config = default_config

    # ── Hyperparameters ──────────────────────────────────────────────────────
    
    prmpt_file = config.paths.prompt_template_file
    train_data_file = config.paths.train_data_file
    val_data_file = config.paths.val_data_file
    model_path = config.paths.model_path
    device = config.training.device
    
    n_grpo_steps = config.training.n_grpo_steps
    group_size = config.training.group_size
    rollout_batch_size = config.training.rollout_batch_size
    n_prompts_per_rollout = rollout_batch_size // group_size
    epochs_per_rollout_batch = config.training.epochs_per_rollout_batch
    train_batch_size = config.training.train_batch_size
    micro_batch_size = config.training.micro_batch_size
    grad_acc_steps = train_batch_size // micro_batch_size

    use_std_normalization = config.training.use_std_norm
    advantage_eps = config.training.advantage_eps
    loss_type = config.training.loss_type
    cliprange = config.training.cliprange
    norm_mode = config.training.norm_mode
    norm_constant = config.training.norm_constant
    max_grad_norm = config.training.max_grad_norm
   
    lr = config.training.learning_rate
    betas = (config.training.adam_beta1, config.training.adam_beta2)
    weight_decay = config.training.weight_decay
    adam_eps = config.training.adam_eps
  
    seed = config.training.seed
    dtype = config.vllm.dtype
    gpu_memory_utilization = config.vllm.gpu_memory_utilization
    enable_prefix_caching = config.vllm.enable_prefix_caching
    use_vllm_sleep_mode = config.training.use_vllm_sleep_mode

    temperature = config.training.temperature
    top_p = config.training.top_p
    max_tokens=config.training.max_tokens
    stop = [str(s) for s in config.training.stop]
    include_stop_str_in_output = config.training.include_stop_str_in_output
    min_tokens = config.training.min_tokens

    eval_interval = config.training.eval_interval
    max_val_examples = config.training.max_val_examples
    
    # ── Data ─────────────────────────────────────────────────────────────────
    
    prompt_template = Path(prmpt_file).read_text()
    train_dataset = load_dataset(data_file = train_data_file, data_type='train')
    val_dataset = load_dataset(data_file = val_data_file, data_type='val')


    # ── Tokenizer + Model + Optimizer ────────────────────────────────────────
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype_mapping[dtype],
        attn_implementation=config.training.attention_type,
        device_map=device
    )
    vllm_model = init_vllm(seed, model_path, gpu_memory_utilization, dtype, enable_prefix_caching, use_vllm_sleep_mode)
    
    optimizer = AdamW(model.parameters(), lr = lr, betas = betas, weight_decay = weight_decay, eps = adam_eps)

    # ── Training loop ────────────────────────────────────────────────────────

    sampling_params = SamplingParams(
        temperature = temperature,
        top_p = top_p,
        max_tokens = max_tokens,
        stop = [str(s) for s in stop],
        include_stop_str_in_output = include_stop_str_in_output,
        min_tokens = min_tokens
    )

    # eval_metrics = evaluate_vllm()
    
    for grpo_step in range(1, n_grpo_steps+1):

        batch = random.sample(train_dataset, n_prompts_per_rollout)     # a list of dics
        prompts = [prompt_template.replace("{question}", e['problem']) for e in batch for _ in range(group_size)]
        ground_truths = [e['answer'] for e in batch for _ in range(group_size)]

        outputs = vllm_model.generate(prompts, sampling_params)
        completions = [output.outputs[0].text for output in outputs]

        advantages, raw_rewards, rewards_meta = compute_group_normalized_rewards(
            r1_zero_reward_fn,
            completions,
            ground_truths,
            group_size,
            advantage_eps,
            use_std_normalization,
        )

        tokenized_train_data = tokenize_prompt_and_output(prompts, completions, tokenizer)
        input_ids, labels, completion_mask = tokenized_train_data['input_ids'], tokenized_train_data['labels'], tokenized_train_data['response_mask']
        mean_response_length = completion_mask.sum(dim=1).float().mean().item()

        model.eval()
        old_logps = []
        for idx in range(0, len(input_ids), micro_batch_size):
            input_ids_idx, labels_idx = map(lambda x: x[idx : idx + micro_batch_size].to(device), [input_ids, labels])
            with torch.no_grad():
                with torch.autocast(device_type=device.split(":")[0]):
                    logps = get_response_log_probs(model, input_ids, labels)["log_probs"]
                old_logps.append(logps)
        old_logps = torch.cat(old_logps, dim=0).cpu() # offload to cpu
        
        torch.cuda.empty_cache() # to save memory

        model.train()
        for train_epoch in range(epochs_per_rollout_batch):

            num_train_steps = rollout_batch_size // train_batch_size
            
            for train_step in range(num_train_steps):

                loss_accum = 0.0
                entropy_accum = 0.0
                clip_fraction_accum = 0.0
                mean_ratio_accum = 0.0
                total_response_tokens = 0

                for idx in range(grad_acc_steps):

                    base_idx = train_step * train_batch_size
                    start_idx = base_idx + idx * micro_batch_size
                    end_idx = start_idx + micro_batch_size

                    # microbatch of input_ids, labels, and response_mask
                    microbatch = {k: v[start_idx:end_idx].to(device) for k, v in tokenized_train_data.items()}
                    
                    with torch.autocast(device_type=device.split(":")[0], dtype=dtype_mapping[dtype]):
                        cur_logps_result = get_response_log_probs(model, microbatch["input_ids"], microbatch["labels"], True)

                    cur_logps_idx = cur_logps_result["log_probs"]
                    token_entropy = cur_logps_result["token_entropy"]
                    old_logps_idx = old_logps[start_idx:end_idx].to(device) if old_logps is not None else None
                    loss, meta = grpo_microbatch_train_step(
                        policy_log_probs = cur_logps_idx,
                        response_mask = microbatch["response_mask"],
                        gradient_accumulation_steps=grad_acc_steps,
                        loss_type = loss_type,
                        raw_rewards = raw_rewards[start_idx:end_idx].unsqueeze(-1).to(device),
                        advantages = advantages[start_idx:end_idx].unsqueeze(-1).to(device),
                        old_log_probs=old_logps_idx,
                        cliprange = cliprange,
                        # norm_mode = norm_mode, <later>
                        # norm_constant = normalize_constant, <later>
                    )

                    # accumulate
                    loss_accum += loss.detach()
                    entropy_accum += (token_entropy * microbatch["response_mask"]).sum().detach()
                    total_response_tokens += microbatch["response_mask"].sum().detach()
                    clip_fraction_accum += meta.get("clip_fraction", 0.0)
                    mean_ratio_accum += meta.get("mean_ratio", 0.0)
                    out = 'loss_accum: {}, entropy_accum: {}, total_response_tokens: {}, clip_fraction_accum: {}, mean_ratio_accum: {}'.format(
                        loss_accum, entropy_accum, total_response_tokens, clip_fraction_accum, mean_ratio_accum
                    )
                    print(tokenize_prompt_and_output)
                    del microbatch, cur_logps_result, cur_logps_idx, token_entropy, loss, old_logps_idx

                avg_entropy = entropy_accum / total_response_tokens
                avg_clip_fraction = clip_fraction_accum / grad_acc_steps
                avg_mean_ratio = mean_ratio_accum / grad_acc_steps
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                # calculate mean rewards for the rollout batch
                step_mean_format_reward = 0.0
                step_mean_answer_reward = 0.0
                step_mean_reward = 0.0
                rewards_len = len(rewards_meta['rewards'])
                for r in rewards_meta['rewards']:
                    step_mean_format_reward += r["format_reward"] / rewards_len
                    step_mean_answer_reward += r["answer_reward"] / rewards_len
                    step_mean_reward += r["reward"] / rewards_len

                torch.cuda.empty_cache()


        # wake vLLM before loading updated policy weights and next generation step
        if config.training.use_vllm_sleep_mode:
            vllm_model.wake_up()

        # load the model weights
        load_policy_into_vllm_instance(model, vllm_model)

        # intermediate evaluation on a subset of val_dataset
        if config.training.eval_interval > 0 and (grpo_step % eval_interval == 0 or grpo_step == n_grpo_steps):
            eval_step += 1
            eval_metrics = evaluate_vllm(
                vllm_model = vllm_model,
                reward_fn = r1_zero_reward_fn,
                val_dataset = val_dataset,
                prompt_template = prompt_template,
                sampling_params = sampling_params,
                max_val_examples = max_val_examples,
            )

    
