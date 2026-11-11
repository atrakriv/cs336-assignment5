# uv run --no-sync pytest -k test_tokenize_prompt_and_output

from typing import Literal

import torch

from transformers import PreTrainedModel


def tokenize_prompt_and_output(prompts, completions, tokenizer):

    prompt_tokens = tokenizer(prompts)['input_ids']
    completion_tokens = tokenizer(completions)['input_ids']
    prompt_completions = [p+c for p,c in zip(prompt_tokens, completion_tokens)]
    prompt_lens = [len(p) for p in prompt_tokens]
    total_lens = [len(pc) for pc in prompt_completions]
    max_len = max(total_lens)
    seq_len = max_len - 1
    batch_size = len(prompt_completions)
    pad_id = tokenizer.eos_token_id

    ''' sol 1: create padded tensors with size max_len - 1'''
    input_ids = torch.full((batch_size, seq_len), pad_id, dtype=torch.long)
    labels = torch.full((batch_size, seq_len), pad_id, dtype=torch.long)
    response_mask = torch.full((batch_size, seq_len), False, dtype=torch.bool)

    for idx, pc in enumerate(prompt_completions):
        t = torch.tensor(pc, dtype=torch.long)
        input_ids[idx, :total_lens[idx]-1] = t[:-1]
        labels[idx, :total_lens[idx]-1] = t[1:]
        response_mask[idx, prompt_lens[idx]-1:total_lens[idx]-1] = True

    # ''' sol 2: create a padded tensor with size max_len'''
    # prmpt_compt_padded = torch.full((batch_size, max(total_lens)), pad_id )
    # response_mask = torch.full((batch_size, max(total_lens)), 0)
    # for i in range(batch_size):
    #     prmpt_compt_padded[i, :prompt_lens[i]] = torch.tensor(prompt_tokens[i])
    #     prmpt_compt_padded[i, prompt_lens[i]: total_lens[i]] = torch.tensor(completion_tokens[i])
    #     response_mask[i, prompt_lens[i]: total_lens[i]] = 1
    # input_ids = prmpt_compt_padded[:, :-1]
    # labels = prmpt_compt_padded[:, 1:]
    # response_mask = response_mask[:, 1:]

    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask,
    }


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    # return -(torch.softmax(logits, dim=-1)*torch.log_softmax(logits, dim=-1)).sum(dim=-1)
    log_p = torch.log_softmax(logits, dim=-1)
    return -(log_p.exp() * log_p).sum(dim=-1)


def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor, return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    labels = labels.unsqueeze(dim=-1)
    logits = model(input_ids).logits
    log_probs = torch.log_softmax(logits, dim=-1)
    response_log_probs = log_probs.gather(dim=-1, index=labels).squeeze(-1)
    if not return_token_entropy: return {"log_probs":response_log_probs}
    else: return {"log_probs":response_log_probs, "token_entropy": compute_entropy(logits)}

def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> torch.Tensor:
    return (tensor*mask).sum(dim=dim)/normalize_constant


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # masked_normalize sums over the sequence (or
    # takes the mean over the sequence if normalize_constant != 1)
    # here, we take the mean over the samples in the bacth
    loss = - (masked_normalize(policy_log_probs, response_mask, normalize_constant, dim=-1).mean()) / gradient_accumulation_steps
    loss.backward()

    response_token_count = response_mask.sum()
    per_token_log_prob = (policy_log_probs * response_mask).sum() / response_token_count
    metadata = {
        "mean_token_log_prob": per_token_log_prob.detach(),
        "perplexity": (-per_token_log_prob).exp().detach(),
        "response_token_count": response_token_count.detach(),
    }
    '''
        The most actionable signal is mean_token_log_prob.
        it lets you spot overfitting (rapidly approaches 0), 
        underfitting, 
        or reward hacking in later RLHF stages where SFT is used as a reference.
    '''
    return loss, metadata


def compute_group_normalized_rewards( 
    reward_fn,
    rollout_responses,
    repeated_ground_truths,
    group_size,
    advantage_eps,
    normalize_by_std,
):
    results = [reward_fn(r, g) for r, g in zip(rollout_responses, repeated_ground_truths)]
    rewards = torch.tensor([x['reward'] for x in results])

    reward_groups = rewards.view(-1, group_size)
    mean = reward_groups.mean(dim=-1, keepdim=True)
    if normalize_by_std:
        std = reward_groups.std(dim=-1, keepdim=True)
        advantages = ((reward_groups - mean) / (std + advantage_eps)).view(-1)
    else:
        advantages = (reward_groups - mean).view(-1)
    '''
    The most practically valuable of these is frac_zero_std_groups.
    if it's high, GRPO is getting no learning signal for those prompts and you'd want to know early.
    '''
    metadata = {
        "mean_reward": rewards.mean().item(),
        "mean_format_reward": sum(x['format_reward'] for x in results) / len(results),
        "mean_answer_reward": sum(x['answer_reward'] for x in results) / len(results),
        "frac_zero_std_groups": (reward_groups.std(dim=-1) < advantage_eps).float().mean().item(),
    }
    return advantages, rewards, metadata

def compute_naive_policy_gradient_loss( 
    raw_rewards_or_advantages: torch.Tensor, 
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    return - raw_rewards_or_advantages * policy_log_probs


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    prob_ratio = torch.exp(policy_log_probs - old_log_probs)
    clipped_ratio = torch.clamp(prob_ratio, 1 - cliprange, 1 + cliprange)
    unclipped_objective = prob_ratio * advantages
    clipped_objective = clipped_ratio * advantages
    loss = -torch.min(unclipped_objective, clipped_objective)
    '''
    clip_fraction: the fraction of tokens where clipping was active
    '''
    metadata = {
        "clip_fraction": (clipped_ratio != prob_ratio).float().mean(),
    }
    return loss, metadata


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if loss_type == "no_baseline":
        if raw_rewards is None:
            raise ValueError("raw_rewards is required for loss_type='no_baseline'")
        return compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs), {}
    elif loss_type == "reinforce_with_baseline":
        if advantages is None:
            raise ValueError("advantages is required for loss_type='reinforce_with_baseline'")
        return compute_naive_policy_gradient_loss(advantages, policy_log_probs), {}
    elif loss_type == "grpo_clip":
        if advantages is None or old_log_probs is None or cliprange is None:
            raise ValueError("advantages, old_log_probs, and cliprange are all required for loss_type='grpo_clip'")
        return compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type!r}. Expected one of: 'no_baseline', 'reinforce_with_baseline', 'grpo_clip'")


def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
) -> torch.Tensor:
    sum_ = (tensor * mask).sum(dim=dim)
    # mask.sum(dim=dim).clamp(min=1) to avoid devision by zero
    count_ = mask.sum(dim=dim) 
    return sum_ / count_


def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss, metadata = compute_policy_gradient_loss(policy_log_probs, loss_type, raw_rewards, advantages, old_log_probs, cliprange)
    loss = masked_mean(loss, response_mask, dim=-1).mean() / gradient_accumulation_steps
    loss.backward()

    # metadata["response_token_count"] = response_mask.sum().detach()
    # if raw_rewards is not None:
    #     metadata["mean_reward"] = raw_rewards.mean().detach()
    # if advantages is not None:
    #     metadata["mean_advantage"] = advantages.mean().detach()

    return loss.detach(), metadata
