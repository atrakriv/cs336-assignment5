from typing import List, Callable
import random

from transformers import PreTrainedModel

from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed

from cs336_alignment.reason.utils import dtype_mapping

# -------------------------------------------------------------#
# functions to initialize the vLLM model and load the policy into the vLLM model during training
# -------------------------------------------------------------#
def init_vllm(seed, model_path, gpu_memory_utilization, dtype, enable_prefix_caching, use_vllm_sleep_mode):
    """
    Start the inference process, here we use vLLM to hold a model on
    the same GPU as the policy.
    """
    vllm_set_random_seed(seed)
    # Monkeypatch from TRL:
    # https://github.com/huggingface/trl/blob/
    # 22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py
    # Patch vLLM to make sure we can place the vLLM model on the desired device
    # world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    # Note: Removed profiling_patch as it's vLLM version-dependent and causes AttributeError in newer versions
    # with world_size_patch:
    vllm_init_params = {
        "model": model_path,
        "gpu_memory_utilization": gpu_memory_utilization,
        "dtype": dtype_mapping[dtype],
        "enable_prefix_caching": enable_prefix_caching,
        "enable_sleep_mode": use_vllm_sleep_mode,
        # "enforce_eager": True,
    }
    return LLM(**vllm_init_params)

# -------------------------------------------------------------#
# Load the policy into the vLLM model
# -------------------------------------------------------------#
def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    """
    Copied from https://github.com/huggingface/trl/blob/
    22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py#L670.
    """
    # If the model was torch.compiled, the real module is in ._orig_mod
    if hasattr(policy, "_orig_mod"):
        policy_for_state = policy._orig_mod
    else:
        policy_for_state = policy
    state_dict = policy_for_state.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


# -------------------------------------------------------------#
# Evaluate the vLLM model
# -------------------------------------------------------------#
def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable,
    val_dataset: List[dict],
    prompt_template: str,
    sampling_params: SamplingParams,
    max_val_examples: int,
) -> dict:
    """
    Evaluate the model on a random subset of val_dataset using vLLM.

    Args:
        vllm_model: The vLLM model to use for generation.
        reward_fn: Reward function that takes (response, ground_truth) and returns a dict
                   with keys 'reward', 'format_reward', 'answer_reward'.
        val_dataset: List of dicts with 'problem' and 'answer' keys.
        prompt_template: Prompt template string with '{question}' placeholder.
        sampling_params: vLLM SamplingParams for generation.
        max_val_examples: Maximum number of examples to evaluate on.

    Returns:
        dict with keys:
            'mean_reward': float
            'mean_format_reward': float
            'mean_answer_reward': float
            'n_examples': int
    """
    n = min(max_val_examples, len(val_dataset))
    subset = random.sample(val_dataset, n)

    prompts = [prompt_template.replace("{question}", e['problem']) for e in subset]
    ground_truths = [e['answer'] for e in subset]

    outputs = vllm_model.generate(prompts, sampling_params)
    responses = [output.outputs[0].text for output in outputs]

    rewards = [reward_fn(response, gt) for response, gt in zip(responses, ground_truths)]

    mean_reward = sum(r['reward'] for r in rewards) / n
    mean_format_reward = sum(r['format_reward'] for r in rewards) / n
    mean_answer_reward = sum(r['answer_reward'] for r in rewards) / n

    return {
        'mean_reward': mean_reward,
        'mean_format_reward': mean_format_reward,
        'mean_answer_reward': mean_answer_reward,
        'n_examples': n,
    }