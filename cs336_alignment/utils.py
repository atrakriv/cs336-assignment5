from pathlib import Path

import json, random 

import torch


dtype_mapping= {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

# return a list of dics
def load_dataset(data_file: str, data_type: str='train'):
    with open(data_file, 'r') as f:
        if data_type in ['train', 'val']:
            data = [json.loads(line) for line in f]
            data = [{'problem': item['problem'], 'answer': item['answer']} for item in data]
            return data
        else:
            raise ValueError(f"Invalid data type: {data_type}")
        

def save_sample_rollouts(prompts, completions, ground_truths, advantages, rewards_meta, n_rollouts_to_log, grpo_step, output_dir):
    save_indices = random.sample(range(len(completions)), min(n_rollouts_to_log, len(completions)))
    rollouts_dir = Path(output_dir) / "rollouts"
    rollouts_dir.mkdir(parents=True, exist_ok=True)
    rollout_records = [
        {
            "prompt": prompts[i],
            "response": completions[i],
            "ground_truth": ground_truths[i],
            "advantage": advantages[i].item(),
            "reward": rewards_meta["rewards"][i]["reward"],
            "format_reward": rewards_meta["rewards"][i]["format_reward"],
            "answer_reward": rewards_meta["rewards"][i]["answer_reward"],
        }
        for i in save_indices
    ]
    rollout_file = rollouts_dir / f"rollouts_step_{grpo_step:03d}.jsonl"
    with open(rollout_file, "w") as f:
        f.write("\n".join(json.dumps(r) for r in rollout_records) + "\n")