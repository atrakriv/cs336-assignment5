from pathlib import Path

from typing import List, Callable, Any, Optional

from datasets import load_dataset

from vllm import LLM, SamplingParams

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    ground_truths: List[Any],
    eval_sampling_params: SamplingParams,
    save_path: Optional[str] = None,
) -> list[dict]:
    outputs = vllm_model.generate(prompts, eval_sampling_params)
    results = []
    for idx, output in enumerate(outputs):
        prompt = output.prompt
        completion = output.outputs[0].text
        ground_truth = ground_truths[idx]
        rewards = reward_fn(completion, ground_truth)
        format_reward, answer_reward = rewards["format_reward"], rewards["answer_reward"]
        results.append({
            "prompt": prompt,
            "completion": completion,
            "ground_truth": ground_truth,
            "format_reward": format_reward,
            "answer_reward": answer_reward,
        })

    f1a0 = [r for r in results if r["format_reward"] == 1 and r["answer_reward"] == 0]
    f1a1 = [r for r in results if r["format_reward"] == 1 and r["answer_reward"] == 1]

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        # with open(save_path, "w") as f:
        #     json.dump(results, f, indent=2)

        with open(save_path, "w") as f:
            f.write("=== FORMAT CORRECT, ANSWER WRONG (format_reward=1, answer_reward=0) ===\n")
            f.write(f"Total samples in this category: {len(f1a0)}\n\n")
            for i, sample in enumerate(f1a0[:10], 1):
                f.write(f"--- Sample {i} ---\n")
                f.write(f"Prompt:\n{sample['prompt']}\n\n")
                f.write(f"Completion:\n{sample['completion']}\n\n")
                f.write(f"Ground Truth: {sample['ground_truth']}\n")
                f.write(f"Format Reward: {sample['format_reward']} | Answer Reward: {sample['answer_reward']}\n")
                f.write("\n")

            f.write("=== FORMAT CORRECT, ANSWER CORRECT (format_reward=1, answer_reward=1) ===\n")
            f.write(f"Total samples in this category: {len(f1a1)}\n\n")
            for i, sample in enumerate(f1a1[:10], 1):
                f.write(f"--- Sample {i} ---\n")
                f.write(f"Prompt:\n{sample['prompt']}\n\n")
                f.write(f"Completion:\n{sample['completion']}\n\n")
                f.write(f"Ground Truth: {sample['ground_truth']}\n")
                f.write(f"Format Reward: {sample['format_reward']} | Answer Reward: {sample['answer_reward']}\n")
                f.write("\n")

    return results


if __name__ == "__main__":

    r1_zero_template = Path("cs336_alignment/prompts/r1_zero.prompt").read_text()
    dataset = 'nlile/hendrycks-MATH-benchmark'
    ds = load_dataset(dataset, split="test")
    df = ds.to_pandas()
    prompts = [r1_zero_template.format(question) for question in list(df['problems'])]
    prompts = [r1_zero_template.format(question=problem) for problem in df['problem']]
    ground_truths = list(df['answer'])

    model_name = 'Qwen/Qwen2.5-Math-1.5B'
    vllm_model = LLM(model=model_name)

    eval_sampling_params = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=1024)
    eval_sampling_params.stop = ["</answer>"]
    eval_sampling_params.include_stop_str_in_output = True

    results = evaluate_vllm(vllm_model, r1_zero_reward_fn, prompts, ground_truths, eval_sampling_params, save_path="outputs/baseline_eval.txt")