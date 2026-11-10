import re
import os
from pathlib import Path
from collections import defaultdict

import pandas as pd
from vllm import LLM, SamplingParams


MMLU_TEST_DIR = "data/mmlu/test"
PROMPT_FILE = "cs336_alignment/instruct/prompts/mmlu.prompt"
MODEL = "meta-llama/Llama-3.2-3B-Instruct"


def load_mmlu_test(test_dir: str):
    records = []
    for csv_file in sorted(Path(test_dir).glob("*_test.csv")):
        subject = csv_file.stem.replace("_test", "").replace("_", " ")
        df = pd.read_csv(csv_file, header=None, names=["question", "A", "B", "C", "D", "answer"])
        for _, row in df.iterrows():
            records.append({"subject": subject, "row": row})
    return records


def make_prompt(template: str, subject: str, row: pd.Series) -> str:
    return template.format(
        subject=subject,
        question=row["question"],
        options=[row["A"], row["B"], row["C"], row["D"]],
    )


def extract_answer(text: str) -> str:
    m = re.search(r"correct answer is\s+([A-D])", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b([A-D])\b", text)
    if m:
        return m.group(1).upper()
    return ""


def main():
    prompt_template = Path(PROMPT_FILE).read_text()
    records = load_mmlu_test(MMLU_TEST_DIR)
    print(f"Loaded {len(records)} questions across {len(set(r['subject'] for r in records))} subjects")

    prompts = [make_prompt(prompt_template, r["subject"], r["row"]) for r in records]
    ground_truths = [r["row"]["answer"].strip().upper() for r in records]
    subjects = [r["subject"] for r in records]

    llm = LLM(model=MODEL, dtype="bfloat16", gpu_memory_utilization=0.85)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=64)

    outputs = llm.generate(prompts, sampling_params)
    responses = [o.outputs[0].text for o in outputs]

    per_subject: dict[str, list[bool]] = defaultdict(list)
    for subject, response, gt in zip(subjects, responses, ground_truths):
        pred = extract_answer(response)
        per_subject[subject].append(pred == gt)

    subject_accs = {s: sum(v) / len(v) for s, v in per_subject.items()}
    overall = sum(sum(v) for v in per_subject.values()) / len(ground_truths)

    print(f"\n{'Subject':<45} {'Acc':>6}  {'N':>5}")
    print("-" * 60)
    for subject in sorted(subject_accs):
        n = len(per_subject[subject])
        print(f"{subject:<45} {subject_accs[subject]:>6.1%}  {n:>5}")
    print("-" * 60)
    print(f"{'Overall':<45} {overall:>6.1%}  {len(ground_truths):>5}")


if __name__ == "__main__":
    main()
