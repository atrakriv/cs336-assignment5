import os
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd



def create_prompt(subject: str, q_dict: dict):
    prompt = prompt_template.format(
        subject = subject,
        question = q_dict["question"],
        options = [q_dict["A"], q_dict["B"], q_dict["C"], q_dict["D"]],
    )
    return prompt

def should_change(val_dir):
    csv_files = [f for f in os.listdir(val_dir) if f.endswith("_test.csv")]
    for csv_file in csv_files:
        df = pd.read_csv(os.path.join(val_dir, csv_file))


    prompt_file = 'cs336_alignment/instruct/prompts/mmlu.prompt'
    test_path = 'data/mmlu/test/global_facts_test.csv'
    df = pd.read_csv(test_path, header=None, names=['question', 'A', 'B', 'C', 'D', 'answer'])
    prompt_template = Path(prompt_file).read_text()
    subject = test_path.split('/')[-1].replace('_test.csv','').replace('_',' ')
    all_prompts = []
    for _, row in df.iterrows():
        prompt = prompt_template.format(
            subject=subject,
            question=row['question'],
            options=[row['A'], row['B'], row['C'], row['D']],
        )
        all_prompts.append(prompt)




class MMLU():
    def __init__(
        self,
        val_dir: str,
        # mmlu_question_file: str,
        prompt_file: str,
        results_dir: Optional[str] = None,
        # eval_filename: Optional[str] = None,
        headers: Optional[List[str]] = None,
    ):
        
        self.val_dir = val_dir
        # self.mmlu_question_file = mmlu_question_file
        self.prompt_file = prompt_file
        self.headers = headers or ["question", "A", "B", "C", "D", "answer"]
        self.answer_col = self.headers[-1]