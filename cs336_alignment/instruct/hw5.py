# uv run --no-sync pytest -k test_tokenize_prompt_and_output

from pathlib import Path
from typing import Literal
import re

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from transformers import PreTrainedModel, PreTrainedTokenizerBase

def mmlu_baseline(model_output):
    model_output = model_output.upper().strip()
    pattern = r'The correct answer is (.)'
    # Patterns to match the extracted answer
    patterns = [
        r"^([ABCD])$",              # Exactly one character A/B/C/D
        r"\b([ABCD])\b",            # Word boundary
        r"ANSWER[: ]*([ABCD])",     # "Answer: C"
        r"OPTION[: ]*([ABCD])",     # "Option D"
        r"CHOICE[: ]*([ABCD])",     # "Choice B"
        r"\(([ABCD])\)",            # "(A)"
    ]
    for pattern in patterns:
        match = re.search(pattern, model_output)
        if match: return match.group(1)
    return None


class SFTDataset(Dataset):
    def __init__(self, tokenizer, dataset_path, seq_length, shuffle, prompt_path:None):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        prompt_template = Path(prompt_path).read()
        pairs = [dataset_path.read()]
        data = [prompt_template.format(question, answer) for question, answer in pairs]
        if shuffle: data.shuffle()
        docs = '|/end_of_text|'.join(data)
        # tokenize here?
        input_ids = tokenizer.encode(docs)
        labels = input_ids[1:] # + end_of_text
        # input_ids = split into non overlapping
        # labels = split into non overlapping
        self.data = [{'input_ids':i, 'labels':l} for i,l in zip(input_ids, labels)]
    def __len__(self):
        return len(self.data)
    def __getitem__(self, i):
        return self.data[i]
    
def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, torch.Tensor]:
    labels = labels.unsqueeze(dim=-1)
    logits = model(input_ids).logits
    log_probs = torch.log_softmax(logits, dim=-1)
    response_log_probs = log_probs.gather(dim=-1, index=labels).squeeze(-1)
    return response_log_probs


def per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
    prompt_template_file: str = 'cs336_alignment/instruct/prompts/alpaca_sft.prompt',
):
    prompt_template = Path(prompt_template_file).read_text().rstrip('\n')
    w_seq = prompt_template.format(instruction=prompt, response=response_chosen+tokenizer.eos_token)
    l_seq = prompt_template.format(instruction=prompt, response=response_rejected+tokenizer.eos_token)

    lm_device = next(lm.parameters()).device
    ref_device = next(lm_ref.parameters()).device

    w_ids = tokenizer(w_seq, return_tensors='pt').input_ids
    l_ids = tokenizer(l_seq, return_tensors='pt').input_ids

    w_logps = get_response_log_probs(lm, w_ids[:, :-1].to(lm_device), w_ids[:, 1:].to(lm_device))
    l_logps = get_response_log_probs(lm, l_ids[:, :-1].to(lm_device), l_ids[:, 1:].to(lm_device))
    with torch.no_grad():
        w_logps_ref = get_response_log_probs(lm_ref, w_ids[:, :-1].to(ref_device), w_ids[:, 1:].to(ref_device))
        l_logps_ref = get_response_log_probs(lm_ref, l_ids[:, :-1].to(ref_device), l_ids[:, 1:].to(ref_device))

    # sum each sequence to a scalar — prompt log-probs cancel in the difference
    w_ratio = (w_logps - w_logps_ref.to(lm_device)).sum()
    l_ratio = (l_logps - l_logps_ref.to(lm_device)).sum()

    return -F.logsigmoid(beta * (w_ratio - l_ratio))
