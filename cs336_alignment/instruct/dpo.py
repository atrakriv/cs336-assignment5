from datasets import load_dataset, Dataset


# All available HH-RLHF subset names
HH_SUBSETS = [
    "harmless-base",
    "helpful-base",
    "helpful-online",
    "helpful-rejection-sampled",
]


def load_hh_rlhf(split: str = "train", subset: str | None = None) -> Dataset:
    """Load one HH-RLHF subset (or all configs concatenated if subset is None)."""
    if subset is not None:
        return load_dataset("Anthropic/hh-rlhf", data_dir=subset, split=split)

    from datasets import concatenate_datasets
    shards = []
    for name in HH_SUBSETS:
        shards.append(load_dataset("Anthropic/hh-rlhf", data_dir=name, split=split))
    return concatenate_datasets(shards)


def _parse_turns(text: str) -> list[dict[str, str]]:
    """Parse a raw HH-RLHF conversation string into a list of {role, content} turns."""
    turns = []
    remaining = text.strip()
    role_map = {"Human": "human", "Assistant": "assistant"}

    while remaining:
        for prefix in ("Human:", "Assistant:"):
            if remaining.startswith(prefix):
                role = role_map[prefix.rstrip(":")]
                remaining = remaining[len(prefix):].lstrip(" ")
                next_human = remaining.find("\n\nHuman:")
                next_asst = remaining.find("\n\nAssistant:")
                candidates = [i for i in (next_human, next_asst) if i != -1]
                if candidates:
                    end = min(candidates)
                    content = remaining[:end].strip()
                    remaining = remaining[end + 2:].strip()
                else:
                    content = remaining.strip()
                    remaining = ""
                turns.append({"role": role, "content": content})
                break
        else:
            break
    return turns


def _is_single_turn(turns: list[dict[str, str]]) -> bool:
    """Return True only if the human spoke exactly once (single-turn exchange)."""
    human_turns = sum(1 for t in turns if t["role"] == "human")
    return human_turns == 1


def get_dpo_examples(
    split: str = "train",
    subset: str | None = None,
) -> list[dict]:
    """
    Load HH-RLHF and return single-turn DPO examples.

    Each returned dict has:
        "instruction": str   — the single human message
        "chosen":      str   — assistant response in the chosen conversation
        "rejected":    str   — assistant response in the rejected conversation
        "source":      str   — subset name the example came from
    """
    if subset is not None:
        subsets_to_load = [subset]
    else:
        subsets_to_load = HH_SUBSETS

    examples = []
    for name in subsets_to_load:
        try:
            dataset = load_dataset("Anthropic/hh-rlhf", data_dir=name, split=split)
        except Exception:
            continue

        for example in dataset:
            chosen_turns = _parse_turns(example["chosen"])
            rejected_turns = _parse_turns(example["rejected"])

            # Skip multi-turn: human must appear exactly once in both conversations
            if not (_is_single_turn(chosen_turns) and _is_single_turn(rejected_turns)):
                continue

            # The instruction is the first (and only) human message.
            # Chosen/rejected responses are the corresponding assistant messages.
            instruction = chosen_turns[0]["content"]
            chosen_response = next(
                (t["content"] for t in chosen_turns if t["role"] == "assistant"), None
            )
            rejected_response = next(
                (t["content"] for t in rejected_turns if t["role"] == "assistant"), None
            )

            if instruction is None or chosen_response is None or rejected_response is None:
                continue

            examples.append({
                "instruction": instruction,
                "chosen": chosen_response,
                "rejected": rejected_response,
                "source": name,
            })

    return examples
