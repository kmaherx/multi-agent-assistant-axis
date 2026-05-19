"""GSM8K loader."""
from __future__ import annotations

import re
from dataclasses import dataclass

from datasets import load_dataset


@dataclass
class GSM8KExample:
    id: str
    question: str
    gold_text: str
    gold_number: float


_NUM_RE = re.compile(r"-?[\d,]+(?:\.\d+)?")


def _parse_gold(answer: str) -> float:
    # GSM8K gold answers end with "#### <number>"
    if "####" in answer:
        tail = answer.split("####", 1)[1].strip()
    else:
        tail = answer.strip()
    matches = _NUM_RE.findall(tail.replace("$", ""))
    if not matches:
        raise ValueError(f"no number in gold: {answer!r}")
    return float(matches[-1].replace(",", ""))


def load_gsm8k_subset(
    n: int,
    seed: int = 42,
    split: str = "test",
) -> list[GSM8KExample]:
    ds = load_dataset("gsm8k", "main", split=split)
    ds = ds.shuffle(seed=seed)
    if n is not None and n > 0:
        ds = ds.select(range(min(n, len(ds))))
    out = []
    for i, row in enumerate(ds):
        try:
            gold = _parse_gold(row["answer"])
        except ValueError:
            continue
        out.append(
            GSM8KExample(
                id=f"gsm8k-{split}-{i}",
                question=row["question"].strip(),
                gold_text=row["answer"].strip(),
                gold_number=gold,
            )
        )
    return out
