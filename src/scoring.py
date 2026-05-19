"""Final-answer extraction + numeric scoring."""
from __future__ import annotations

import re

_FINAL_RE = re.compile(
    r"final\s*answer\s*[:\-=]\s*\$?\s*(-?[\d,]+(?:\.\d+)?)",
    re.IGNORECASE,
)
_NUM_RE = re.compile(r"-?[\d,]+(?:\.\d+)?")


def extract_final_answer(text: str) -> float | None:
    """Try to extract a numeric final answer from a model response.

    Strategy:
      1. Last "Final answer: <num>" match.
      2. Fallback: last number in the text.
      3. Returns None if neither yields a parsable number.
    """
    if not text:
        return None
    matches = _FINAL_RE.findall(text)
    if matches:
        candidate = matches[-1]
    else:
        nums = _NUM_RE.findall(text)
        if not nums:
            return None
        candidate = nums[-1]
    candidate = candidate.replace(",", "").replace("$", "").strip()
    try:
        return float(candidate)
    except ValueError:
        return None


def is_correct(pred: float | None, gold: float, atol: float = 1e-4) -> bool:
    if pred is None:
        return False
    if gold == 0:
        return abs(pred) <= atol
    return abs(pred - gold) <= max(atol, atol * abs(gold))
