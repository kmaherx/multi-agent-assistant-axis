"""Aggregate per-example rows into per-condition metrics."""
from __future__ import annotations

from collections import defaultdict


def per_example_row(transcript, scoring) -> dict:
    """Build the JSONL row for a single transcript using src.scoring."""
    initial = scoring.extract_final_answer(transcript.initial_text)
    final = scoring.extract_final_answer(transcript.final_text)
    initial_correct = scoring.is_correct(initial, transcript.gold_number)
    final_correct = scoring.is_correct(final, transcript.gold_number)
    return {
        "example_id": transcript.example_id,
        "question": transcript.question,
        "gold_number": transcript.gold_number,
        "alpha_solver": transcript.alpha_solver,
        "alpha_critic": transcript.alpha_critic,
        "initial_text": transcript.initial_text,
        "critique_text": transcript.critique_text,
        "final_text": transcript.final_text,
        "initial_parsed": initial,
        "final_parsed": final,
        "initial_correct": initial_correct,
        "final_correct": final_correct,
        "answer_changed": (initial != final) and (initial is not None) and (final is not None),
        "initial_parse_failed": initial is None,
        "final_parse_failed": final is None,
    }


def aggregate(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    n_init_correct = sum(1 for r in rows if r["initial_correct"])
    n_final_correct = sum(1 for r in rows if r["final_correct"])
    n_w2r = sum(1 for r in rows if not r["initial_correct"] and r["final_correct"])
    n_r2w = sum(1 for r in rows if r["initial_correct"] and not r["final_correct"])
    n_initial_wrong = sum(1 for r in rows if not r["initial_correct"])
    n_initial_right = sum(1 for r in rows if r["initial_correct"])
    n_answer_changed = sum(1 for r in rows if r["answer_changed"])
    n_change_when_wrong = sum(1 for r in rows if not r["initial_correct"] and r["answer_changed"])
    n_change_when_right = sum(1 for r in rows if r["initial_correct"] and r["answer_changed"])
    n_init_parse_fail = sum(1 for r in rows if r["initial_parse_failed"])
    n_final_parse_fail = sum(1 for r in rows if r["final_parse_failed"])

    def _safe(a: int, b: int) -> float:
        return float(a) / b if b > 0 else 0.0

    return {
        "n": n,
        "initial_accuracy": n_init_correct / n,
        "final_accuracy": n_final_correct / n,
        "wrong_to_right_rate": _safe(n_w2r, n_initial_wrong),
        "right_to_wrong_rate": _safe(n_r2w, n_initial_right),
        "answer_change_rate": n_answer_changed / n,
        "answer_change_when_wrong": _safe(n_change_when_wrong, n_initial_wrong),
        "answer_change_when_right": _safe(n_change_when_right, n_initial_right),
        "initial_parse_failure_rate": n_init_parse_fail / n,
        "final_parse_failure_rate": n_final_parse_fail / n,
    }


def per_condition_metrics(rows: list[dict]) -> list[dict]:
    """Group rows by (alpha_solver, alpha_critic), aggregate each group."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["alpha_solver"], r["alpha_critic"])].append(r)
    out = []
    for (a_s, a_c), group in sorted(groups.items()):
        agg = aggregate(group)
        agg = {"alpha_solver": a_s, "alpha_critic": a_c, **agg}
        out.append(agg)
    return out
