"""Smoke test: one example, configurable alpha, full transcript printed."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from src import scoring  # noqa: E402
from src.axis import SteeringHook, load_axis, validate_axis  # noqa: E402
from src.data import load_gsm8k_subset  # noqa: E402
from src.generate import GenConfig, batched_generate, load_model_and_tokenizer  # noqa: E402
from src.protocols import (  # noqa: E402
    _critic_messages,
    _revision_messages,
    _solver_messages,
)
from src.utils import ensure_hf_token, set_seed  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--axis_repo", default="Butanium/llama-3.1-8b-instruct-assistant-axis")
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--alpha_solver", type=float, default=0.0)
    ap.add_argument("--alpha_critic", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_examples", type=int, default=1)
    ap.add_argument("--max_new_tokens_solver", type=int, default=512)
    ap.add_argument("--max_new_tokens_critic", type=int, default=256)
    args = ap.parse_args()

    ensure_hf_token()
    set_seed(args.seed)

    print(f"[smoke] loading axis from {args.axis_repo}")
    axis = load_axis(args.axis_repo)
    print(f"[smoke] axis shape: {tuple(axis.shape)}, dtype: {axis.dtype}")

    print(f"[smoke] loading model: {args.model}")
    model, tokenizer = load_model_and_tokenizer(args.model)
    print(f"[smoke] model dtype: {next(model.parameters()).dtype}; "
          f"hidden: {model.config.hidden_size}; layers: {model.config.num_hidden_layers}")
    validate_axis(axis, model)

    examples = load_gsm8k_subset(args.n_examples, seed=args.seed, split="test")
    print(f"[smoke] loaded {len(examples)} example(s)")

    solver_gen = GenConfig(max_new_tokens=args.max_new_tokens_solver)
    critic_gen = GenConfig(max_new_tokens=args.max_new_tokens_critic)

    vec = axis[args.layer] if axis.dim() == 2 else axis

    def steer(alpha):
        if alpha == 0.0:
            return None
        return SteeringHook(model, vec, args.layer, alpha)

    questions = [ex.question for ex in examples]

    print(f"\n[smoke] === SOLVER pass (α={args.alpha_solver}) ===")
    initials = batched_generate(
        model, tokenizer,
        [_solver_messages(q) for q in questions],
        solver_gen, steering_ctx=steer(args.alpha_solver), batch_size=8,
    )

    print(f"\n[smoke] === CRITIC pass (α={args.alpha_critic}) ===")
    critiques = batched_generate(
        model, tokenizer,
        [_critic_messages(q, ia) for q, ia in zip(questions, initials)],
        critic_gen, steering_ctx=steer(args.alpha_critic), batch_size=8,
    )

    print(f"\n[smoke] === REVISION pass (α={args.alpha_solver}) ===")
    finals = batched_generate(
        model, tokenizer,
        [_revision_messages(q, ia, cr) for q, ia, cr in zip(questions, initials, critiques)],
        solver_gen, steering_ctx=steer(args.alpha_solver), batch_size=8,
    )

    smoke_path = Path(__file__).resolve().parents[1] / "outputs" / "smoke" / "transcripts.jsonl"
    smoke_path.parent.mkdir(parents=True, exist_ok=True)

    for ex, ia, cr, fi in zip(examples, initials, critiques, finals):
        ip = scoring.extract_final_answer(ia)
        fp = scoring.extract_final_answer(fi)
        ic = scoring.is_correct(ip, ex.gold_number)
        fc = scoring.is_correct(fp, ex.gold_number)
        print("\n" + "=" * 78)
        print(f"Q: {ex.question}")
        print(f"Gold: {ex.gold_number}")
        print(f"\n-- initial ({args.alpha_solver}) --\n{ia}\n  → parsed={ip}, correct={ic}")
        print(f"\n-- critique ({args.alpha_critic}) --\n{cr}")
        print(f"\n-- final ({args.alpha_solver}) --\n{fi}\n  → parsed={fp}, correct={fc}")
        print("=" * 78)

        row = {
            "example_id": ex.id,
            "question": ex.question,
            "gold_number": ex.gold_number,
            "alpha_solver": float(args.alpha_solver),
            "alpha_critic": float(args.alpha_critic),
            "layer": int(args.layer),
            "initial_text": ia,
            "critique_text": cr,
            "final_text": fi,
            "initial_parsed": ip,
            "final_parsed": fp,
            "initial_correct": bool(ic),
            "final_correct": bool(fc),
        }
        with open(smoke_path, "a") as f:
            f.write(json.dumps(row, default=lambda o: None) + "\n")
    print(f"[smoke] appended {len(examples)} row(s) to {smoke_path}")

    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
