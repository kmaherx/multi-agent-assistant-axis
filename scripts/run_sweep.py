"""Pilot sweep: GSM8K × (α_solver, α_critic) grid."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src import scoring  # noqa: E402
from src.axis import load_axis, validate_axis  # noqa: E402
from src.data import load_gsm8k_subset  # noqa: E402
from src.generate import GenConfig, load_model_and_tokenizer  # noqa: E402
from src.metrics import per_condition_metrics, per_example_row  # noqa: E402
from src.protocols import run_two_agent  # noqa: E402
from src.utils import (  # noqa: E402
    append_jsonl,
    dump_json,
    ensure_hf_token,
    git_commit,
    gpu_summary,
    run_dir_name,
    set_seed,
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--model_short", default="llama31_8b")
    ap.add_argument("--axis_repo", default="Butanium/llama-3.1-8b-instruct-assistant-axis")
    ap.add_argument("--dataset", default="gsm8k")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n_examples", type=int, default=200)
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--alphas", type=float, nargs="+", default=[-5.0, -2.0, 0.0, 2.0, 5.0])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_new_tokens_solver", type=int, default=512)
    ap.add_argument("--max_new_tokens_critic", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--output_root", default="outputs")
    ap.add_argument("--resume", action="store_true",
                    help="If set, skip (α_s,α_c) pairs whose rows are already in the jsonl.")
    return ap.parse_args()


def main():
    args = parse_args()
    ensure_hf_token()
    set_seed(args.seed)

    run_dir = Path(args.output_root) / run_dir_name(
        args.dataset, args.model_short, args.layer, args.seed
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = run_dir / "per_example_results.jsonl"

    # Resume: collect (alpha_s, alpha_c, example_id) tuples already present.
    done = set()
    if args.resume and jsonl_path.exists():
        with open(jsonl_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                done.add((r["alpha_solver"], r["alpha_critic"], r["example_id"]))
        print(f"[sweep] resume: {len(done)} rows already in {jsonl_path}")

    print(f"[sweep] loading axis: {args.axis_repo}")
    axis = load_axis(args.axis_repo)
    print(f"[sweep] axis shape: {tuple(axis.shape)}")

    print(f"[sweep] loading model: {args.model}")
    model, tokenizer = load_model_and_tokenizer(args.model)
    validate_axis(axis, model)

    print(f"[sweep] loading {args.dataset}/{args.split} n={args.n_examples}")
    examples = load_gsm8k_subset(args.n_examples, seed=args.seed, split=args.split)
    print(f"[sweep] got {len(examples)} examples")

    solver_gen = GenConfig(
        max_new_tokens=args.max_new_tokens_solver,
        temperature=args.temperature,
        do_sample=args.temperature > 0,
    )
    critic_gen = GenConfig(
        max_new_tokens=args.max_new_tokens_critic,
        temperature=args.temperature,
        do_sample=args.temperature > 0,
    )

    config = {
        "model": args.model,
        "model_short": args.model_short,
        "axis_repo": args.axis_repo,
        "layer": args.layer,
        "alphas": list(args.alphas),
        "dataset": args.dataset,
        "split": args.split,
        "n_examples": len(examples),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "max_new_tokens_solver": args.max_new_tokens_solver,
        "max_new_tokens_critic": args.max_new_tokens_critic,
        "temperature": args.temperature,
        "git_commit": git_commit(),
        "hardware": gpu_summary(),
    }
    dump_json(run_dir / "config.json", config)

    t0 = time.time()
    pairs = [(a_s, a_c) for a_s in args.alphas for a_c in args.alphas]
    print(f"[sweep] {len(pairs)} (α_s, α_c) pairs × {len(examples)} examples × 3 passes")

    for a_s, a_c in tqdm(pairs, desc="conditions"):
        pending = [ex for ex in examples if (a_s, a_c, ex.id) not in done]
        if not pending:
            continue
        transcripts = run_two_agent(
            pending,
            model,
            tokenizer,
            axis=axis,
            layer_idx=args.layer,
            alpha_solver=a_s,
            alpha_critic=a_c,
            solver_gen=solver_gen,
            critic_gen=critic_gen,
            batch_size=args.batch_size,
        )
        for tr in transcripts:
            row = per_example_row(tr, scoring)
            append_jsonl(jsonl_path, row)
            done.add((a_s, a_c, tr.example_id))
        torch.cuda.empty_cache()
        elapsed = time.time() - t0
        print(f"[sweep] (α_s={a_s}, α_c={a_c}) done; elapsed {elapsed/60:.1f} min")

    # Aggregate.
    rows: list[dict] = []
    with open(jsonl_path) as f:
        for line in f:
            rows.append(json.loads(line))

    per_cond = per_condition_metrics(rows)
    pd.DataFrame(per_cond).to_csv(run_dir / "per_condition_metrics.csv", index=False)

    overall = {
        "n_rows": len(rows),
        "best_final_accuracy": max((c["final_accuracy"] for c in per_cond), default=0.0),
        "baseline_unsteered": next(
            (c for c in per_cond if c["alpha_solver"] == 0 and c["alpha_critic"] == 0),
            None,
        ),
        "per_condition": per_cond,
        "elapsed_sec": time.time() - t0,
    }
    dump_json(run_dir / "metrics.json", overall)
    print(f"[sweep] DONE. Best final acc: {overall['best_final_accuracy']:.3f}; "
          f"unsteered: {overall['baseline_unsteered']['final_accuracy']:.3f} "
          f"({overall['n_rows']} rows in {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
