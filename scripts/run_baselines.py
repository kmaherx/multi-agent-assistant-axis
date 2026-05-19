"""Baselines: single-agent direct, single-agent self-revision, random-vector matched-norm."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from src import scoring  # noqa: E402
from src.axis import load_axis, random_matched_norm_vector, validate_axis  # noqa: E402
from src.data import load_gsm8k_subset  # noqa: E402
from src.generate import GenConfig, load_model_and_tokenizer  # noqa: E402
from src.metrics import aggregate, per_example_row  # noqa: E402
from src.protocols import (  # noqa: E402
    run_single_agent_direct,
    run_single_agent_self_revision,
    run_two_agent,
)
from src.utils import dump_json, ensure_hf_token, set_seed  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="The existing sweep run dir to attach baselines to.")
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--axis_repo", default="Butanium/llama-3.1-8b-instruct-assistant-axis")
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--n_examples", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_new_tokens_solver", type=int, default=512)
    ap.add_argument("--max_new_tokens_critic", type=int, default=256)
    ap.add_argument(
        "--random_alpha_pairs",
        type=float,
        nargs="+",
        default=[5.0, 5.0, -5.0, -5.0],
        help="Flat list of (α_s, α_c) pairs for random-vector baseline.",
    )
    ap.add_argument(
        "--skip_random_vector",
        action="store_true",
        help="If set, skip the random-vector baseline (saves time).",
    )
    return ap.parse_args()


def _save(rows: list[dict], out_path: Path) -> dict:
    rows_clean = [{k: v for k, v in r.items()} for r in rows]
    agg = aggregate(rows_clean)
    out = {
        "n": agg["n"],
        "metrics": agg,
        "rows": rows_clean,
    }
    dump_json(out_path, out)
    return agg


def main():
    args = parse_args()
    ensure_hf_token()
    set_seed(args.seed)

    run_dir = Path(args.run_dir)
    bdir = run_dir / "baselines"
    bdir.mkdir(parents=True, exist_ok=True)

    print(f"[baselines] loading model {args.model}")
    model, tokenizer = load_model_and_tokenizer(args.model)

    examples = load_gsm8k_subset(args.n_examples, seed=args.seed, split="test")
    solver_gen = GenConfig(max_new_tokens=args.max_new_tokens_solver)
    critic_gen = GenConfig(max_new_tokens=args.max_new_tokens_critic)

    summary: dict = {}

    # 1) single-agent direct
    print("[baselines] single-agent direct")
    t = time.time()
    direct = run_single_agent_direct(examples, model, tokenizer, solver_gen, args.batch_size)
    rows = [per_example_row(tr, scoring) for tr in direct]
    summary["single_agent_direct"] = _save(rows, bdir / "single_agent_direct.json")
    print(f"  → final_acc={summary['single_agent_direct']['final_accuracy']:.3f} "
          f"({time.time()-t:.0f}s)")
    torch.cuda.empty_cache()

    # 2) single-agent self-revision
    print("[baselines] single-agent self-revision")
    t = time.time()
    sr = run_single_agent_self_revision(examples, model, tokenizer, solver_gen, args.batch_size)
    rows = [per_example_row(tr, scoring) for tr in sr]
    summary["single_agent_self_revision"] = _save(rows, bdir / "single_agent_self_revision.json")
    print(f"  → final_acc={summary['single_agent_self_revision']['final_accuracy']:.3f} "
          f"({time.time()-t:.0f}s)")
    torch.cuda.empty_cache()

    # 3) random-vector matched-norm @ specified pairs (uses the same protocol as sweep)
    if not args.skip_random_vector:
        print(f"[baselines] loading axis for random-norm reference: {args.axis_repo}")
        axis = load_axis(args.axis_repo)
        validate_axis(axis, model)
        reference = axis[args.layer] if axis.dim() == 2 else axis
        rand_vec = random_matched_norm_vector(reference, seed=args.seed)
        rand_axis = rand_vec.unsqueeze(0).expand(axis.shape).contiguous() if axis.dim() == 2 else rand_vec

        pairs = list(zip(args.random_alpha_pairs[0::2], args.random_alpha_pairs[1::2]))
        rand_summary = {}
        for a_s, a_c in pairs:
            print(f"[baselines] random-vector @ (α_s={a_s}, α_c={a_c})")
            t = time.time()
            transcripts = run_two_agent(
                examples,
                model,
                tokenizer,
                axis=rand_axis,
                layer_idx=args.layer,
                alpha_solver=a_s,
                alpha_critic=a_c,
                solver_gen=solver_gen,
                critic_gen=critic_gen,
                batch_size=args.batch_size,
            )
            rows = [per_example_row(tr, scoring) for tr in transcripts]
            key = f"alpha_s={a_s}_alpha_c={a_c}"
            sub = _save(rows, bdir / f"random_vector__{key}.json")
            rand_summary[key] = sub
            print(f"  → final_acc={sub['final_accuracy']:.3f} ({time.time()-t:.0f}s)")
            torch.cuda.empty_cache()
        summary["random_vector_matched_norm"] = rand_summary

    dump_json(bdir / "summary.json", summary)
    print(f"[baselines] DONE. summary at {bdir/'summary.json'}")


if __name__ == "__main__":
    main()
