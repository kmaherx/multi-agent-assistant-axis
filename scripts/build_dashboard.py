"""Build docs/index.html.

Writes:
  docs/data/index.json                — heatmaps, per_condition, baselines summary, cell index, run config
  docs/data/cells/{a_s}__{a_c}.json   — ALL transcripts for that condition (full text)
  docs/data/baselines/{name}.json     — ALL transcripts for each baseline run
  docs/data/smoke.json (optional)     — smoke-test transcripts if outputs/smoke/transcripts.jsonl exists
  docs/index.html                     — single page that fetches the above lazily
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jinja2 import Environment, FileSystemLoader, select_autoescape  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
TEMPLATES = REPO / "docs" / "templates"
DASHBOARD = REPO / "docs"
DATA = DASHBOARD / "data"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--smoke_jsonl", default=str(REPO / "outputs" / "smoke" / "transcripts.jsonl"))
    ap.add_argument("--out_html", default=str(DASHBOARD / "index.html"))
    ap.add_argument("--out_data_dir", default=str(DATA))
    return ap.parse_args()


def _load_rows(jsonl_path: Path) -> list[dict]:
    rows = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _heatmap_matrix(per_cond: list[dict], metric: str):
    alphas_s = sorted({c["alpha_solver"] for c in per_cond})
    alphas_c = sorted({c["alpha_critic"] for c in per_cond})
    by = {(c["alpha_solver"], c["alpha_critic"]): c for c in per_cond}
    z = [[by.get((a_s, a_c), {}).get(metric) for a_c in alphas_c] for a_s in alphas_s]
    return {"x": alphas_c, "y": alphas_s, "z": z, "metric": metric}


def _cell_key(a_s: float, a_c: float) -> str:
    return f"{a_s}__{a_c}"


def _cell_summary(rows: list[dict]) -> dict:
    """Quick counters used in the cell list view (without loading full text)."""
    return {
        "n": len(rows),
        "n_initial_correct": sum(1 for r in rows if r["initial_correct"]),
        "n_final_correct": sum(1 for r in rows if r["final_correct"]),
        "n_wrong_to_right": sum(1 for r in rows if not r["initial_correct"] and r["final_correct"]),
        "n_right_to_wrong": sum(1 for r in rows if r["initial_correct"] and not r["final_correct"]),
        "n_unchanged": sum(1 for r in rows if r["initial_correct"] == r["final_correct"]),
    }


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_data = Path(args.out_data_dir)
    (out_data / "cells").mkdir(parents=True, exist_ok=True)
    (out_data / "baselines").mkdir(parents=True, exist_ok=True)

    config = json.loads((run_dir / "config.json").read_text())
    metrics = json.loads((run_dir / "metrics.json").read_text())
    per_cond = metrics["per_condition"]
    sweep_rows = _load_rows(run_dir / "per_example_results.jsonl")

    # ---- heatmaps
    heatmaps = {
        m: _heatmap_matrix(per_cond, m)
        for m in (
            "final_accuracy",
            "initial_accuracy",
            "wrong_to_right_rate",
            "right_to_wrong_rate",
            "answer_change_rate",
            "final_parse_failure_rate",
        )
    }

    # ---- group sweep rows by cell, write per-cell file with ALL transcripts
    cells_by_key: dict[str, list[dict]] = defaultdict(list)
    for r in sweep_rows:
        key = _cell_key(r["alpha_solver"], r["alpha_critic"])
        cells_by_key[key].append(r)

    cell_index = []
    for key, rows in sorted(cells_by_key.items()):
        a_s, a_c = key.split("__")
        # Sort each cell's rows by example_id for stable browsing.
        rows = sorted(rows, key=lambda r: r["example_id"])
        cell_path = out_data / "cells" / f"{key}.json"
        cell_path.write_text(json.dumps({
            "alpha_solver": float(a_s),
            "alpha_critic": float(a_c),
            "rows": rows,
        }))
        cell_index.append({
            "key": key,
            "alpha_solver": float(a_s),
            "alpha_critic": float(a_c),
            "summary": _cell_summary(rows),
            "file": f"data/cells/{key}.json",
        })

    # ---- baselines: write each baseline's rows to its own file
    baseline_index = {}
    bdir = run_dir / "baselines"
    if bdir.exists():
        for f in sorted(bdir.glob("*.json")):
            if f.name == "summary.json":
                continue
            obj = json.loads(f.read_text())
            metrics_obj = obj.get("metrics", {})
            rows = obj.get("rows", [])
            bpath = out_data / "baselines" / f.name
            bpath.write_text(json.dumps({"metrics": metrics_obj, "rows": rows}))
            baseline_index[f.stem] = {
                "name": f.stem,
                "metrics": metrics_obj,
                "n": len(rows),
                "file": f"data/baselines/{f.name}",
            }

    # ---- smoke
    smoke_path = Path(args.smoke_jsonl)
    smoke_info = None
    if smoke_path.exists():
        smoke_rows = _load_rows(smoke_path)
        smoke_file = out_data / "smoke.json"
        smoke_file.write_text(json.dumps({"rows": smoke_rows}))
        smoke_info = {"n": len(smoke_rows), "file": "data/smoke.json"}

    # ---- summary
    best = max(per_cond, key=lambda c: c["final_accuracy"]) if per_cond else None
    unsteered = next(
        (c for c in per_cond if c["alpha_solver"] == 0 and c["alpha_critic"] == 0),
        None,
    )
    summary = {
        "best": best,
        "unsteered": unsteered,
        "n_rows": metrics.get("n_rows"),
    }

    index_payload = {
        "config": config,
        "per_condition": per_cond,
        "heatmaps": heatmaps,
        "cell_index": cell_index,
        "baselines": baseline_index,
        "smoke": smoke_info,
        "summary": summary,
    }
    (out_data / "index.json").write_text(json.dumps(index_payload, indent=2))
    print(f"[docs] wrote {out_data/'index.json'} (+ {len(cell_index)} cell files, "
          f"{len(baseline_index)} baseline files)")

    env = Environment(
        loader=FileSystemLoader(TEMPLATES),
        autoescape=select_autoescape(enabled_extensions=("html", "j2")),
    )
    tpl = env.get_template("index.html.j2")
    html = tpl.render(repo_url="https://github.com/kmaherx/multi-agent-assistant-axis")
    Path(args.out_html).write_text(html)
    print(f"[docs] wrote {args.out_html}")


if __name__ == "__main__":
    main()
