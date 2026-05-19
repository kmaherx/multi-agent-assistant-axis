# CLAUDE.md

## Project conventions

- Package manager: **uv** (`uv venv --system-site-packages && uv sync`). Never let uv replace the system torch — `[tool.uv].override-dependencies` pins it out. Use `uv add` / `uv run`, never bare `pip`.
- Working tree lives under `/workspace` (persistent). `/root` is transient.
- HuggingFace cache: `HF_HOME=/workspace/.cache/huggingface/`. Token is at `/workspace/.hf_token` — export as `HF_TOKEN` before any HF download.
- Verify environment claims directly: `nvidia-smi`, `df -h`, `uv pip list`. Do not trust summaries.

## Experiment workflow

- Plans go under `plans/NNN_<slug>.md` and are append-only (don't rewrite a plan after it's been run).
- After each experiment run, append a row to `PROGRESS.md`.
- Each run writes one directory under `outputs/<run_dir>/`:
  - `config.json` — model, axis, layer, grid, seed, gen kwargs, git commit, hardware
  - `metrics.json` — overall summary
  - `per_condition_metrics.csv` — one row per `(α_solver, α_critic)`
  - `per_example_results.jsonl` — full transcripts, streamed during the run
  - `baselines/*.json` — baseline results
- Dashboard is built into `dashboard/index.html` (static, deployable on GitHub Pages).

## Code style

- `src/` modules export functions; `scripts/` modules contain the CLI argparse + thin glue.
- Greedy generation (T=0) for reproducibility unless explicitly testing otherwise.
- Always set `torch.manual_seed`, `random.seed`, `np.random.seed` from a single seed.
- Store raw model output text in transcripts — never lose it to a parser bug.
