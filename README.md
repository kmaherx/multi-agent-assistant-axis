# multi-agent-assistant-axis

Two Llama-3.1-8B-Instruct agents in a **solver → critic → revision** loop on GSM8K, each steered along [Butanium's assistant-axis](https://huggingface.co/Butanium/llama-3.1-8b-instruct-assistant-axis) at independent coefficients `(α_solver, α_critic)`.

The experiment maps the 2D performance landscape `F(α_solver, α_critic)` and tests whether *heterogeneous* steering improves multi-agent QA over homogeneous or no steering.

**Interactive dashboard:** https://kmaherx.github.io/multi-agent-assistant-axis/

## Pilot result (n=50, GSM8K, layer 16)

| condition | final accuracy |
|---|---:|
| single-agent direct (no critic) | **86.0%** |
| 2-agent best @ (α_s=0, α_c=−2) and (α_s=0, α_c=+5) | 80.0% |
| 2-agent unsteered (α_s=0, α_c=0) | 78.0% |
| random-vector matched-norm @ α=(5,5) | 84.0% |
| random-vector matched-norm @ α=(−5,−5) | 74.0% |
| 2-agent assistant-axis @ α=(5,5) | 66.0% |
| single-agent self-revision (no critic, α=0) | 62.0% |

Two findings worth keeping:
1. The critic-then-revise loop is **net harmful** at this scale — the unsteered solver alone (86%) beats every 2-agent condition (best 80%).
2. The assistant-axis at strong positive α does **real semantic damage** — α=(5,5) drops to 66% on the actual axis but only to 84% on a random matched-norm vector.

## Quick start

```bash
export HF_HOME=/workspace/.cache/huggingface
export HF_TOKEN=$(cat /workspace/.hf_token)

uv venv --system-site-packages
uv sync

# Smoke test (1 example, α=(0,0))
uv run python scripts/smoke_test.py

# Full pilot sweep: 200 GSM8K examples × 5×5 α-grid
uv run python scripts/run_sweep.py \
  --n_examples 200 \
  --layer 16 \
  --alphas -5 -2 0 2 5 \
  --seed 42 \
  --batch_size 16

# Baselines: single-agent direct, self-revision, random-vector matched-norm
uv run python scripts/run_baselines.py \
  --run_dir outputs/gsm8k__llama31_8b__axis_layer16__seed42

# Build dashboard
uv run python scripts/build_dashboard.py \
  --run_dir outputs/gsm8k__llama31_8b__axis_layer16__seed42

# View locally
cd docs && python -m http.server 8080
```

## Repo layout

```
src/        axis hook, data, scoring, generation, protocols, metrics
scripts/    smoke_test, run_sweep, run_baselines, build_dashboard
docs/       static HTML/Plotly dashboard (served from GitHub Pages)
plans/      versioned experiment plans
outputs/    per-run results (transcripts + metrics)
```

See `plans/001_mvp_assistant_axis_qa.md` for the full experimental spec.
