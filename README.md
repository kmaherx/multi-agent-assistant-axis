# multi-agent-assistant-axis

Two Llama-3.1-8B-Instruct agents in a **solver → critic → revision** loop on GSM8K, each steered along [Butanium's assistant-axis](https://huggingface.co/Butanium/llama-3.1-8b-instruct-assistant-axis) at independent coefficients `(α_solver, α_critic)`.

The experiment maps the 2D performance landscape `F(α_solver, α_critic)` and tests whether *heterogeneous* steering improves multi-agent QA over homogeneous or no steering.

**Interactive dashboard:** https://kmaherx.github.io/multi-agent-assistant-axis/

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
cd dashboard && python -m http.server 8080
```

## Repo layout

```
src/        axis hook, data, scoring, generation, protocols, metrics
scripts/    smoke_test, run_sweep, run_baselines, build_dashboard
plans/      versioned experiment plans
outputs/    per-run results (transcripts + metrics)
dashboard/  static HTML/Plotly dashboard (GitHub Pages)
```

See `plans/001_mvp_assistant_axis_qa.md` for the full experimental spec.
