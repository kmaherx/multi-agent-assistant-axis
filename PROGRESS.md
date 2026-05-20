# PROGRESS

Append one row per experiment run. Most recent at the bottom.

| Timestamp | Run dir | Model | Dataset | Layer | Grid | Best final acc | Baseline (α=0,0) | Notes |
|---|---|---|---|---:|---|---:|---:|---|
| 2026-05-20 00:09 | `outputs/gsm8k__llama31_8b__axis_layer16__seed42` | Llama-3.1-8B-Instruct | GSM8K test n=50 | 16 | -5,-2,0,2,5 | 80.0% @ (0,−2) and (0,+5) | 78.0% | 30 min sweep + 4 min baselines. **Single-agent direct = 86%** beats every 2-agent condition; self-revision drops to 62%; random-vector at α=(5,5) hits 84% while assistant-axis at (5,5) collapses to 66%. The critic/revision loop is net harmful here — the strongest signal is that assistant-axis at strong positive α does *real* semantic damage (random-norm vector does not). |
