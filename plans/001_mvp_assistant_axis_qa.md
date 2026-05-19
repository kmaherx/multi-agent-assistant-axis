# plan.md — Multi-agent QA assistant-axis steering

## Goal

Build a small, reproducible experiment to test whether steering two Llama-3.1-8B-Instruct agents along the assistant-axis changes collective QA performance.

Core object:

\[
F(\alpha_\text{solver}, \alpha_\text{critic})
\]

where \(\alpha_\text{solver}\) and \(\alpha_\text{critic}\) are activation-steering coefficients applied to the two agents in a solver → critic → revision protocol.

The first success criterion is **not** SOTA QA. It is a clean, inspectable 2D performance landscape showing whether heterogeneous steering improves multi-agent QA.

---

## External resources

Use:

- Model: `meta-llama/Llama-3.1-8B-Instruct`
- Assistant-axis vector: `Butanium/llama-3.1-8b-instruct-assistant-axis`
  - file: `assistant_axis.pt`
  - shape: `[num_layers, hidden_dim]`
  - recommended initial layer: `16`
  - positive coefficient = more default assistant-like behavior
  - negative coefficient = more role-play / character-compliant behavior
- Environment/protocol conventions: `https://github.com/kmaherx/bio-refusal-classifier`
  - use `uv`
  - put project under `/workspace`
  - set `HF_HOME=/workspace/.cache/huggingface/`
  - treat `/workspace` as persistent and `/root` as transient
  - archive plans under `plans/NNN_*.md`
  - keep `PROGRESS.md` updated after experiment runs
  - verify environment claims directly with `nvidia-smi`, `df -h`, `uv pip list`, etc.

---

## Main experiment

### Protocol

For each QA example:

1. **Solver pass**
   - Agent A receives the question.
   - Steering coefficient: \(\alpha_A\).
   - Produces an initial answer.

2. **Critic pass**
   - Agent B receives the question and Agent A's answer.
   - Steering coefficient: \(\alpha_B\).
   - Produces a critique / verification.

3. **Revision pass**
   - Agent A receives the question, initial answer, and critique.
   - Steering coefficient: \(\alpha_A\).
   - Produces the final answer.

4. **Scoring**
   - Extract final answer.
   - Compare against gold answer.
   - Record per-example transcript and metrics.

### Initial dataset

Use GSM8K first.

Reasons:

- exact-answer scoring
- small enough for cheap sweeps
- familiar failure modes
- good for diagnosing wrong-to-right vs right-to-wrong corrections

Start with `n=200` test examples for the pilot. Then scale to full test split if the pipeline works.

### Initial coefficient grid

Use a small grid first:

```python
alphas = [-5, -2, 0, 2, 5]
```

This gives 25 conditions. For 200 examples, this is:

```text
200 examples × 25 coefficient pairs × 3 generations = 15,000 generations
```

If this is too slow, start with:

```python
alphas = [-5, 0, 5]
```

Then expand.

### Steering layer

Start with layer 16, matching the dataset card recommendation.

Later sweep:

```python
layers = [8, 12, 16, 20, 24]
```

But do not mix layer sweeps into the MVP unless the basic coefficient landscape is stable.

---

## Prompts

Keep prompts boring and stable. Avoid persona language.

### Solver prompt

```text
Solve the following math word problem. Think step by step, then give the final answer on a separate line as:

Final answer: <answer>

Problem:
{question}
```

### Critic prompt

```text
You are checking another model's solution.

Problem:
{question}

Proposed solution:
{initial_answer}

Check the solution carefully. Identify any mistake if present. If the solution is correct, say it is correct. Be concise.
```

### Revision prompt

```text
You previously solved this problem and received a critique.

Problem:
{question}

Your initial solution:
{initial_answer}

Critique:
{critique}

Revise your solution if needed. Think step by step, then give the final answer on a separate line as:

Final answer: <answer>
```

Important: the steering intervention supplies the “role” signal. Do not confound the experiment with colorful role prompts.

---

## Metrics

Primary:

- `final_accuracy`: final revised answer exact-match accuracy

Secondary:

- `initial_accuracy`
- `wrong_to_right_rate`: among initially wrong examples, fraction fixed by final answer
- `right_to_wrong_rate`: among initially correct examples, fraction broken by final answer
- `answer_change_rate`
- `critique_length_tokens`
- `final_length_tokens`
- `parse_failure_rate`

Useful conditional metrics:

```text
P(final correct | initial wrong)
P(final wrong | initial correct)
P(answer changed | initial wrong)
P(answer changed | initial correct)
```

These matter more than raw final accuracy because the central question is whether the critic/revision loop helps without corrupting correct answers.

---

## Expected outputs

Each run should write one directory:

```text
outputs/
  gsm8k__llama31_8b__axis_layer16__seed42/
    config.json
    metrics.json
    per_condition_metrics.csv
    per_example_results.jsonl
    heatmap_final_accuracy.png
    heatmap_wrong_to_right.png
    heatmap_right_to_wrong.png
    heatmap_answer_change.png
```

### `config.json`

Include:

- model name
- assistant-axis repo/path
- layer index
- coefficient grid
- dataset name/split/count
- seed
- generation kwargs
- hardware summary
- git commit if available

### `per_example_results.jsonl`

One row per `(example, alpha_solver, alpha_critic)`:

```json
{
  "example_id": "...",
  "question": "...",
  "gold_answer": "...",
  "alpha_solver": 0,
  "alpha_critic": 5,
  "initial_text": "...",
  "critique_text": "...",
  "final_text": "...",
  "initial_parsed": "...",
  "final_parsed": "...",
  "initial_correct": false,
  "final_correct": true,
  "answer_changed": true
}
```

### `per_condition_metrics.csv`

One row per `(alpha_solver, alpha_critic)` with aggregate metrics.

---

## Repository structure

Use a simple modular layout:

```text
multi-agent-assistant-axis/
  CLAUDE.md
  PROGRESS.md
  README.md
  pyproject.toml
  plans/
    001_mvp_assistant_axis_qa.md
  scripts/
    run_sweep.py
    smoke_test.py
    make_plots.py
  src/
    __init__.py
    axis.py
    data.py
    generate.py
    protocols.py
    scoring.py
    metrics.py
    plotting.py
    utils.py
  outputs/
    .gitkeep
```

### `src/axis.py`

Responsibilities:

- download/load `assistant_axis.pt`
- validate shape
- normalize vector if needed
- provide steering hook/context manager

Implementation target:

```python
with activation_steering(
    model=model,
    vector=axis[layer_idx],
    layer_idx=layer_idx,
    coefficient=alpha,
):
    output = generate(...)
```

Prefer a lightweight custom hook if straightforward. Otherwise use the `assistant_axis` library from the dataset card, but keep the wrapper isolated so it can be swapped later.

### `src/data.py`

Responsibilities:

- load GSM8K
- select deterministic subset
- return records with `id`, `question`, `answer`, `gold_number`

### `src/scoring.py`

Responsibilities:

- extract `Final answer: ...`
- normalize numbers
- compare to GSM8K gold answer
- flag parse failures

Keep parsing simple first. Store raw text so parser bugs can be fixed retrospectively.

### `src/protocols.py`

Responsibilities:

- implement solver → critic → revision loop
- ensure the same `alpha_solver` is used for initial and final solver passes
- return full transcript

### `src/metrics.py`

Responsibilities:

- aggregate per-example rows into per-condition metrics
- compute wrong-to-right and right-to-wrong rates

### `src/plotting.py`

Responsibilities:

- heatmaps over \((\alpha_\text{solver}, \alpha_\text{critic})\)
- one metric per plot
- fixed axes/order across plots

---

## Environment setup on RunPod

Assume a fresh pod.

```bash
cd /workspace
git clone <NEW_REPO_URL> multi-agent-assistant-axis
cd multi-agent-assistant-axis

export HF_HOME=/workspace/.cache/huggingface
export HF_TOKEN=<your_token_if_needed>

uv venv --system-site-packages
uv sync
```

Use `uv` for all package operations:

```bash
uv add transformers datasets accelerate huggingface-hub pandas numpy matplotlib tqdm
uv add --dev ruff pytest
```

If using an RTX 5090 / Blackwell pod, do not let uv replace system torch. Mirror the prior repo pattern in `pyproject.toml`:

```toml
[tool.uv]
override-dependencies = ["torch ; sys_platform == 'never'"]
```

Verify before running:

```bash
nvidia-smi
df -h
uv pip list | grep -E "torch|transformers|datasets|accelerate"
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
PY
```

---

## Run commands

### Smoke test

Run one example and one coefficient pair:

```bash
uv run python scripts/smoke_test.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --axis_repo Butanium/llama-3.1-8b-instruct-assistant-axis \
  --layer 16 \
  --alpha_solver 0 \
  --alpha_critic 0
```

This should print:

- loaded model
- loaded axis shape
- one full transcript
- parsed final answer
- correctness

### Pilot sweep

```bash
uv run python scripts/run_sweep.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --axis_repo Butanium/llama-3.1-8b-instruct-assistant-axis \
  --dataset gsm8k \
  --split test \
  --n_examples 200 \
  --layer 16 \
  --alphas -5 -2 0 2 5 \
  --seed 42 \
  --max_new_tokens_solver 512 \
  --max_new_tokens_critic 256 \
  --temperature 0.0
```

### Plot only

```bash
uv run python scripts/make_plots.py \
  --run_dir outputs/gsm8k__llama31_8b__axis_layer16__seed42
```

---

## Analysis questions

After the pilot, answer:

1. Is unsteered two-agent QA better than single-agent?
2. Does steering the solver help?
3. Does steering the critic help?
4. Is the optimum homogeneous, e.g. \((+,+)\), or heterogeneous, e.g. \((+,0)\), \((0,+)\), \((+,-)\)?
5. Does positive assistantness increase correctness, or mainly reduce weird role-play?
6. Does negative steering ever help the critic by making it less assistant-like / less agreeable?
7. Is any gain driven by wrong-to-right corrections, or by parser/format effects?
8. Does oversteering increase right-to-wrong corruption?

---

## Hypotheses

Likely outcomes:

- Moderate positive solver steering may improve answer formatting and reduce weirdness.
- Moderate positive critic steering may improve concise verification.
- Very positive steering may increase deference/sycophancy and reduce useful criticism.
- Negative critic steering may sometimes catch errors better, but may also become chaotic or less reliable.
- Best pair may be heterogeneous rather than both maximally positive.

Do not overinterpret one seed or 200 examples. The pilot is for signal detection.

---

## Failure modes

- Steering hook hits the wrong module name for Llama 3.1.
- Axis dtype/device mismatch.
- Prompt formatting differs from Llama chat template expectations.
- Greedy generation produces brittle outputs.
- Exact-answer parser fails too often.
- Coefficient scale is too weak or too strong.
- Multi-agent gains vanish when compared against a single-agent self-revision baseline.
- Critic mostly agrees and adds tokens without information.
- Outputs are dominated by formatting changes rather than reasoning changes.

Mitigations:

- Add a no-critic self-revision baseline.
- Add parse-failure metrics.
- Inspect transcripts from each heatmap corner.
- Try coefficient grid `[-10, -5, 0, 5, 10]` if no effect.
- Try layer sweep only after MVP works.
- Keep all raw transcripts.

---

## Baselines

Minimum baselines:

1. Single-agent direct answer, no steering.
2. Single-agent self-revision, no steering.
3. Two-agent solver/critic/revision, no steering.
4. Two-agent solver/critic/revision, coefficient grid.

Optional but useful:

5. Same number of generated tokens as two-agent, but no critic.
6. Critic prompt replaced by generic “think again” prompt.
7. Random steering vector with matched norm.

The random-vector baseline is important if the assistant-axis appears to help. It tests whether improvement is semantically meaningful or just activation noise/regularization.

---

## Documentation protocol

Follow the prior repo conventions:

- Put this plan under `plans/001_mvp_assistant_axis_qa.md`.
- Maintain `PROGRESS.md`.
- After each run, append:
  - timestamp
  - run directory
  - model
  - dataset/count
  - layer
  - coefficient grid
  - best condition
  - baseline condition
  - main metrics
  - notable failures/surprises
- Do not trust environment summaries. Verify directly.

Example `PROGRESS.md` row:

```markdown
| Timestamp | Run | Model | Data | Layer | Grid | Best final acc | Baseline final acc | Notes |
|---|---|---|---|---:|---|---:|---:|---|
| 2026-05-19 22:10 | outputs/gsm8k__... | Llama-3.1-8B-Instruct | GSM8K test n=200 | 16 | -5,-2,0,2,5 | TBD | TBD | smoke/pilot |
```

---

## MVP definition of done

The MVP is complete when there is:

1. A working RunPod setup using `uv`.
2. Successful model + assistant-axis loading.
3. One smoke-test transcript with steering enabled.
4. A 200-example GSM8K sweep over at least a 3×3 coefficient grid.
5. `per_example_results.jsonl` with full transcripts.
6. `per_condition_metrics.csv`.
7. Heatmaps for final accuracy, wrong-to-right, right-to-wrong, and answer-change rate.
8. `PROGRESS.md` updated with the first result summary.

---

## Stretch goals

Only after MVP:

1. Full GSM8K test split.
2. Layer sweep.
3. Add MATH or MMLU/GPQA-style multiple choice.
4. Add criticality/sycophancy steering vectors.
5. Train CSPs to imitate useful steering regions.
6. Compare CSP self-verbalization against steering-induced behavior.
