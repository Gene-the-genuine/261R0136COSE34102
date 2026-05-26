# RF390: A Context-Aware Task Feasibility Benchmark for SLMs under SLO-constraints

This repository accompanies the paper *"RF390: A Context-Aware Task Feasibility Benchmark for SLMs under SLO-constraints"* and contains the full
benchmark harness, scoring policy, experiments, and raw run traces used to
produce the reported results.

The core idea: under a **3-stage time-budget**, an SLM must choose one of four
actions per turn — `SOLVE`, `REJECT`, `ASK`, `THINK`. The harness collects
raw I/O; a separate scoring policy assigns category-based scores penalised by
token usage.

```
prompt → State 0 → call → action token
                            ├─ SOLVE / REJECT      → End
                            ├─ ASK   → user simulator (codex) → State 1
                            └─ THINK → "계속."                 → State 1
State 1: same transitions
State 2: ANY action → End
```

Each call is **two-phase** to the vLLM OpenAI-compatible API:
- phase 1: `max_tokens = stage_max - stage_margin` (free reasoning)
- phase 2: if phase 1 hit the length cap, prefill `\n >>> Action :` and emit
  one bracketed action within `margin` tokens.

State transition trigger: action token emit **or** token cap reached.

---

## Repository layout

```
RF390/
├── README.md                # this file
├── LICENSE                  # MIT
├── requirements.txt
├── .gitignore
├── serve.sh                 # vLLM dispatch (6 model presets)
│
├── slm_bench/               # core harness (Python package)
│   ├── pyproject.toml
│   ├── configs/             # 8 system prompts + model_presets.json
│   ├── src/slm_bench/       # harness.py, parser.py, simulator.py
│   ├── scripts/             # run_harness.py, render_html.py
│   └── fixtures/
│       ├── RF390_dataset.jsonl                       # main benchmark (390)
│       └── hf/                                       # GSM8K, MMLU subjects
│
├── scoring/                 # scoring + analysis tools
│   ├── criteria.md          # scoring policy specification
│   ├── score.py             # RF criteria scorer (4 categories, [0,1] piecewise)
│   ├── score_accuracy.py    # GSM8K/MMLU accuracy scorer
│   ├── compare.py           # n-way comparison
│   ├── build_hf_harness.py  # HF datasets → harness input
│   ├── build_presets.py     # measured rates → model_presets.json
│   └── measure_rates.py     # vLLM throughput probe
│
├── experiments/             # EXP1–5 runner + analyzer scripts
│   ├── exp1_chain.sh        # EXP1: 7 cells × Qwen3-4B (RF390 × 3 labels + MMLU × 3 + GSM8K)
│   ├── exp1_report.py
│   ├── exp2_analysis.py     # EXP2: final-action state + tokens distribution
│   ├── exp3_chain.sh        # EXP3: 8 system prompts × RF390
│   ├── exp3_analysis.py
│   ├── exp4_phaseA.sh       # EXP4-A: 5 models × {×1, ×0.5}
│   ├── exp4_phaseA_report.py
│   ├── exp4_phaseB.sh       # EXP4-B: Qwen3-4B × 7 max_tokens scales
│   ├── exp4_phaseB_report.py
│   ├── exp5_chain.sh        # EXP5: Llama-3.2-3B + reuses EXP4 for the other 5
│   └── exp5_report.py
│
└── results/                 # raw outputs (for reproducibility verification)
    ├── runs/                # one JSONL per (experiment, cell)
    ├── reports/             # per-experiment JSON summaries
    └── derived_fixtures/    # GSM8K/MMLU converted to harness-input format
```

---

## Quick start

### Install

```bash
# 1) Python virtualenv (we use uv; substitute pip if you prefer)
cd RF390
uv venv .venv
source .venv/bin/activate
uv pip install -e ./slm_bench
uv pip install -r requirements.txt

# 2) Download models (see "Models" section below). Llama-3.2 is gated:
#    https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
hf auth login

# 3) Launch vLLM with a model preset (see serve.sh for the list)
bash serve.sh qwen3-4b   # or: gemma | qwen3-0.6b | qwen3-1.7b | qwen3-4b
                         #     llama-3.2-3b | phi-3.5-mini
```

### Smoke test — 1 prompt

```bash
head -1 slm_bench/fixtures/RF390_dataset.jsonl > /tmp/one.jsonl
python slm_bench/scripts/run_harness.py \
  --model qwen3-4b --preset qwen3-4b \
  --system-prompt slm_bench/configs/system_prompt_default.txt \
  --inputs /tmp/one.jsonl \
  --output /tmp/smoke.jsonl \
  --chat-template-kwargs '{"enable_thinking":false}' \
  --simulator codex --concurrency 1
```

### Visualize a run

```bash
python slm_bench/scripts/render_html.py /tmp/smoke.jsonl
# → /tmp/smoke.html (self-contained, open in a browser)
```

### Score a run (RF criteria)

```bash
python scoring/score.py /tmp/smoke.jsonl \
  --fixture slm_bench/fixtures/RF390_dataset.jsonl
```

### Reproduce a paper experiment

```bash
bash experiments/exp5_chain.sh   # runs Llama-3.2-3B + reuses cached EXP4 results
                                 # then prints the 6-model comparison table
```

---

## Models

Set the preset's local path inside `serve.sh` after downloading. The defaults
used in the paper:

| preset | HF id | quant | notes |
|---|---|---|---|
| `gemma4-E2B-it` | `google/gemma4-E2B-it` | NVFP4 | requires NVFP4-aware vLLM |
| `qwen3-0.6b` | `Qwen/Qwen3-0.6B` | bf16 | very fast |
| `qwen3-1.7b` | `Qwen/Qwen3-1.7B` | bf16 | |
| `qwen3-4b` | `Qwen/Qwen3-4B` | bf16 | **paper default** |
| `llama-3.2-3b` | `meta-llama/Llama-3.2-3B-Instruct` | bf16 | gated |
| `phi-3.5-mini` | `microsoft/Phi-3.5-mini-instruct` | bf16 | |

Per-model `max_tokens` / `margin` derived from measured throughput live in
`slm_bench/configs/model_presets.json`. Re-measure with:

```bash
python scoring/measure_rates.py --model qwen3-4b --out /tmp/rate_qwen3-4b.json
python scoring/build_presets.py /tmp/rate_*.json \
  --out slm_bench/configs/model_presets.json
```

---

## CLI parameters

| arg | meaning | example |
|---|---|---|
| `--model` | vLLM `--served-model-name` | `qwen3-4b` |
| `--preset` | per-model defaults from `model_presets.json` (max_tokens + margin) | `qwen3-4b` |
| `--base-url` | OpenAI-compatible endpoint | `http://127.0.0.1:8000/v1` |
| `--system-prompt` | path to the system-prompt text file | `slm_bench/configs/system_prompt_default.txt` |
| `--max-tokens S0 S1 S2` | per-stage `max_tokens` (overrides preset) | `95 140 230` |
| `--margin S0 S1 S2` | per-stage SUFFIX-injection reserve (overrides preset) | `30 30 50` |
| `--slos S0 S1 S2` | cumulative wall-clock SLO seconds | `2.0 5.0 10.0` |
| `--inputs` | input JSONL (each row: `{id, prompt}`) | `slm_bench/fixtures/RF390_dataset.jsonl` |
| `--output` | output JSONL path | `results/runs/run_<ts>.jsonl` |
| `--simulator` | ASK responder backend | `codex` |
| `--codex-model`, `--codex-effort` | codex CLI options | `gpt-5.4`, `medium` |
| `--temperature` | sampling temperature | `0.0` |
| `--chat-template-kwargs` | JSON forwarded to vLLM (e.g. Qwen3 thinking) | `'{"enable_thinking":false}'` |
| `--concurrency` | parallel prompts | `4` |
| `--limit` | 0 = all rows | `0` |

### `max_tokens` / `margin` intuition

```
stage_budget_tokens ≈ slo_seconds × decode_TPS × safety
margin ≥ 30                    # action header + 1-line justification reserve
max_tokens = stage_budget_tokens   # phase1 + phase2 combined
```

Recommended via measurement: see `scoring/measure_rates.py` and
`scoring/build_presets.py`.

---

## Output JSONL schema

One row per prompt = full trace:

```json
{
  "id": "ADLQ-0001",
  "prompt": "Context: ...\n\nUser input: ...",
  "config": {
    "model": "qwen3-4b",
    "system_prompt_path": "...",
    "system_prompt_sha": "abc123def4567890",
    "max_tokens": [154, 234, 390],
    "margin": [30, 30, 50],
    "slos_cumulative_s": [2.0, 5.0, 10.0],
    "stage_budgets_s": [2.0, 3.0, 5.0],
    "simulator_backend": "codex",
    "temperature": 0.0,
    "chat_template_kwargs": {"enable_thinking": false}
  },
  "turns": [
    {
      "state": 0,
      "messages_in": [{"role":"system","content":"..."}, {"role":"user","content":"..."}],
      "phase1": {"raw_text":"...", "finish_reason":"length", "output_tokens":65, "request_max_tokens":124, "latency_ms":845.2},
      "phase2": {"raw_text":" [THINK]\n...", "finish_reason":"stop", "output_tokens":12, "request_max_tokens":30, "latency_ms":312.7},
      "full_assistant_text": "Reasoning: ...\n >>> Action : [THINK]\n...",
      "action_detected": "THINK",
      "next_user_msg": "계속.",
      "simulator_call": null,
      "elapsed_ms": 1158.0
    }
    // turn k≥1: messages_in is a DELTA — only the new user msg.
    // Use slm_bench.harness.reconstruct_messages_in(run, k) to rebuild full history.
  ],
  "end_reason": "solve_action",      // or reject_action | state2_terminal_<action> | error
  "total_latency_ms": 8378.6,
  "started_at": "2026-05-26T...",
  "error": null
}
```

---

## Scoring policy

See [`scoring/criteria.md`](scoring/criteria.md) for the formal specification.
Summary:

- 22 possible 1–3-step trajectories map to 4 categories: `gt`, `possible`,
  `delayed`, `wrong`. Base scores `+1.0 / +0.5 / −1.0 / −2.0`.
- Token penalty `0.4 × (1 − exp(−x))` with `x = consumed / (sum(max_tokens) / 2)`.
- Subtracted from base, then **piecewise-linearly rescaled** to `[0, 1]`
  (raw 0 → 0.5; positive 1 unit ≈ 0.5; negative side compressed 1:2.4 to
  preserve the heavier penalty on wrong actions).
- Final score = average over non-error prompts; error rate reported separately.

The RF390 fixture is the union of four sub-domains (autodrive logic QA,
finlogiqa, hdfs-finlogiqa, and a small Korean-reasoning code set), all
re-labelled to `{ask, reject, solve}`.

---

## Reproducing the paper experiments

Each experiment has a chain script (`experiments/exp*.sh`) that runs the
necessary cells and an analyzer (`experiments/exp*_report.py`) that prints
the table. Outputs land under `results/runs/` and `results/reports/`.

| exp | what | model | wall-clock (approx.) |
|---|---|---|---|
| EXP1 | 7-cell GT% (RF390 × 3 labels + MMLU × 3 + GSM8K) | Qwen3-4B | ~35 min |
| EXP2 | final-action state + tokens distribution (from EXP1 traces) | — | seconds |
| EXP3 | 8 system_prompts × RF390 | Qwen3-4B | ~50 min |
| EXP4 | max_tokens sensitivity. Phase A: 5 models × {×1, ×0.5}; Phase B: Qwen3-4B × 7 scales | Qwen3-4B + 4 | ~2 h |
| EXP5 | 6-model comparison on RF390 (reuses EXP4 cached cells) | Llama + 5 reused | ~10 min |

Pre-generated outputs are checked in under `results/`; running a chain
overwrites the matching files.

---

## Dependencies

- **External**: vLLM OpenAI-compatible endpoint (`http://127.0.0.1:8000/v1`)
- **External**: Codex CLI (`codex` binary, `gpt-5.4` access) — ASK responder
- **Python**: see `requirements.txt`

---

## Known design decisions / caveats

1. **Stage transition counts model calls, not codex latency**. The simulator's
   wall-clock isn't attributed to a stage budget.
2. **No SLO timeout enforcement inside the harness** — `max_tokens` is the
   physical cap; the SLO is informational and used only by the scoring policy.
3. **`THINK` not blocked at State 2** — it's recorded as
   `end_reason=state2_terminal_think` and the trace ends.
4. **Simulator depends on external LLM**. If codex hangs, the harness applies
   per-prompt timeouts inherited from the chain script (`timeout 1800`).
5. **Ground truth is downstream**. The harness records only model action; the
   scorer needs a fixture with `label` (RF) or `task/answer` (HF) to grade.

## License

MIT — see [LICENSE](LICENSE).
