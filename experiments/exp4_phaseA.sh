#!/usr/bin/env bash
# EXP4 Phase A — 5 models × 2 scales (×1, ×0.5). Qwen3-4B ×1 skipped
# (reuse EXP1 result by copy). margin scaled with max_tokens for x<1.
set -u
cd "$(dirname "$0")/.."
PYBIN=.venv/bin/python
LOG=/tmp/exp4_phaseA.log
echo "=== $(date +%Y-%m-%dT%H:%M:%S%z) EXP4 Phase A START ===" | tee -a "$LOG"

# Reuse EXP1 Qwen3-4B preset result (×1) as exp4 output
cp results/runs/run_exp1_rf390_solve.jsonl   /tmp/_rf_solve.tmp 2>/dev/null || true
# Build a single RF390 run from EXP1 by concat of 3 labels (EXP1 was split by label)
$PYBIN <<'PY'
import json
out = []
for lab in ("solve","ask","reject"):
    with open(f"results/runs/run_exp1_rf390_{lab}.jsonl") as f:
        for line in f:
            out.append(line.rstrip())
with open("results/runs/run_exp4_qwen3-4b_x1.0.jsonl","w") as f:
    f.write("\n".join(out) + "\n")
print(f"reused EXP1 → run_exp4_qwen3-4b_x1.0.jsonl: {len(out)} rows")
PY

scaled_tokens() {
  # $1 = "a b c"  $2 = factor (float)
  echo "$1" | awk -v f="$2" '{for(i=1;i<=NF;i++) printf "%d ", int($i*f+0.5); print ""}' | sed 's/ *$//'
}

# preset max_tokens (model presets, margin all 30/30/50 → defined in presets.json)
declare -A PRESETS
PRESETS[gemma]="106 162 271"
PRESETS[qwen3-0.6b]="354 535 891"
PRESETS[qwen3-1.7b]="324 490 817"
PRESETS[qwen3-4b]="154 234 390"
PRESETS[phi-3.5-mini]="171 260 434"

# served-model-name (vLLM --served-model-name), if different from preset key
declare -A SERVED
SERVED[gemma]="gemma-4-E2B-it"

run_one() {
  local model=$1
  local scale=$2
  local out="results/runs/run_exp4_${model}_x${scale}.jsonl"
  if [ -f "$out" ] && [ "$(wc -l < "$out")" = "390" ]; then
    echo "SKIP $model x$scale (exists 390 rows)" | tee -a "$LOG"
    return 0
  fi
  local base_max="${PRESETS[$model]}"
  local mt=$(scaled_tokens "$base_max" "$scale")
  # margin: scale with max if scale<1, else keep preset (30 30 50)
  local mg
  if awk "BEGIN{exit !($scale<1)}"; then
    mg=$(scaled_tokens "30 30 50" "$scale")
  else
    mg="30 30 50"
  fi
  # chat_template_kwargs for Qwen3
  local ctk_args=()
  if [[ "$model" == qwen3-* ]]; then
    ctk_args=(--chat-template-kwargs '{"enable_thinking":false}')
  fi
  local served="${SERVED[$model]:-$model}"
  echo "=== $(date +%H:%M:%S) START $model x$scale max=[$mt] margin=[$mg] served=$served ===" | tee -a "$LOG"
  timeout 1800 $PYBIN slm_bench/scripts/run_harness.py \
    --model "$served" \
    --system-prompt slm_bench/configs/system_prompt_default.txt \
    --max-tokens $mt --margin $mg \
    --slos 2.0 5.0 10.0 \
    --inputs slm_bench/fixtures/RF390_dataset.jsonl \
    --output "$out" \
    "${ctk_args[@]}" \
    --simulator codex --concurrency 4 2>&1 | tail -3 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  local n=$(wc -l < "$out" 2>/dev/null || echo 0)
  echo "DONE $model x$scale (rc=$rc, $n/390) @ $(date +%H:%M:%S)" | tee -a "$LOG"
}

for model in gemma qwen3-0.6b qwen3-1.7b qwen3-4b phi-3.5-mini; do
  echo "" | tee -a "$LOG"
  echo "=== $(date +%H:%M:%S) vLLM $model UP ===" | tee -a "$LOG"
  nohup bash serve.sh "$model" > "/tmp/vllm-$model.log" 2>&1 &
  vpid=$!
  ready=0
  for i in $(seq 1 120); do
    if curl -s -m 2 http://127.0.0.1:8000/v1/models 2>/dev/null | grep -q '"id"'; then
      ready=1; break
    fi
    sleep 2
  done
  if [ $ready -eq 0 ]; then
    echo "FAIL: vLLM $model not ready" | tee -a "$LOG"
    tail -20 "/tmp/vllm-$model.log" | tee -a "$LOG"
    kill $vpid 2>/dev/null
    continue
  fi
  sleep 3
  for scale in 1.0 0.5; do
    run_one "$model" "$scale"
  done
  kill $vpid 2>/dev/null
  for i in $(seq 1 30); do
    if ! kill -0 $vpid 2>/dev/null; then break; fi
    sleep 1
  done
  sleep 3
done
echo "" | tee -a "$LOG"
echo "=== $(date +%Y-%m-%dT%H:%M:%S%z) EXP4 Phase A END ===" | tee -a "$LOG"
