#!/usr/bin/env bash
# EXP4 Phase B — Qwen3-4B × 7 scales on RF390.
# Reuses Phase A x0.5 and x1.0. New 5 runs.
set -u
cd "$(dirname "$0")/.."
PYBIN=.venv/bin/python
LOG=/tmp/exp4_phaseB.log
echo "=== $(date +%Y-%m-%dT%H:%M:%S%z) EXP4 Phase B START ===" | tee -a "$LOG"

# Each spec: scale:"max1 max2 max3":"mg1 mg2 mg3"
declare -a SPECS=(
  "0.25:38 58 97:8 8 13"
  "0.5:77 117 195:15 15 25"
  "0.75:115 175 292:23 23 38"
  "1.0:154 234 390:30 30 50"
  "1.5:231 351 585:30 30 50"
  "2.0:308 468 780:30 30 50"
  "3.0:462 702 1170:30 30 50"
)

nohup bash serve.sh qwen3-4b > /tmp/vllm-q4b.log 2>&1 &
vpid=$!
echo "vLLM pid=$vpid" | tee -a "$LOG"
ready=0
for i in $(seq 1 120); do
  if curl -s -m 2 http://127.0.0.1:8000/v1/models 2>/dev/null | grep -q '"id"'; then
    ready=1; break
  fi
  sleep 2
done
if [ $ready -eq 0 ]; then
  echo "FAIL: vLLM not ready" | tee -a "$LOG"
  tail -30 /tmp/vllm-q4b.log | tee -a "$LOG"
  kill $vpid 2>/dev/null; exit 1
fi
sleep 3
echo "READY @ $(date +%H:%M:%S)" | tee -a "$LOG"

for spec in "${SPECS[@]}"; do
  IFS=":" read -r scale mt mg <<< "$spec"
  out="results/runs/run_exp4_qwen3-4b_x${scale}.jsonl"
  if [ -f "$out" ] && [ "$(wc -l < "$out")" = "390" ]; then
    echo "SKIP x$scale (exists 390 rows)" | tee -a "$LOG"
    continue
  fi
  echo "" | tee -a "$LOG"
  echo "=== $(date +%H:%M:%S) START x$scale max=[$mt] margin=[$mg] ===" | tee -a "$LOG"
  timeout 1800 $PYBIN slm_bench/scripts/run_harness.py \
    --model qwen3-4b \
    --system-prompt slm_bench/configs/system_prompt_default.txt \
    --max-tokens $mt --margin $mg \
    --slos 2.0 5.0 10.0 \
    --inputs slm_bench/fixtures/RF390_dataset.jsonl \
    --output "$out" \
    --chat-template-kwargs '{"enable_thinking":false}' \
    --simulator codex --concurrency 4 2>&1 | tail -3 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  n=$(wc -l < "$out" 2>/dev/null || echo 0)
  echo "DONE x$scale (rc=$rc, $n/390) @ $(date +%H:%M:%S)" | tee -a "$LOG"
done

kill $vpid 2>/dev/null
for i in $(seq 1 30); do
  if ! kill -0 $vpid 2>/dev/null; then break; fi
  sleep 1
done
echo "" | tee -a "$LOG"
echo "=== $(date +%Y-%m-%dT%H:%M:%S%z) EXP4 Phase B END ===" | tee -a "$LOG"
