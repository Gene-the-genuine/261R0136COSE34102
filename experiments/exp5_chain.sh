#!/usr/bin/env bash
# EXP5 — Llama-3.2-3B × RF390 × preset × default SP. Other 5 models reused
# from EXP4 Phase A. After the run, runs exp5_report.py for the 6-model table.
set -u
cd "$(dirname "$0")/.."
PYBIN=.venv/bin/python
LOG=/tmp/exp5_chain.log
echo "=== $(date +%Y-%m-%dT%H:%M:%S%z) EXP5 chain START ===" | tee -a "$LOG"

OUT=results/runs/run_exp5_llama-3.2-3b.jsonl
if [ -f "$OUT" ] && [ "$(wc -l < "$OUT")" = "390" ]; then
  echo "SKIP llama-3.2-3b (exists 390 rows)" | tee -a "$LOG"
else
  nohup bash serve.sh llama-3.2-3b > /tmp/vllm-llama.log 2>&1 &
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
    tail -30 /tmp/vllm-llama.log | tee -a "$LOG"
    kill $vpid 2>/dev/null
    exit 1
  fi
  sleep 3
  echo "READY @ $(date +%H:%M:%S)" | tee -a "$LOG"

  echo "=== $(date +%H:%M:%S) START llama-3.2-3b preset ===" | tee -a "$LOG"
  timeout 1800 $PYBIN slm_bench/scripts/run_harness.py \
    --model llama-3.2-3b --preset llama-3.2-3b \
    --system-prompt slm_bench/configs/system_prompt_default.txt \
    --inputs slm_bench/fixtures/RF390_dataset.jsonl \
    --output "$OUT" \
    --simulator codex --concurrency 4 2>&1 | tail -3 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  n=$(wc -l < "$OUT" 2>/dev/null || echo 0)
  echo "DONE llama-3.2-3b (rc=$rc, $n/390) @ $(date +%H:%M:%S)" | tee -a "$LOG"

  kill $vpid 2>/dev/null
  for i in $(seq 1 30); do
    if ! kill -0 $vpid 2>/dev/null; then break; fi
    sleep 1
  done
fi

echo "" | tee -a "$LOG"
echo "=== $(date +%H:%M:%S) RUN EXP5 REPORT ===" | tee -a "$LOG"
$PYBIN experiments/exp5_report.py 2>&1 | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "=== $(date +%Y-%m-%dT%H:%M:%S%z) EXP5 chain END ===" | tee -a "$LOG"
