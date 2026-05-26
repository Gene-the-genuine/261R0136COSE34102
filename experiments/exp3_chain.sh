#!/usr/bin/env bash
# EXP3: 8 system prompts × RF390 × Qwen3-4B
set -u
cd "$(dirname "$0")/.."
PYBIN=.venv/bin/python

LOG=/tmp/exp3_chain.log
echo "=== $(date +%Y-%m-%dT%H:%M:%S%z) EXP3 chain START ===" | tee -a "$LOG"

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

declare -a SPECS=(
  "0shot_direct:slm_bench/configs/0shot_system_prompt_direct.txt"
  "0shot_cot:slm_bench/configs/0shot_system_prompt_cot.txt"
  "0shot_valaware:slm_bench/configs/0shot_system_prompt_valaware.txt"
  "0shot_default:slm_bench/configs/0shot_system_prompt_default.txt"
  "fewshot_direct:slm_bench/configs/system_prompt_direct.txt"
  "fewshot_cot:slm_bench/configs/system_prompt_cot.txt"
  "fewshot_valaware:slm_bench/configs/system_prompt_valaware.txt"
  "fewshot_default:slm_bench/configs/system_prompt_default.txt"
)

for spec in "${SPECS[@]}"; do
  IFS=":" read -r name sp <<< "$spec"
  out=results/runs/run_exp3_${name}.jsonl
  echo "" | tee -a "$LOG"
  echo "=== $(date +%H:%M:%S) START $name ===" | tee -a "$LOG"
  timeout 1800 $PYBIN slm_bench/scripts/run_harness.py \
    --model qwen3-4b --preset qwen3-4b \
    --system-prompt "$sp" \
    --inputs slm_bench/fixtures/RF390_dataset.jsonl \
    --output "$out" \
    --chat-template-kwargs '{"enable_thinking":false}' \
    --simulator codex --concurrency 4 2>&1 | tail -3 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  n=$(wc -l < "$out" 2>/dev/null || echo 0)
  echo "DONE $name (rc=$rc, $n/390) @ $(date +%H:%M:%S)" | tee -a "$LOG"
done

kill $vpid 2>/dev/null
for i in $(seq 1 30); do
  if ! kill -0 $vpid 2>/dev/null; then break; fi
  sleep 1
done
echo "" | tee -a "$LOG"
echo "=== $(date +%Y-%m-%dT%H:%M:%S%z) EXP3 chain END ===" | tee -a "$LOG"
