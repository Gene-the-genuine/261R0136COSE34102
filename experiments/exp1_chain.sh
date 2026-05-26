#!/usr/bin/env bash
# EXP1 chain: 7 cells × Qwen3-4B with codex simulator.
# Output: <repo-root>/results/runs/run_exp1_<cell>.jsonl
# Log:    appended to /tmp/exp1_chain.log
set -u
cd "$(dirname "$0")/.."
PYBIN=.venv/bin/python   # adapt if your venv is elsewhere

LOG=/tmp/exp1_chain.log
echo "=== $(date +%Y-%m-%dT%H:%M:%S%z) EXP1 chain START ===" | tee -a "$LOG"

# Build per-label RF390 prompt subsets (in-memory tmp files)
mkdir -p /tmp/exp1_inputs
$PYBIN <<'PY'
import json, os
os.makedirs("/tmp/exp1_inputs", exist_ok=True)
with open("slm_bench/fixtures/RF390_dataset.jsonl") as f:
    rows = [json.loads(l) for l in f]
for lab in ("ask","reject","solve"):
    with open(f"/tmp/exp1_inputs/rf390_{lab}.jsonl","w") as out:
        for r in rows:
            if r["label"] == lab:
                out.write(json.dumps({"id": r["id"], "prompt": r["prompt"]}, ensure_ascii=False)+"\n")
print("built per-label inputs")
PY

# Step 1: ensure vLLM is fresh (Qwen3-4B)
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
  echo "FAIL: vLLM did not become ready in 240s" | tee -a "$LOG"
  tail -30 /tmp/vllm-q4b.log | tee -a "$LOG"
  kill $vpid 2>/dev/null
  exit 1
fi
sleep 3
echo "READY @ $(date +%H:%M:%S)" | tee -a "$LOG"

# Step 2: 7 cells sequential
declare -a SPECS=(
  "rf390_solve:/tmp/exp1_inputs/rf390_solve.jsonl:1800"
  "rf390_reject:/tmp/exp1_inputs/rf390_reject.jsonl:1800"
  "rf390_ask:/tmp/exp1_inputs/rf390_ask.jsonl:1800"
  "mmlu_alg:slm_bench/fixtures/hf/mmlu_abstract_algebra_test.jsonl:1800"
  "mmlu_bio:slm_bench/fixtures/hf/mmlu_high_school_biology_test.jsonl:2700"
  "gsm8k:slm_bench/fixtures/hf/gsm8k_test.jsonl:3600"
  "mmlu_law:slm_bench/fixtures/hf/mmlu_professional_law_test.jsonl:3600"
)
# Note: MMLU/GSM8K cells require derived_fixtures format ({id, prompt}).
# For convenience this chain uses the variant produced by scoring/build_hf_harness.py.
# Run that script once first to regenerate them at results/derived_fixtures/.

for spec in "${SPECS[@]}"; do
  IFS=":" read -r name in tmo <<< "$spec"
  # MMLU/GSM8K cells: use derived_fixtures if available
  case "$name" in
    mmlu_alg)  in=results/derived_fixtures/mmlu_abstract_algebra_prompts.jsonl ;;
    mmlu_bio)  in=results/derived_fixtures/mmlu_high_school_biology_prompts.jsonl ;;
    gsm8k)     in=results/derived_fixtures/gsm8k_test_prompts.jsonl ;;
    mmlu_law)  in=results/derived_fixtures/mmlu_professional_law_prompts.jsonl ;;
  esac
  out=results/runs/run_exp1_${name}.jsonl
  echo "" | tee -a "$LOG"
  echo "=== $(date +%H:%M:%S) START $name (timeout ${tmo}s) ===" | tee -a "$LOG"
  timeout "$tmo" $PYBIN slm_bench/scripts/run_harness.py \
    --model qwen3-4b --preset qwen3-4b \
    --system-prompt slm_bench/configs/system_prompt_default.txt \
    --inputs "$in" --output "$out" \
    --chat-template-kwargs '{"enable_thinking":false}' \
    --simulator codex --concurrency 4 2>&1 | tail -3 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  done_count=$(wc -l < "$out" 2>/dev/null || echo 0)
  in_count=$(wc -l < "$in")
  echo "DONE $name (rc=$rc, $done_count/$in_count) @ $(date +%H:%M:%S)" | tee -a "$LOG"
done

# Step 3: shutdown
kill $vpid 2>/dev/null
for i in $(seq 1 30); do
  if ! kill -0 $vpid 2>/dev/null; then break; fi
  sleep 1
done
echo "" | tee -a "$LOG"
echo "=== $(date +%Y-%m-%dT%H:%M:%S%z) EXP1 chain END ===" | tee -a "$LOG"
