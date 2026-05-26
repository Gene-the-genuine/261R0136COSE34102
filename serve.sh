#!/usr/bin/env bash
# Dispatch vLLM serve by model preset.
#
# Usage:
#   bash serve.sh                  # default = gemma  (backward-compatible)
#   bash serve.sh <preset>         # see PRESETS below
#   PORT=8001 bash serve.sh qwen3-0.6b
#
# Presets:
#   gemma          Gemma-4-E2B-it NVFP4 (default)
#   qwen3-0.6b     Qwen/Qwen3-0.6B
#   qwen3-1.7b     Qwen/Qwen3-1.7B
#   qwen3-4b       Qwen/Qwen3-4B
#   phi-3.5-mini   microsoft/Phi-3.5-mini-instruct
#   llama-3.2-3b   meta-llama/Llama-3.2-3B-Instruct  (gated; needs HF login)
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
export PATH="$HOME/.local/bin:$PATH"
source "$ROOT/.venv/bin/activate"

PRESET="${1:-gemma}"
[ "$#" -ge 1 ] && shift

QUANT_FLAG=""
KV_FLAG=""
GPU_UTIL="${GPU_UTIL:-0.72}"

case "$PRESET" in
  gemma)
    MODEL="$ROOT/models/Gemma-4-E2B-it-NVFP4"
    SERVED_NAME="gemma-4-E2B-it"
    QUANT_FLAG="--quantization modelopt_fp4"
    KV_FLAG="--kv-cache-dtype fp8"
    ;;
  qwen3-0.6b)
    MODEL="$ROOT/models/Qwen3-0.6B"
    SERVED_NAME="qwen3-0.6b"
    ;;
  qwen3-1.7b)
    MODEL="$ROOT/models/Qwen3-1.7B"
    SERVED_NAME="qwen3-1.7b"
    ;;
  qwen3-4b)
    MODEL="$ROOT/models/Qwen3-4B"
    SERVED_NAME="qwen3-4b"
    ;;
  phi-3.5-mini)
    MODEL="$ROOT/models/Phi-3.5-mini-instruct"
    SERVED_NAME="phi-3.5-mini"
    ;;
  llama-3.2-3b)
    MODEL="$ROOT/models/Llama-3.2-3B-Instruct"
    SERVED_NAME="llama-3.2-3b"
    ;;
  *)
    echo "Unknown preset: $PRESET" >&2
    echo "Use one of: gemma, qwen3-0.6b, qwen3-1.7b, qwen3-4b, phi-3.5-mini, llama-3.2-3b" >&2
    exit 1
    ;;
esac

PORT="${PORT:-8000}"
MAX_LEN="${MAX_LEN:-8192}"

export VLLM_NVFP4_GEMM_BACKEND="${VLLM_NVFP4_GEMM_BACKEND:-cutlass}"

echo "[serve.sh] preset=$PRESET model=$MODEL served-name=$SERVED_NAME port=$PORT"

exec vllm serve "$MODEL" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --served-model-name "$SERVED_NAME" \
  $QUANT_FLAG \
  $KV_FLAG \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --enforce-eager \
  --limit-mm-per-prompt '{"image":0,"audio":0}' \
  "$@"
