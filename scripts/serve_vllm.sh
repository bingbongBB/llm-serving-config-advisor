#!/usr/bin/env bash
# Serve the model with vLLM under a named config. Usage: bash scripts/serve_vllm.sh {fp16|fp8|awq}
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

CONFIG="${1:-fp16}"

case "$CONFIG" in
  fp16) MODEL="Qwen/Qwen2.5-3B-Instruct";     EXTRA="" ;;
  fp8)  MODEL="Qwen/Qwen2.5-3B-Instruct";     EXTRA="--quantization fp8" ;;
  awq)  MODEL="Qwen/Qwen2.5-3B-Instruct-AWQ"; EXTRA="" ;;
  *) echo "unknown config: $CONFIG (use fp16|fp8|awq)"; exit 1 ;;
esac

echo "Serving $MODEL ($CONFIG) on :8000 ..."
# 8 GB VRAM: FP16 weights ~6.2 GB leave ~1 GB KV cache at 0.92 utilization.
# --max-model-len 4096 covers the 2k-input batch workload; if FP16 OOMs at high
# concurrency, that IS a finding — record it, quantized configs will have headroom.
exec vllm serve "$MODEL" \
  --host 0.0.0.0 --port 8000 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.92 \
  $EXTRA
