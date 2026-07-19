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

# WSL2 workarounds (documented in docs/weekly-log/week1.md):
#   - V2 model runner needs UVA, unsupported in WSL2 CUDA -> force V1
#   - FlashInfer sampler needs nvcc JIT, no CUDA toolkit installed -> disable
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_USE_FLASHINFER_SAMPLER=0

echo "Serving $MODEL ($CONFIG) on :8000 ..."
# 8 GB VRAM shared with Windows desktop (~1 GB) -> only ~6.9 GB free:
#   - utilization 0.85 (6.8 GB budget)
#   - --enforce-eager frees CUDA-graph memory for KV cache
#     (all configs use eager mode -> internally fair; absolute numbers
#      slightly below CUDA-graph mode, noted in README)
exec vllm serve "$MODEL" \
  --host 0.0.0.0 --port 8000 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --enforce-eager \
  --max-num-batched-tokens 1024 \
  --max-num-seqs 16 \
  $EXTRA