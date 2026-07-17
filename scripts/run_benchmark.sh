#!/usr/bin/env bash
# Drive load with GenAI-Perf against a running vLLM server.
# Usage: bash scripts/run_benchmark.sh <config> <workload> <concurrency>
#   e.g. bash scripts/run_benchmark.sh fp16 interactive 8
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

CONFIG="${1:?config (fp16|fp8|awq)}"
WORKLOAD="${2:?workload (interactive|batch_summarization)}"
CONCURRENCY="${3:?concurrency (int)}"

# Pull length distribution for the workload from configs/workloads.yaml
read -r ISL ISL_SD OSL OSL_SD MODEL <<< "$(python - "$WORKLOAD" <<'EOF'
import sys, yaml
cfg = yaml.safe_load(open("configs/workloads.yaml"))
w = cfg["workloads"][sys.argv[1]]
print(w["input_tokens_mean"], w["input_tokens_stddev"],
      w["output_tokens_mean"], w["output_tokens_stddev"], cfg["model"])
EOF
)"

# AWQ serves a different model name on the endpoint
if [ "$CONFIG" = "awq" ]; then MODEL="${MODEL}-AWQ"; fi

OUT="results/raw/${CONFIG}_${WORKLOAD}_c${CONCURRENCY}"
mkdir -p "$OUT"

genai-perf profile \
  -m "$MODEL" \
  --endpoint-type chat \
  --url http://localhost:8000 \
  --streaming \
  --synthetic-input-tokens-mean "$ISL" --synthetic-input-tokens-stddev "$ISL_SD" \
  --output-tokens-mean "$OSL" --output-tokens-stddev "$OSL_SD" \
  --concurrency "$CONCURRENCY" \
  --request-count $(( CONCURRENCY * 20 )) \
  --warmup-request-count "$CONCURRENCY" \
  --artifact-dir "$OUT"

echo "Done → $OUT"
