#!/usr/bin/env bash
# Week 2 — full concurrency sweep for one config across both workloads.
# Start the server for the config first (scripts/serve_vllm.sh <config>), then:
#   bash scripts/run_sweep.sh fp16
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${1:?config (fp16|fp8|awq)}"

for WORKLOAD in interactive batch_summarization; do
  SWEEP=$(python - "$WORKLOAD" <<'EOF'
import sys, yaml
cfg = yaml.safe_load(open("configs/workloads.yaml"))
print(" ".join(str(c) for c in cfg["workloads"][sys.argv[1]]["concurrency_sweep"]))
EOF
)
  for C in $SWEEP; do
    echo "=== $CONFIG / $WORKLOAD / concurrency=$C ==="
    bash scripts/run_benchmark.sh "$CONFIG" "$WORKLOAD" "$C"
  done
done
