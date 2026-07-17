#!/usr/bin/env bash
# Week 1 / Block 1 — create venv, install vLLM + GenAI-Perf.
# vLLM wheels bundle their own CUDA runtime; no CUDA toolkit install needed for serving.
set -euo pipefail
cd "$(dirname "$0")/.."

sudo apt-get update -y
sudo apt-get install -y python3-venv python3-pip git

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# Pin versions once installed (pip freeze > requirements.lock) for reproducibility.
pip install vllm
pip install genai-perf
pip install pandas matplotlib pyyaml

echo
echo "== Versions =="
python -c "import vllm; print('vllm', vllm.__version__)"
genai-perf --version || true

echo
echo "OK. Activate with: source .venv/bin/activate"
echo "Next: bash scripts/serve_vllm.sh fp16"
