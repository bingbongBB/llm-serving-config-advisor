#!/usr/bin/env bash
# Week 1 / Block 1 — verify the WSL2 environment can see the GPU.
# Run inside WSL2 Ubuntu.
set -euo pipefail

echo "== 1. GPU visible? =="
if ! command -v nvidia-smi &>/dev/null; then
  echo "FAIL: nvidia-smi not found."
  echo "  In WSL2 you should NOT install a Linux driver — only the Windows"
  echo "  NVIDIA driver is needed; WSL2 exposes it automatically."
  echo "  Update the Windows driver, then restart WSL: wsl --shutdown"
  exit 1
fi
nvidia-smi

echo
echo "== 2. Compute capability (expect 8.9 for RTX 4090 → native FP8) =="
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv

echo
echo "== 3. Python =="
python3 --version || { echo "FAIL: need python3 (sudo apt install python3 python3-venv)"; exit 1; }

echo
echo "OK — environment looks good. Next: bash setup/02_install.sh"
