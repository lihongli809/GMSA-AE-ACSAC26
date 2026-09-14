#!/usr/bin/env bash
# GMSA AE - install script (stage 2 draft)
#
# Intended environment (verified on the authors' server):
#   Linux, Python 3.8.20, NVIDIA GPU, CUDA 11.7-compatible driver,
#   torch 1.13.1+cu117, torchvision 0.14.1+cu117 (2 x RTX 3080 10GB).
#
# NOTE: torch/torchvision 1.13.1+cu117 are NOT plain PyPI packages.
# They must be installed from the PyTorch CUDA 11.7 wheel index.
set -euo pipefail

# WARNING: this installer has NOT been validated on a clean Linux+CUDA host.
# It documents the intended install steps for the verified server environment.

PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "[GMSA-AE] install.sh (stage 2 draft)"
"$PYTHON_BIN" --version

# 1. PyTorch + torchvision from the PyTorch CUDA 11.7 wheel index
"$PYTHON_BIN" -m pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117

# 2. Remaining minimal closure dependencies
"$PYTHON_BIN" -m pip install numpy==1.23.0 pandas

echo "[GMSA-AE] Dependency install completed."
echo "[GMSA-AE] A CUDA-capable NVIDIA GPU is required for the three claims:"
echo "[GMSA-AE] dp/noise_add.py calls .cuda() unconditionally (DP noise path)."