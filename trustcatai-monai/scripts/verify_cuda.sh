#!/usr/bin/env bash
set -euo pipefail

echo "[*] Host GPU availability"
nvidia-smi

echo "[*] PyTorch CUDA availability"
python - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device name:", torch.cuda.get_device_name(0))
PY

echo "[*] Docker runtime test"
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
