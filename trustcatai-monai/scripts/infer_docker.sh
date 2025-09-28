#!/usr/bin/env bash
set -euo pipefail

IMG="trustcatai/monai-infer:latest"

docker build -t "$IMG" -f dockerfiles/monai-infer.Dockerfile .

WEIGHTS="${WEIGHTS:-$PWD/runs/best.ckpt}"
PORT="${PORT:-8080}"

if [[ ! -f "$WEIGHTS" ]]; then
  echo "[!] WEIGHTS not found: $WEIGHTS" >&2
  exit 1
fi

docker run --rm --gpus all -p ${PORT}:8080 \
  -v "$WEIGHTS":/weights/best.ckpt \
  "$IMG"
