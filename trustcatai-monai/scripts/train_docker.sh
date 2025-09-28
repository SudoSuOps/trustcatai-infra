#!/usr/bin/env bash
set -euo pipefail

IMG="trustcatai/monai-train:latest"

docker build -t "$IMG" -f dockerfiles/monai-train.Dockerfile .

DATA_DIR="${DATA_DIR:-$PWD/data}"
OUT_DIR="${OUT_DIR:-$PWD/runs}"
CFG="${CFG:-trustcatai/config/train_bratssyn_unet3d.yaml}"

mkdir -p "$DATA_DIR" "$OUT_DIR"

docker run --rm --gpus all \
  -v "$DATA_DIR":/data \
  -v "$OUT_DIR":/runs \
  -v "$PWD":/workspace \
  "$IMG" --config "$CFG" --data_root /data --runs /runs "$@"
