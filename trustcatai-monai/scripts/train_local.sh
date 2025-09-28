#!/usr/bin/env bash
set -euo pipefail

if [[ ! -d .venv ]]; then
  echo "[!] .venv not found. Run ./scripts/setup_env.sh first." >&2
  exit 1
fi

source .venv/bin/activate
python -m trustcatai.trainer.agent \
  --config ${CFG:-trustcatai/config/train_bratssyn_unet3d.yaml} \
  --data_root ${DATA_DIR:-./data} \
  --runs ${OUT_DIR:-./runs} "$@"
