#!/usr/bin/env bash
set -euo pipefail

echo "[*] Bootstrapping trustcatai-monai repo structure..."

mkdir -p \
  docs \
  dockerfiles \
  kubernetes/charts/trustcatai-infer/templates \
  scripts \
  ansible/inventories/cluster \
  ansible/playbooks \
  ansible/roles/monai_gpu/tasks \
  trustcatai/{config,data,models,pipelines,trainer,infer} \
  .github/workflows

cat <<'GITIGNORE' > .gitignore
.venv/
__pycache__/
runs/
data/
outputs/
*.ckpt
*.pt
*.log
.DS_Store
wandb/
GITIGNORE

cat <<'REQ' > requirements.txt
monai[all]==1.4.0
torch==2.4.0
torchvision==0.19.0
nibabel>=5.1
pydicom>=2.4
SimpleITK>=2.3
scikit-image>=0.24
tqdm>=4.66
tensorboard>=2.17
mlflow>=2.16
pandas>=2.2
omegaconf>=2.3
pyyaml>=6.0
fastapi>=0.114
uvicorn[standard]>=0.30
REQ

cat <<'PYP' > pyproject.toml
[build-system]
requires = ["setuptools", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "trustcatai-monai"
version = "0.1.0"
description = "TrustCatAI MONAI training and inference toolkit"
authors = [{name = "TrustCatAI", email = "ops@sudosuops.ai"}]
dependencies = [
  "monai==1.3.0",
  "torch==2.4.0",
  "torchvision==0.19.0",
  "torchaudio==2.4.0",
  "numpy",
  "pydantic",
  "fastapi",
  "uvicorn[standard]",
  "matplotlib",
  "scikit-image",
  "tqdm",
  "pyyaml"
]

[project.urls]
Homepage = "https://github.com/SudoSuOps/trustcatai-monai"
PYP

cat <<'MK' > Makefile
SHELL := /bin/bash

.PHONY: venv verify train-local train-docker infer-docker build-train build-infer

venv:
	./scripts/setup_env.sh

verify:
	./scripts/verify_cuda.sh

train-local:
	./scripts/train_local.sh

train-docker:
	./scripts/train_docker.sh

infer-docker:
	./scripts/infer_docker.sh

build-train:
	docker build -t trustcatai/monai-train:latest -f dockerfiles/monai-train.Dockerfile .

build-infer:
	docker build -t trustcatai/monai-infer:latest -f dockerfiles/monai-infer.Dockerfile .
MK

cat <<'DOC' > docs/README.md
# TrustCatAI MONAI — Terminal-Grade Infra Guide

Fleet-ready MONAI infra for training + inference on NVIDIA GPUs (Blackwell / 5090).
Includes Docker images, Helm charts, Ansible playbooks, and Python training/inference agents.

---

## 📚 Sources
- MONAI: https://monai.io | https://github.com/Project-MONAI
- CUDA (Linux): https://docs.nvidia.com/cuda/cuda-installation-guide-linux/
- NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
- GPU Operator: https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/getting-started.html
- Medical Segmentation Decathlon datasets: http://medicaldecathlon.com/

---

## 1) Clone + bootstrap repo

```bash
git clone git@github.com:sudosuops/trustcatai-monai.git
cd trustcatai-monai
./bootstrap.sh
```

## 2) GPU sanity

```bash
make verify
```

## 3) Train locally (venv)

```bash
make venv
source .venv/bin/activate
make train-local
```

## 4) Train via Docker

```bash
DATA_DIR=$PWD/data OUT_DIR=$PWD/runs make train-docker
```

## 5) Inference via Docker

```bash
WEIGHTS=$PWD/runs/best.ckpt make infer-docker
curl -F "file=@/path/to/image.nii.gz" http://localhost:8080/predict
```

## 6) Kubernetes Deploy

```bash
helm upgrade --install trustcatai-infer kubernetes/charts/trustcatai-infer \
  -n trustcatai --create-namespace \
  --set image.repository=ghcr.io/sudosuops/trustcatai-monai \
  --set image.tag=monai-infer-latest
```

## 7) Ansible Remote Training

```bash
ansible-playbook -i ansible/inventories/cluster/inventory.ini ansible/playbooks/train-run.yml
```
DOC

cat <<'DT' > dockerfiles/monai-train.Dockerfile
FROM nvcr.io/nvidia/pytorch:24.06-py3

WORKDIR /workspace
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y git curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY trustcatai /workspace/trustcatai
COPY pyproject.toml /workspace/

ENV PYTHONPATH=/workspace

ENTRYPOINT ["python", "-m", "trustcatai.trainer.agent"]
DT

cat <<'DI' > dockerfiles/monai-infer.Dockerfile
FROM nvcr.io/nvidia/pytorch:24.06-py3

WORKDIR /workspace

COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY trustcatai /workspace/trustcatai
COPY pyproject.toml /workspace/

ENV PYTHONPATH=/workspace
EXPOSE 8080

ENTRYPOINT ["uvicorn", "trustcatai.infer.server:app", "--host", "0.0.0.0", "--port", "8080"]
DI

cat <<'STS' > scripts/setup_env.sh
#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo "[ok] venv ready. run: source .venv/bin/activate"
STS

cat <<'TRL' > scripts/train_local.sh
#!/usr/bin/env bash
set -euo pipefail

if [[ ! -d .venv ]]; then
  echo "[!] .venv not found. Run make venv first." >&2
  exit 1
fi

source .venv/bin/activate
python -m trustcatai.trainer.agent \
  --config ${CFG:-trustcatai/config/train_bratssyn_unet3d.yaml} \
  --data_root ${DATA_DIR:-./data} \
  --runs ${OUT_DIR:-./runs} "$@"
TRL

cat <<'TRD' > scripts/train_docker.sh
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
TRD

cat <<'INF' > scripts/infer_docker.sh
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
INF

cat <<'VER' > scripts/verify_cuda.sh
#!/usr/bin/env bash
set -euo pipefail

nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
VER

chmod +x scripts/setup_env.sh scripts/train_local.sh scripts/train_docker.sh scripts/infer_docker.sh scripts/verify_cuda.sh

cat <<'INV' > ansible/inventories/cluster/inventory.ini
[trainers]
# trainer1 ansible_host=10.0.0.11 ansible_user=ubuntu

[trainers:vars]
ansible_python_interpreter=/usr/bin/python3
INV

cat <<'PLAY' > ansible/playbooks/train-run.yml
- hosts: trainers
  gather_facts: yes
  become: yes
  roles:
    - monai_gpu
PLAY

cat <<'ROLE' > ansible/roles/monai_gpu/tasks/main.yml
---
- name: Ensure Docker is present
  apt:
    name: docker.io
    state: present
    update_cache: yes

- name: Pull training image
  docker_image:
    name: trustcatai/monai-train
    tag: latest
    source: pull

- name: Run training container
  docker_container:
    name: monai-train
    image: trustcatai/monai-train:latest
    state: started
    restart_policy: unless-stopped
    detach: yes
    volumes:
      - "{{ monai_data_dir | default('/data') }}:/data"
      - "{{ monai_runs_dir | default('/runs') }}:/runs"
    command: >-
      python -m trustcatai.trainer.agent
      --config trustcatai/config/train_bratssyn_unet3d.yaml
      --data_root /data
      --runs /runs
ROLE

cat <<'INIT' > trustcatai/__init__.py
"""TrustCatAI MONAI package."""

__all__ = ["models", "trainer", "infer", "pipelines"]
INIT

cat <<'CFG' > trustcatai/config/train_bratssyn_unet3d.yaml
batch_size: 2
epochs: 50
lr: 0.0008
data_root: ./data
runs: ./runs
CFG

cat <<'ICFG' > trustcatai/config/infer_bratssyn.yaml
weights: /weights/best.ckpt
num_workers: 2
ICFG

cat <<'AGENT' > trustcatai/trainer/agent.py
import argparse
import os
import json
import time
from omegaconf import OmegaConf

from .train import run_training
from .evaluate import run_eval


def main():
    parser = argparse.ArgumentParser("trustcatai-trainer")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--runs", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--eval_only", action="store_true")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    cfg.data_root = args.data_root
    cfg.runs = args.runs
    cfg.resume = args.resume

    os.makedirs(cfg.runs, exist_ok=True)

    with open(os.path.join(cfg.runs, "run_cmd.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    start = time.time()
    if args.eval_only:
        run_eval(cfg)
    else:
        run_training(cfg)
        run_eval(cfg)
    print(f"[trustcatai] completed in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
AGENT

cat <<'TRAIN' > trustcatai/trainer/train.py
import os
import torch
from monai.networks.nets import UNet
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.data import CacheDataset, DataLoader, load_decathlon_datalist
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    Orientationd,
    ScaleIntensityRanged,
    CropForegroundd,
    RandCropByPosNegLabeld,
    EnsureTyped,
)
from torch.optim import Adam


def build_transforms(pixdim=(1.5, 1.5, 2.0)):
    train = Compose([
        LoadImaged(keys=["
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=pixdim, mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["image"], a_min=-100, a_max=400, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        RandCropByPosNegLabeld(keys=["image", "label"], label_key="label", spatial_size=(96, 96, 96), pos=1, neg=1, num_samples=4),
        EnsureTyped(keys=["image", "label"]),
    ])
    val = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=pixdim, mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["image"], a_min=-100, a_max=400, b_min=0.0, b_max=1.0, clip=True),
        EnsureTyped(keys=["image", "label"]),
    ])
    return train, val


def get_dataloaders(cfg):
    data_json = os.path.join(cfg.data_root, "dataset.json")
    train_files = load_decathlon_datalist(data_json, True, "training")
    val_files = load_decathlon_datalist(data_json, True, "validation")
    train_tfms, val_tfms = build_transforms()
    train_ds = CacheDataset(train_files, train_tfms, cache_rate=0.2, num_workers=4)
    val_ds = CacheDataset(val_files, val_tfms, cache_rate=0.2, num_workers=4)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader


def build_model(cfg):
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )
    return model.cuda()


def run_training(cfg):
    train_loader, val_loader = get_dataloaders(cfg)
    model = build_model(cfg)
    loss_fn = DiceLoss(to_onehot_y=True, softmax=True)
    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
    optimizer = Adam(model.parameters(), lr=cfg.lr)

    best_dice = 0.0
    for epoch in range(cfg.epochs):
        model.train()
        for batch in train_loader:
            inputs, labels = batch["image"].cuda(), batch["label"].cuda()
            logits = model(inputs)
            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        dice_metric.reset()
        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch["image"].cuda(), batch["label"].cuda()
                outputs = torch.softmax(model(inputs), dim=1)
                dice_metric(y_pred=outputs, y=labels)
        mean_dice = dice_metric.aggregate().item()
        print(f"[epoch {epoch + 1}] mean_dice={mean_dice:.4f}")

        if mean_dice > best_dice:
            best_dice = mean_dice
            os.makedirs(cfg.runs, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(cfg.runs, "best.ckpt"))

    print(f"[train] best dice={best_dice:.4f}")
TRAIN

cat <<'EVAL' > trustcatai/trainer/evaluate.py
import os
import torch
from monai.metrics import DiceMetric

from .train import get_dataloaders, build_model


def run_eval(cfg):
    _, val_loader = get_dataloaders(cfg)
    model = build_model(cfg)

    ckpt_path = os.path.join(cfg.runs, "best.ckpt")
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location="cuda:0" if torch.cuda.is_available() else "cpu"))

    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    dice_metric = DiceMetric(include_background=False, reduction="mean")

    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = batch["image"].cuda(), batch["label"].cuda()
            outputs = torch.softmax(model(inputs), dim=1)
            dice_metric(y_pred=outputs, y=labels)

    mean_dice = dice_metric.aggregate().item()
    print(f"[eval] mean_dice={mean_dice:.4f}")
EVAL

cat <<'UNET' > trustcatai/models/unet3d.py
from monai.networks.nets import UNet


def build_unet3d(in_channels: int = 1, out_channels: int = 2):
    return UNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )
UNET

cat <<'ZOO' > trustcatai/models/zoo.py
from typing import Callable, Dict

from .unet3d import build_unet3d

_MODEL_REGISTRY: Dict[str, Callable[..., object]] = {
    "unet3d": build_unet3d,
}


def get_model(name: str, **kwargs):
    if name not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}")
    return _MODEL_REGISTRY[name](**kwargs)
ZOO

cat <<'XFM' > trustcatai/pipelines/transforms_3d.py
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    Orientationd,
    ScaleIntensityRanged,
    CropForegroundd,
    EnsureTyped,
)


def get_infer_transforms(pixdim=(1.5, 1.5, 2.0)):
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Spacingd(keys=["image"], pixdim=pixdim, mode="bilinear"),
        Orientationd(keys=["image"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["image"], a_min=-100, a_max=400, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image"], source_key="image"),
        EnsureTyped(keys=["image"]),
    ])
XFM

cat <<'POST' > trustcatai/pipelines/postprocess.py
from monai.transforms import Compose, Activationsd, AsDiscreted, EnsureTyped


def get_post_transforms():
    return Compose([
        EnsureTyped(keys=["pred"]),
        Activationsd(keys=["pred"], softmax=True),
        AsDiscreted(keys=["pred"], argmax=True),
    ])
POST

cat <<'SERVER' > trustcatai/infer/server.py
import os
import tempfile

import torch
from fastapi import FastAPI, UploadFile, File
from monai.transforms import (
    Compose,
    LoadImage,
    EnsureChannelFirst,
    Spacing,
    Orientation,
    ScaleIntensityRange,
    EnsureType,
)

from ..trainer.train import build_model

app = FastAPI(title="trustcatai-infer")

model = None
preproc = Compose([
    LoadImage(image_only=True),
    EnsureChannelFirst(),
    Spacing(pixdim=(1.5, 1.5, 2.0), mode="bilinear"),
    Orientation(axcodes="RAS"),
    ScaleIntensityRange(a_min=-100, a_max=400, b_min=0.0, b_max=1.0, clip=True),
    EnsureType(),
])


@app.on_event("startup")
def load_model():
    global model
    weights_path = os.environ.get("WEIGHTS", "/weights/best.ckpt")
    cfg_stub = type("Cfg", (object,), {})()
    model = build_model(cfg_stub)
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location="cuda:0" if torch.cuda.is_available() else "cpu"))
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await file.read())
        tmp.flush()
        image = preproc(tmp.name).unsqueeze(0)
    os.unlink(tmp.name)

    if torch.cuda.is_available():
        image = image.cuda()

    with torch.no_grad():
        prob = torch.softmax(model(image), dim=1)
        score = (prob[:, 1] > 0.5).float().mean().item()
    return {"ok": True, "score": score}
SERVER

cat <<'CHART' > kubernetes/charts/trustcatai-infer/Chart.yaml
apiVersion: v2
name: trustcatai-infer
version: 0.1.0
appVersion: "0.1.0"
CHART

cat <<'VALUES' > kubernetes/charts/trustcatai-infer/values.yaml
replicaCount: 1

image:
  repository: ghcr.io/sudosuops/trustcatai-monai
  tag: monai-infer-latest
  pullPolicy: IfNotPresent

resources:
  limits:
    nvidia.com/gpu: 1

service:
  type: ClusterIP
  port: 8080

weights:
  pvc: ""
VALUES

cat <<'DEPLOY' > kubernetes/charts/trustcatai-infer/templates/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: trustcatai-infer
spec:
  replicas: {{ .Values.replicaCount }}
  selector:
    matchLabels:
      app: trustcatai-infer
  template:
    metadata:
      labels:
        app: trustcatai-infer
    spec:
      containers:
        - name: app
          image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}"
          ports:
            - containerPort: 8080
          resources:
            limits:
              nvidia.com/gpu: 1
          env:
            - name: WEIGHTS
              value: "/weights/best.ckpt"
          volumeMounts:
            - name: weights
              mountPath: /weights
      volumes:
        - name: weights
          emptyDir: {}
---
apiVersion: v1
kind: Service
metadata:
  name: trustcatai-infer
spec:
  type: {{ .Values.service.type }}
  selector:
    app: trustcatai-infer
  ports:
    - port: {{ .Values.service.port }}
      targetPort: 8080
DEPLOY

cat <<'DATA' > trustcatai/data/README_DATA.md
# Data Layout

Expect Decathlon-style structure:

```
data/
  dataset.json
  imagesTr/
  labelsTr/
  imagesTs/
```

Each entry in `dataset.json` should include paths, e.g.:

```
[
  { "image": "imagesTr/case_0001.nii.gz", "label": "labelsTr/case_0001.nii.gz" },
  { "image": "imagesTr/case_0002.nii.gz", "label": "labelsTr/case_0002.nii.gz" }
]
```

Download from http://medicaldecathlon.com/ and update config paths accordingly.
DATA

cat <<'WF' > .github/workflows/build-docker.yml
name: Build & Push Train/Infer Images

on:
  push:
    branches: ["main"]
  workflow_dispatch:

jobs:
  build:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        image:
          - monai-train
          - monai-infer

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up QEMU
        uses: docker/setup-qemu-action@v3

      - name: Set up Buildx
        uses: docker/setup-buildx-action@v3

      - name: Log in to GHCR
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.repository_owner }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Log in to Docker Hub
        uses: docker/login-action@v3
        with:
          username: ${{ secrets.DOCKERHUB_USERNAME }}
          password: ${{ secrets.DOCKERHUB_TOKEN }}

      - name: Build & Push ${{ matrix.image }}
        uses: docker/build-push-action@v6
        with:
          context: .
          file: dockerfiles/${{ matrix.image }}.Dockerfile
          platforms: linux/amd64
          push: true
          tags: |
            ghcr.io/${{ github.repository_owner }}/trustcatai-monai:${{ matrix.image }}-latest
            docker.io/${{ secrets.DOCKERHUB_NAMESPACE || secrets.DOCKERHUB_USERNAME }}/trustcatai-${{ matrix.image }}:latest
WF

chmod +x bootstrap.sh

echo "[✓] trustcatai-monai structure ready."
