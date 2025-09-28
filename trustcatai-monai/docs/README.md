# TrustCat.ai — MONAI Infra (Training + Inference)

> Terminal-grade ops: NVIDIA GPUs (Blackwell/5090), Docker, MONAI.

## TL;DR
```bash
make verify        # check GPUs
make venv          # setup local env (optional)
make train-local   # train in venv
make train-docker  # train via Docker
make infer-docker  # start FastAPI infer server
helm upgrade --install trustcatai-infer ./kubernetes/charts/trustcatai-infer -n trustcatai
```

## Sources
- MONAI: <https://monai.io> • <https://github.com/Project-MONAI>
- CUDA (Linux): <https://docs.nvidia.com/cuda/cuda-installation-guide-linux/>
- NVIDIA Container Toolkit: <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html>
- GPU Operator: <https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/getting-started.html>
- MSD datasets: <http://medicaldecathlon.com/>

## Workflow Highlights
- `make verify` — host + Docker GPU check
- `make train-local` / `make train-docker` — MONAI training
- `make infer-docker` — FastAPI inference
- `helm upgrade --install` — K8s deployment
- `ansible-playbook ...train-run.yml` — remote training automation

## Repo Layout (key paths)
- `dockerfiles/` – training/inference images
- `scripts/` – env setup, train/infer shortcuts
- `trustcatai/` – models, trainer agent, inference server
- `kubernetes/` – Helm chart + GPU manifests
- `ansible/` – remote training playbooks
```
