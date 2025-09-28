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
