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
