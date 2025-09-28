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
