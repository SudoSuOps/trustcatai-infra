import os
import torch
from monai.metrics import DiceMetric

from .train import get_dataloaders, build_model

def run_eval(cfg):
    _, val_loader = get_dataloaders(cfg)
    model = build_model(cfg)

    ckpt_path = os.path.join(cfg.runs, "best.ckpt")
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path))

    model.eval()
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = batch["image"].cuda(), batch["label"].cuda()
            outputs = torch.softmax(model(inputs), dim=1)
            dice_metric(y_pred=outputs, y=labels)

    mean_dice = dice_metric.aggregate().item()
    print(f"[eval] mean_dice={mean_dice:.4f}")
