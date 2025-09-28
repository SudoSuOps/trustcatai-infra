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
