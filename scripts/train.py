#!/usr/bin/env python3
"""
Train segmentation model on unified 7-class off-road ontology.

Supports:
  - DDRNet23-Slim — Qualcomm NPU optimized (RECOMMENDED for IQ-9075)
  - EfficientViT-Seg (B0, B1, B2) — MIT Han Lab (not INT8-safe)
  - FFNet (40S, 54S, 78S) — Qualcomm AI Research

Usage:
    conda activate offroad

    # DDRNet23-Slim (RECOMMENDED for IQ-9075 deployment)
    python scripts/train.py --model ddrnet23-slim --fast --num_workers 8

    # EfficientViT-B1 (high accuracy, but INT8 issues on NPU)
    python scripts/train.py --model efficientvit-b1 --fast --num_workers 8
"""

import os
import sys
import argparse
import time

sys.path.insert(0, "./efficientvit")
sys.path.append(".")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.amp import autocast, GradScaler

from src.dataset import (
    build_dataset,
    FocalLoss,
    EMA,
    NUM_CLASSES,
    CLASS_NAMES,
)
from src.models import build_model, load_checkpoint, SUPPORTED_MODELS


# -------------------------------------------------------------------
# Evaluation
# -------------------------------------------------------------------

def compute_confusion_matrix(pred, label, num_classes):
    """Compute confusion matrix for a single prediction-label pair."""
    mask = (label >= 0) & (label < num_classes)
    label = label[mask]
    pred = pred[mask]
    k = (label * num_classes + pred).astype(np.int64)
    bincount = np.bincount(k, minlength=num_classes * num_classes)
    return bincount.reshape(num_classes, num_classes)


def evaluate_miou(model, dataloader, num_classes):
    """Evaluate mean IoU on a validation set."""
    model.eval()
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.cuda()
            labels = labels.cuda()

            outputs = model(images)
            if outputs.shape[2:] != labels.shape[1:]:
                outputs = nn.functional.interpolate(
                    outputs,
                    size=labels.shape[1:],
                    mode="bilinear",
                    align_corners=False,
                )

            preds = outputs.argmax(dim=1).cpu().numpy()
            labels_np = labels.cpu().numpy()

            for p, g in zip(preds, labels_np):
                confusion += compute_confusion_matrix(p, g, num_classes)

    ious = []
    for c in range(num_classes):
        tp = confusion[c, c]
        fp = confusion[:, c].sum() - tp
        fn = confusion[c, :].sum() - tp
        denom = tp + fp + fn
        if denom > 0:
            ious.append(tp / denom)
        else:
            ious.append(float("nan"))

    miou = np.nanmean(ious) * 100.0
    return miou, ious


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train segmentation model on unified 7-class off-road ontology"
    )
    parser.add_argument(
        "--model", type=str, default="ddrnet23-slim",
        choices=list(SUPPORTED_MODELS.keys()),
        help="Model to train (default: ddrnet23-slim)"
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument(
        "--crop_size", type=str, default="544,640",
        help="Crop size as 'H,W' (e.g. '544,640' for S10 Ultra) or single int"
    )
    parser.add_argument("--eval_interval", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint path")
    parser.add_argument("--no_amp", action="store_true",
                        help="Disable automatic mixed precision")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--fast", action="store_true",
                        help="Use pre-resized datasets (*_fast dirs) for faster I/O. "
                             "Run: python scripts/preprocess_datasets.py first.")
    parser.add_argument("--bb_lr_factor", type=float, default=0.1,
                        help="Backbone LR = lr * bb_lr_factor (default: 0.1). "
                             "Prevents catastrophic forgetting of pretrained features.")
    parser.add_argument("--quiet", action="store_true",
                        help="Disable real-time progress bar. "
                             "Print one summary line per epoch when finished.")
    args = parser.parse_args()

    # Parse crop_size: "544,640" -> (544, 640) or "512" -> (512, 512)
    crop_parts = args.crop_size.split(",")
    if len(crop_parts) == 2:
        crop_size = (int(crop_parts[0]), int(crop_parts[1]))
    else:
        crop_size = (int(crop_parts[0]), int(crop_parts[0]))

    # Checkpoint directory includes model name
    ckpt_dir = os.path.join(args.checkpoint_dir, args.model)
    os.makedirs(ckpt_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    suffix = "_fast" if args.fast else ""
    data_config = {
        "rellis_root": f"./data/Rellis-3D{suffix}",
        "rellis_split_train": f"./data/Rellis-3D{suffix}/split_custom/train_70.lst",
        "rellis_split_val": f"./data/Rellis-3D{suffix}/split_custom/test_30.lst",
        "rugd_root": f"./data/RUGD{suffix}",
        "goose_root": f"./data/GOOSE{suffix}",
        "crop_size": crop_size,
    }
    if args.fast:
        print("[FAST MODE] Using pre-resized datasets")

    train_set = build_dataset(data_config, is_train=True)
    val_set = build_dataset(data_config, is_train=False)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=max(1, args.batch_size // 2),
        shuffle=False,
        num_workers=args.num_workers,
    )

    # ------------------------------------------------------------------
    # Model (unified: DDRNet / EfficientViT / FFNet)
    # ------------------------------------------------------------------
    model = build_model(args.model, num_classes=NUM_CLASSES, pretrained=True)

    if args.resume:
        model = load_checkpoint(model, args.resume)

    model = model.cuda()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Total params: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"[Model] Trainable:    {trainable_params:,} ({trainable_params/1e6:.2f}M)")

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    # Class weights for Focal Loss (alpha).
    # Minimum weight is 1.0 to prevent mode collapse.
    # Previous weights (Sky=0.3, Vegetation=0.5) were too low --
    # the model ignored these classes entirely, collapsing to
    # predict everything as Obstacle.
    class_weights = torch.tensor(
        #  SmoothGnd  RoughGnd  Veg   Obstacle  Water  Sky   Dynamic
        [  1.5,       3.0,      1.0,  1.0,      5.0,   1.0,  5.0  ],
        dtype=torch.float32,
    ).cuda()

    criterion = FocalLoss(alpha=class_weights, gamma=2.0, ignore_index=255)

    # ------------------------------------------------------------------
    # Optimizer and Scheduler
    # ------------------------------------------------------------------
    # Differential Learning Rate:
    #   Backbone (pretrained): lr * bb_lr_factor (default 0.1x)
    #   Head (Xavier init):    lr (full)
    #
    # Without this, backbone features are destroyed before the head
    # learns to classify, causing mode collapse (all pixels → Obstacle).
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "final_layer" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    bb_lr = args.lr * args.bb_lr_factor
    print(f"[Optimizer] Backbone LR: {bb_lr:.6f} ({len(backbone_params)} tensors)")
    print(f"[Optimizer] Head LR:     {args.lr:.6f} ({len(head_params)} tensors)")

    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": bb_lr},
        {"params": head_params,     "lr": args.lr},
    ], weight_decay=0.01)

    warmup = LinearLR(optimizer, start_factor=1e-6, total_iters=args.warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs)
    scheduler = SequentialLR(
        optimizer, [warmup, cosine], milestones=[args.warmup_epochs]
    )

    # ------------------------------------------------------------------
    # EMA and AMP
    # ------------------------------------------------------------------
    ema = EMA(model, decay=0.9999)
    use_amp = not args.no_amp and torch.cuda.is_available()
    scaler = GradScaler("cuda") if use_amp else None

    # ------------------------------------------------------------------
    # Training Loop
    # ------------------------------------------------------------------
    best_miou = 0.0

    model_info = SUPPORTED_MODELS[args.model]
    npu_tag = "INT8-safe" if model_info.get("int8_safe") else "INT8 warning"

    print(f"\n{'='*60}")
    print(f"  Model:    {args.model} ({npu_tag})")
    print(f"  Classes:  {NUM_CLASSES}-class unified ontology")
    print(f"  Datasets: {len(train_set)} train / {len(val_set)} val samples")
    print(f"  Epochs:   {args.epochs}, Eval every {args.eval_interval}")
    print(f"  Batch:    {args.batch_size}, Crop: {crop_size[0]}x{crop_size[1]}")
    print(f"  LR:       backbone={bb_lr:.6f}, head={args.lr:.6f}")
    print(f"  Loss:     FocalLoss (gamma=2.0)")
    print(f"  EMA:      decay=0.9999")
    print(f"  AMP:      {'ON' if use_amp else 'OFF'}")
    print(f"  Ckpt dir: {ckpt_dir}")
    print(f"{'='*60}\n")

    num_batches = len(train_loader)
    bar_w = 20

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        epoch_start = time.time()

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            optimizer.zero_grad()

            if use_amp:
                with autocast("cuda"):
                    outputs = model(images)
                    if outputs.shape[2:] != labels.shape[1:]:
                        outputs = nn.functional.interpolate(
                            outputs, size=labels.shape[1:],
                            mode="bilinear", align_corners=False,
                        )
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                if outputs.shape[2:] != labels.shape[1:]:
                    outputs = nn.functional.interpolate(
                        outputs, size=labels.shape[1:],
                        mode="bilinear", align_corners=False,
                    )
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            ema.update(model)
            total_loss += loss.item()

            # Real-time progress bar (skip if --quiet)
            if not args.quiet:
                n = batch_idx + 1
                pct = n / num_batches
                filled = int(bar_w * pct)
                bar = "\u2588" * filled + " " * (bar_w - filled)
                elapsed = time.time() - epoch_start
                eta = elapsed / n * (num_batches - n)
                avg_l = total_loss / n
                print(f"\r  Epoch {epoch+1:3d}/{args.epochs} "
                      f"|{bar}| {pct*100:5.1f}% "
                      f"Loss:{avg_l:.4f} ETA:{eta:.0f}s   ",
                      end="", flush=True)

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        current_lr = optimizer.param_groups[-1]["lr"]  # head LR
        epoch_time = time.time() - epoch_start

        if not args.quiet:
            # Overwrite the progress bar with final summary (same line)
            bar = "\u2588" * bar_w
            print(f"\r  Epoch {epoch+1:3d}/{args.epochs} "
                  f"|{bar}| LR:{current_lr:.6f} "
                  f"Loss:{avg_loss:.4f} Time:{epoch_time:.1f}s")
        else:
            # Quiet mode: one line per epoch
            print(
                f"  Epoch {epoch+1:3d}/{args.epochs} | "
                f"LR: {current_lr:.6f} | "
                f"Loss: {avg_loss:.4f} | "
                f"Time: {epoch_time:.1f}s"
            )

        # Periodic checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(
                model.state_dict(),
                os.path.join(ckpt_dir, f"epoch_{epoch + 1}.pth"),
            )

        # Validation with EMA weights
        if (epoch + 1) % args.eval_interval == 0:
            ema.apply_shadow(model)

            miou, ious = evaluate_miou(model, val_loader, NUM_CLASSES)
            print(f"  [Val] mIoU: {miou:.2f}%")
            for c_idx, iou_val in enumerate(ious):
                if not np.isnan(iou_val):
                    print(f"    {CLASS_NAMES[c_idx]:>16s}: {iou_val * 100:.1f}%")

            if miou > best_miou:
                best_miou = miou
                torch.save(
                    model.state_dict(),
                    os.path.join(ckpt_dir, "best_model.pth"),
                )
                print(f"  >>> New best mIoU: {miou:.2f}% -- saved.")

            ema.restore(model)

    # Save final EMA weights
    ema.apply_shadow(model)
    torch.save(model.state_dict(), os.path.join(ckpt_dir, "final_ema_model.pth"))
    print(f"\nTraining complete. Best mIoU: {best_miou:.2f}%")
    print(f"Checkpoints saved to: {ckpt_dir}/")

    # Deployment reminder for INT8-safe models
    if model_info.get("int8_safe"):
        print(f"\n{'='*60}")
        print(f"  Next: IQ-9075 Deployment")
        print(f"{'='*60}")
        print(f"  python scripts/export_qnn.py \\")
        print(f"      --checkpoint {ckpt_dir}/best_model.pth \\")
        print(f"      --method hub")
        print()


if __name__ == "__main__":
    main()