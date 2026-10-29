# -*- coding: utf-8 -*-
"""Train SA-ONet.

Usage:
    python train.py --data_root ./data --epochs 70 --lr 1e-3 --reduction 16
"""
import argparse
import os

import torch

import config
from dataset import get_loaders
from losses import build_criterion
from model import build_model
from utils import CSVLogger, class_counts, evaluate, train_one_epoch


def parse_args():
    p = argparse.ArgumentParser(description="Train SA-ONet")
    p.add_argument("--data_root", type=str, default=config.DATA_ROOT)
    p.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    p.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    p.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--reduction", type=int, default=config.SE_REDUCTION,
                   help="SE channel reduction ratio r (8/16/32)")
    p.add_argument("--loss", type=str, default="aore",
                   choices=["ce", "weighted_ce", "focal", "aore"])
    p.add_argument("--variant", type=str, default="sa_onet",
                   choices=["sa_onet", "resnet50", "resnet50_fpn", "resnet50_se"])
    p.add_argument("--beta", type=float, default=config.AORE_BETA)
    p.add_argument("--num_workers", type=int, default=config.NUM_WORKERS)
    p.add_argument("--tag", type=str, default="main",
                   help="tag used in output file names")
    return p.parse_args()


def main():
    args = parse_args()
    device = config.DEVICE
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    train_loader, val_loader, _, num_classes = get_loaders(
        args.data_root, args.batch_size, args.num_workers)
    print(f"[SA-ONet] classes={num_classes} device={device}")

    counts = class_counts(train_loader.dataset)
    config.AORE_BETA = args.beta
    criterion = build_criterion(args.loss, num_classes, counts, device)
    model = build_model(args.variant, num_classes,
                        se_reduction=args.reduction).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=config.MOMENTUM,
                                weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    logger = CSVLogger(
        os.path.join(config.OUTPUT_DIR, f"train_log_{args.tag}.csv"),
        ["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

    best_acc = 0.0
    best_path = os.path.join(config.OUTPUT_DIR, f"best_model_{args.tag}.pth")
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion,
                                          optimizer, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        if hasattr(criterion, "update_dynamic_weights"):
            criterion.update_dynamic_weights()   # Aore-Loss Eq.(9)

        logger.log(epoch=epoch, train_loss=round(tr_loss, 6),
                   train_acc=round(tr_acc, 6), val_loss=round(va_loss, 6),
                   val_acc=round(va_acc, 6))
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f}")

        if va_acc > best_acc:
            best_acc = va_acc
            torch.save({"model_state_dict": model.state_dict(),
                        "num_classes": num_classes,
                        "epoch": epoch, "val_acc": va_acc,
                        "args": vars(args)}, best_path)
    logger.close()
    print(f"Best val acc: {best_acc:.4f}, model saved to {best_path}")


if __name__ == "__main__":
    main()
