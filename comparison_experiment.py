# -*- coding: utf-8 -*-
"""Comparison experiment against representative baselines and loss functions.

Compares SA-ONet (loss=aore) with classical baselines (ResNet50 etc.) and
compares the Aore-Loss with CrossEntropy / Weighted-CE / Focal Loss.

Usage:
    python comparison_experiment.py --data_root ./data --epochs 70
"""
import argparse

import torch

import config
from dataset import get_loaders
from losses import build_criterion
from model import build_model
from utils import CSVLogger, class_counts, evaluate, train_one_epoch


def run_once(variant, loss_name, loaders, num_classes, counts, args, device):
    train_loader, val_loader, _, _ = loaders
    criterion = build_criterion(loss_name, num_classes, counts, device)
    model = build_model(variant, num_classes).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=config.MOMENTUM,
                                weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=args.epochs)
    best = 0.0
    for epoch in range(args.epochs):
        train_one_epoch(model, train_loader, criterion, optimizer, device)
        _, acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        if hasattr(criterion, "update_dynamic_weights"):
            criterion.update_dynamic_weights()
        best = max(best, acc)
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default=config.DATA_ROOT)
    p.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    p.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    p.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--num_workers", type=int, default=config.NUM_WORKERS)
    p.add_argument("--losses", nargs="+",
                   default=["ce", "weighted_ce", "focal", "aore"])
    args = p.parse_args()

    device = config.DEVICE
    loaders = get_loaders(args.data_root, args.batch_size, args.num_workers)
    num_classes = loaders[3]
    counts = class_counts(loaders[0].dataset)

    logger = CSVLogger(
        f"{config.OUTPUT_DIR}/loss_comparison.csv",
        ["loss", "epochs", "lr", "best_val_acc"])

    # Full SA-ONet under different loss functions.
    for loss_name in args.losses:
        print(f"\n=== SA-ONet with {loss_name} ===")
        best = run_once("sa_onet", loss_name, loaders, num_classes, counts,
                        args, device)
        print(f"{loss_name}: best val acc = {best:.4f}")
        logger.log(loss=loss_name, epochs=args.epochs, lr=args.lr,
                   best_val_acc=round(best, 6))
    logger.close()


if __name__ == "__main__":
    main()
