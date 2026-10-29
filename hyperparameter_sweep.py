# -*- coding: utf-8 -*-
"""Hyper-parameter sweeps used in the paper's hyper-parameter analysis:

* training epochs 10 - 100 (step 10)   -> run_sweep("epochs")
* SE reduction ratio r in {8, 16, 32}  -> run_sweep("reduction")
* learning rates                        -> run_sweep("lr")

Usage examples:
    python hyperparameter_sweep.py --sweep epochs --data_root ./data
    python hyperparameter_sweep.py --sweep reduction --data_root ./data
    python hyperparameter_sweep.py --sweep lr --data_root ./data --lr_list 0.0001 0.001 0.01
"""
import argparse

import torch

import config
from dataset import get_loaders
from losses import build_criterion
from model import build_model
from utils import CSVLogger, class_counts, evaluate, train_one_epoch


def train_and_eval(loaders, num_classes, counts, args, device,
                   epochs=None, lr=None, reduction=None):
    train_loader, val_loader, _, _ = loaders
    criterion = build_criterion("aore", num_classes, counts, device)
    model = build_model("sa_onet", num_classes,
                        se_reduction=reduction or args.reduction).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr or args.lr,
                                momentum=config.MOMENTUM,
                                weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs or args.epochs)

    best = 0.0
    for epoch in range(epochs or args.epochs):
        train_one_epoch(model, train_loader, criterion, optimizer, device)
        _, acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        if hasattr(criterion, "update_dynamic_weights"):
            criterion.update_dynamic_weights()
        best = max(best, acc)
    return best


def run_sweep(sweep, args):
    device = config.DEVICE
    loaders = get_loaders(args.data_root, args.batch_size, args.num_workers)
    num_classes = loaders[3]
    counts = class_counts(loaders[0].dataset)

    fieldnames = ["sweep", "value", "best_val_acc"]
    logger = CSVLogger(f"{config.OUTPUT_DIR}/sweep_{sweep}.csv", fieldnames)

    if sweep == "epochs":
        values = [e for e in range(10, 101, 10)]
        for v in values:
            print(f"\n=== epochs = {v} ===")
            best = train_and_eval(loaders, num_classes, counts, args, device,
                                  epochs=v)
            logger.log(sweep=sweep, value=v, best_val_acc=round(best, 6))
    elif sweep == "reduction":
        values = [8, 16, 32]
        for v in values:
            print(f"\n=== SE reduction r = {v} ===")
            best = train_and_eval(loaders, num_classes, counts, args, device,
                                  reduction=v)
            logger.log(sweep=sweep, value=v, best_val_acc=round(best, 6))
    elif sweep == "lr":
        values = args.lr_list or [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
        for v in values:
            print(f"\n=== lr = {v} ===")
            best = train_and_eval(loaders, num_classes, counts, args, device,
                                  lr=v)
            logger.log(sweep=sweep, value=v, best_val_acc=round(best, 6))
    else:
        raise ValueError(f"Unknown sweep: {sweep}")
    logger.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep", type=str, required=True,
                   choices=["epochs", "reduction", "lr"])
    p.add_argument("--data_root", type=str, default=config.DATA_ROOT)
    p.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    p.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    p.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--reduction", type=int, default=config.SE_REDUCTION)
    p.add_argument("--num_workers", type=int, default=config.NUM_WORKERS)
    p.add_argument("--lr_list", nargs="+", type=float, default=None)
    args = p.parse_args()
    run_sweep(args.sweep, args)


if __name__ == "__main__":
    main()
