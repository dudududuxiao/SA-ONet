# -*- coding: utf-8 -*-
"""Evaluate a trained SA-ONet checkpoint on the test set (Top-1 / Top-5).

Usage:
    python test.py --checkpoint outputs/best_model_main.pth --data_root ./data
"""
import argparse

import torch

import config
from dataset import get_loaders
from model import build_model
from utils import accuracy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data_root", type=str, default=config.DATA_ROOT)
    p.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    args = p.parse_args()

    device = config.DEVICE
    _, _, test_loader, num_classes = get_loaders(args.data_root,
                                                 args.batch_size)

    model = build_model("sa_onet", num_classes).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    model.eval()
    correct1, correct5, total = 0, 0, 0
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            correct1 += accuracy(outputs, targets, topk=(1,))[0].item() * targets.size(0)
            k5 = min(5, num_classes)
            correct5 += accuracy(outputs, targets, topk=(k5,))[0].item() * targets.size(0)
            preds = outputs.argmax(1)
            for t, pr in zip(targets, preds):
                confusion[t.item(), pr.item()] += 1
            total += targets.size(0)

    print(f"Test Top-1 accuracy: {correct1 / total:.4f} ({correct1:.0f}/{total})")
    print(f"Test Top-{min(5, num_classes)} accuracy: {correct5 / total:.4f}")
    print("Confusion matrix (rows=ground truth, cols=prediction):")
    print(confusion)


if __name__ == "__main__":
    main()
