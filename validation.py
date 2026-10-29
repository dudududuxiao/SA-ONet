# -*- coding: utf-8 -*-
"""Validate a trained SA-ONet checkpoint on the validation set.

Usage:
    python validation.py --checkpoint outputs/best_model_main.pth --data_root ./data
"""
import argparse

import torch

import config
from dataset import get_loaders
from model import build_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data_root", type=str, default=config.DATA_ROOT)
    p.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    args = p.parse_args()

    device = config.DEVICE
    _, val_loader, _, num_classes = get_loaders(args.data_root,
                                                args.batch_size)

    model = build_model("sa_onet", num_classes).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    model.eval()
    correct, total = 0, 0
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    with torch.no_grad():
        for images, targets in val_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            preds = outputs.argmax(1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            for t, pr in zip(targets, preds):
                class_total[t.item()] += 1
                class_correct[t.item()] += int(t == pr)

    print(f"Validation accuracy: {correct / total:.4f} ({correct}/{total})")
    names = val_loader.dataset.classes
    for i in range(num_classes):
        if class_total[i]:
            name = names[i] if i < len(names) else str(i)
            print(f"  class {i} ({name}): {class_correct[i] / class_total[i]:.4f}")


if __name__ == "__main__":
    main()
