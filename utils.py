# -*- coding: utf-8 -*-
"""Shared training / evaluation utilities."""
import csv
import os

import torch

import config


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    maxk = max(topk)
    _, pred = output.topk(maxk, dim=1)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) / target.size(0)
            for k in topk]


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        bs = targets.size(0)
        total_loss += loss.item() * bs
        total_correct += (outputs.argmax(1) == targets).sum().item()
        total += bs
    return total_loss / total, total_correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, targets)

        bs = targets.size(0)
        total_loss += loss.item() * bs
        total_correct += (outputs.argmax(1) == targets).sum().item()
        total += bs
    return total_loss / total, total_correct / total


class CSVLogger:
    """Append-only CSV logger for experiment results."""

    def __init__(self, path: str, fieldnames):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.file = open(path, "a", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=fieldnames)
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            self.writer.writeheader()

    def log(self, **kwargs):
        self.writer.writerow(kwargs)
        self.file.flush()

    def close(self):
        self.file.close()


def class_counts(dataset) -> list:
    """Per-class sample counts of an ImageFolder dataset (for Aore-Loss)."""
    counts = [0] * len(dataset.classes)
    for _, label in dataset.samples:
        counts[label] += 1
    return counts


def get_device():
    return config.DEVICE
