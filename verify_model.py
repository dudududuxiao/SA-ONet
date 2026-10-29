# -*- coding: utf-8 -*-
"""Quick smoke test: builds the model and runs a forward/backward pass on
random data. Useful to verify the installation without the dataset.

Usage:
    python verify_model.py
"""
import torch

import config
from losses import AoreLoss, FocalLoss
from model import build_model


def main():
    device = config.DEVICE
    print(f"Device: {device}")
    for variant in ["resnet50", "resnet50_fpn", "resnet50_se", "sa_onet"]:
        model = build_model(variant, num_classes=7).to(device)
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        x = torch.randn(2, 3, config.IMG_SIZE, config.IMG_SIZE, device=device)
        y = torch.randint(0, 7, (2,), device=device)
        out = model(x)
        loss = AoreLoss(7)(out, y)
        loss.backward()
        print(f"{variant:12s} params={n_params:.1f}M out={tuple(out.shape)} "
              f"loss={loss.item():.4f} OK")

    # Check every loss factory.
    logits = torch.randn(4, 7)
    labels = torch.randint(0, 7, (4,))
    for crit in [AoreLoss(7), FocalLoss(7), torch.nn.CrossEntropyLoss()]:
        crit(logits, labels)
    print("All losses OK.")


if __name__ == "__main__":
    main()
