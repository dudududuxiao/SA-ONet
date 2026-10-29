# -*- coding: utf-8 -*-
"""SA-ONet model and ablation variants.

SA-ONet = ResNet50 backbone + FPN (MSFEM) + SE attention (AAPM) + classifier.

Forward path (matching Fig.5 of the paper):
    input -> ResNet50 -> C2,C3,C4,C5  (C1 is not considered)
          -> 1x1 conv + 2x upsample on every level
          -> top-down lateral connections + element-wise fusion
          -> 3x3 conv (stride 1, 4 in/out via padding) -> P2..P5
          -> independent SE block on each P_k
          -> Feature Concatenation -> Global Average Pooling -> FC -> softmax
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

import config


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block (Eq.(3)-(5))."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        z = F.adaptive_avg_pool2d(x, 1).view(b, c)          # Fsq  (Eq.3)
        s = self.fc(z).view(b, c, 1, 1)                     # Fex  (Eq.4)
        return x * s                                        # Fscale (Eq.5)


class MSFEM(nn.Module):
    """Multi-Scale Feature Extraction Module: ResNet50 + FPN."""

    def __init__(self, in_channels_list=(256, 512, 1024, 2048),
                 out_channels: int = 256, pretrained: bool = True):
        super().__init__()
        backbone = resnet50(
            weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)

        # Stem (C1) is kept for the forward pass but its output is not used
        # in the pyramid, as stated in the paper ("C1 is not considered").
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu,
                                  backbone.maxpool)
        self.layer1 = backbone.layer1   # C2: 256  x H/4
        self.layer2 = backbone.layer2   # C3: 512  x H/8
        self.layer3 = backbone.layer3   # C4: 1024 x H/16
        self.layer4 = backbone.layer4   # C5: 2048 x H/32

        self.lateral_convs = nn.ModuleList(
            nn.Conv2d(c, out_channels, kernel_size=1)
            for c in in_channels_list)
        self.fusion_convs = nn.ModuleList(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in in_channels_list)

    def forward(self, x):
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        # Eq.(2): 1x1 conv then bilinear 2x upsample, top-down laterals.
        p5 = self.lateral_convs[3](c5)
        p4 = self.lateral_convs[2](c4) + F.interpolate(
            p5, size=c4.shape[-2:], mode="bilinear", align_corners=False)
        p3 = self.lateral_convs[1](c3) + F.interpolate(
            p4, size=c3.shape[-2:], mode="bilinear", align_corners=False)
        p2 = self.lateral_convs[0](c2) + F.interpolate(
            p3, size=c2.shape[-2:], mode="bilinear", align_corners=False)

        p2 = self.fusion_convs[0](p2)
        p3 = self.fusion_convs[1](p3)
        p4 = self.fusion_convs[2](p4)
        p5 = self.fusion_convs[3](p5)
        return p2, p3, p4, p5


class SAONet(nn.Module):
    """Full SA-ONet (ResNet50 + FPN + SE)."""

    def __init__(self, num_classes: int = None, fpn_channels: int = None,
                 se_reduction: int = None, pretrained: bool = None,
                 use_fpn: bool = True, use_se: bool = True):
        super().__init__()
        num_classes = num_classes or config.NUM_CLASSES
        fpn_channels = fpn_channels or config.FPN_OUT_CHANNELS
        se_reduction = se_reduction or config.SE_REDUCTION
        if pretrained is None:
            pretrained = config.PRETRAINED_BACKBONE

        self.msfem = MSFEM(out_channels=fpn_channels, pretrained=pretrained)
        self.use_fpn = use_fpn
        self.use_se = use_se
        # Without FPN the pooled feature has 2048 channels (backbone output).
        se_channels = fpn_channels if use_fpn else 2048
        if use_se:
            self.se_blocks = nn.ModuleList(
                SEBlock(se_channels, se_reduction) for _ in range(4))

        feat_dim = fpn_channels * 4 if use_fpn else 2048
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        if self.use_fpn:
            feats = self.msfem(x)                      # P2..P5
            if self.use_se:                            # AAPM: SE per level
                feats = [se(p) for se, p in zip(self.se_blocks, feats)]
            feats = [F.adaptive_avg_pool2d(p, 1).flatten(1) for p in feats]
            feats = torch.cat(feats, dim=1)            # Feature Concatenation
        else:
            # Ablation: backbone only (no FPN)
            x = self.msfem.stem(x)
            feats = self.msfem.layer4(self.msfem.layer3(
                self.msfem.layer2(self.msfem.layer1(x))))
            if self.use_se:
                feats = self.se_blocks[0](feats)
            feats = F.adaptive_avg_pool2d(feats, 1).flatten(1)
        return self.classifier(feats)                  # FC (+ softmax in loss)


def build_model(variant: str = "sa_onet", num_classes: int = None,
                se_reduction: int = None, pretrained: bool = None) -> SAONet:
    """Factory for the ablation / comparison experiments.

    variant:
        "sa_onet"       -> full model (default)
        "resnet50"      -> plain ResNet50 baseline
        "resnet50_fpn"  -> ResNet50 + FPN, without SE
        "resnet50_se"   -> ResNet50 + SE, without FPN
    """
    variant = variant.lower()
    if variant == "sa_onet":
        return SAONet(num_classes, se_reduction=se_reduction,
                      pretrained=pretrained)
    if variant == "resnet50":
        net = SAONet(num_classes, use_fpn=False, use_se=False,
                     pretrained=pretrained)
        return net
    if variant == "resnet50_fpn":
        return SAONet(num_classes, use_fpn=True, use_se=False,
                      pretrained=pretrained)
    if variant == "resnet50_se":
        return SAONet(num_classes, use_fpn=False, use_se=True,
                      se_reduction=se_reduction, pretrained=pretrained)
    raise ValueError(f"Unknown variant: {variant}")
