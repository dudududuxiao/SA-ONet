# -*- coding: utf-8 -*-
"""Global configuration for SA-ONet experiments."""
import os
import torch

# ==================== Dataset ====================
# NOTE: The tungsten ore X-ray dataset is proprietary and NOT distributed
# with this repository. Organize your data as ImageFolder-style directories:
#   data_root/
#     train/<class_name>/*.jpg
#     val/<class_name>/*.jpg
#     test/<class_name>/*.jpg
DATA_ROOT = os.environ.get("SAONET_DATA_ROOT", "./data")

# Class names of the fine-grained 7-class task (descending mineral content).
# For the 4-class task, modify accordingly.
CLASS_NAMES = [
    "pure_black_ore",
    "obvious_large_spots",
    "obvious_small_spots",
    "fine_spotted_ore",
    "weak_spots_ore",
    "low_gray_waste",
    "high_gray_waste",
]
NUM_CLASSES = len(CLASS_NAMES)

# ==================== Model ====================
FPN_OUT_CHANNELS = 256   # channel dimension of P2-P5
SE_REDUCTION = 16        # reduction ratio r of the SE block (best: 16)
PRETRAINED_BACKBONE = True

# ==================== Training ====================
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 70
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 5e-5
MOMENTUM = 0.9
NUM_WORKERS = 4

# ==================== Aore-Loss ====================
AORE_BETA = 1.0          # sensitivity of the dynamic difficulty weight
USE_DYNAMIC_WEIGHT = True  # Eq.(9): dynamic weight from previous-epoch loss

# ==================== Output ====================
OUTPUT_DIR = "./outputs"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
