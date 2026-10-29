# SA-ONet: Scale Attentive Ore Network for Multi-Classification of X-ray Images of Tungsten Ores

Official PyTorch implementation of **SA-ONet** proposed in *"Multi-Scale Attention Enhanced Network for Multi-Classification of X-ray Images of Tungsten Ores"*.

SA-ONet consists of two modules and an improved loss function:

- **MSFEM** (Multi-Scale Feature Extraction Module): a ResNet50 backbone integrated with a Feature Pyramid Network (FPN). Hierarchical features C2–C5 are extracted (C1 is not considered), projected by 1×1 convolutions, upsampled with bilinear interpolation (2×), fused through top-down lateral connections, and smoothed by 3×3 convolutions to produce pyramid features **P2–P5**.
- **AAPM** (Adaptive Attention Perception Module): an independent **SE block** on each pyramid level, followed by Feature Concatenation, Global Average Pooling (GAP) and a Fully Connected (FC) classifier.
- **Aore-Loss**: a difficulty-aware re-weighted cross entropy. Static class weights use normalized inverse class frequency (Eq.7–8), and dynamic per-class weights are updated from the average loss of the previous epoch via `1/(1+exp(-β·loss_c))` (Eq.9).

> ⚠️ **Dataset notice**: the tungsten ore X-ray dataset is proprietary and **not distributed** with this repository. Only the training / validation / test code is released. Image preprocessing code is excluded for commercial confidentiality reasons.

## Repository Structure

```
SA-ONet/
├── config.py                  # Global configuration (paths, hyper-parameters)
├── dataset.py                 # ImageFolder-style dataset / DataLoader
├── model.py                   # SA-ONet (ResNet50 + FPN + SE) and ablation variants
├── losses.py                  # Aore-Loss, Focal Loss, (Weighted) CrossEntropy
├── utils.py                   # Training / evaluation helpers, CSV logger
├── train.py                   # Main training script
├── validation.py              # Validation on the val set (per-class accuracy)
├── test.py                    # Test on the test set (Top-1 / Top-5, confusion matrix)
├── verify_model.py            # Smoke test without the dataset
├── ablation_study.py          # Ablation: ResNet50 / +FPN / +SE / full SA-ONet
├── comparison_experiment.py   # Loss comparison: CE / Weighted-CE / Focal / Aore
├── hyperparameter_sweep.py    # Sweeps: epochs 10-100, SE r {8,16,32}, learning rates
├── requirements.txt
└── .gitignore
```

## Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0 (CUDA recommended)
- torchvision ≥ 0.15

```bash
pip install -r requirements.txt
```

## Dataset Preparation

Organize the X-ray images in an ImageFolder layout:

```
data_root/
├── train/
│   ├── pure_black_ore/*.jpg
│   ├── obvious_large_spots/*.jpg
│   ├── ...
├── val/
│   └── ...
└── test/
    └── ...
```

Then either edit `DATA_ROOT` in `config.py` or set the environment variable:

```bash
# Windows (PowerShell)
$env:SAONET_DATA_ROOT = "D:\path\to\your\data"
# Linux
export SAONET_DATA_ROOT=/path/to/your/data
```

## Quick Start

Verify the installation (no dataset required):

```bash
python verify_model.py
```

Train the full model (default: 70 epochs, lr = 0.001, SE reduction r = 16, Aore-Loss):

```bash
python train.py --data_root ./data --epochs 70 --lr 0.001 --reduction 16 --loss aore
```

Validate and test a checkpoint:

```bash
python validation.py --checkpoint outputs/best_model_main.pth --data_root ./data
python test.py       --checkpoint outputs/best_model_main.pth --data_root ./data
```

Key arguments of `train.py`:

| Argument | Default | Description |
|---|---|---|
| `--epochs` | 70 | Number of training epochs |
| `--lr` | 0.001 | Initial learning rate (SGD, cosine annealing) |
| `--reduction` | 16 | SE channel reduction ratio r (8 / 16 / 32) |
| `--loss` | aore | `ce` / `weighted_ce` / `focal` / `aore` |
| `--variant` | sa_onet | `sa_onet` / `resnet50` / `resnet50_fpn` / `resnet50_se` |
| `--beta` | 1.0 | Sensitivity β of the dynamic weight in Aore-Loss |

## Ablation Study

Verifies the contribution of the FPN (MSFEM) and the SE blocks (AAPM):

```bash
python ablation_study.py --data_root ./data --epochs 70
```

Variants: `resnet50` (baseline), `resnet50_fpn` (backbone + FPN, no SE), `resnet50_se` (backbone + SE, no FPN), `sa_onet` (full model). Results are written to `outputs/ablation_study.csv`.

## Loss Function Comparison

```bash
python comparison_experiment.py --data_root ./data --epochs 70
```

Compares the proposed Aore-Loss with CrossEntropy, Weighted CrossEntropy and Focal Loss. Results are written to `outputs/loss_comparison.csv`.

## Hyper-parameter Sweeps

```bash
# Training epochs: 10, 20, ..., 100
python hyperparameter_sweep.py --sweep epochs --data_root ./data

# SE reduction ratio: 8, 16, 32
python hyperparameter_sweep.py --sweep reduction --data_root ./data

# Learning rates (default: 1e-4, 5e-4, 1e-3, 5e-3, 1e-2)
python hyperparameter_sweep.py --sweep lr --data_root ./data --lr_list 0.0001 0.001 0.01
```

Each sweep appends `sweep, value, best_val_acc` rows to `outputs/sweep_<name>.csv`.

## Aore-Loss Details

```
L_Aore = -(1/N) Σ_i Σ_c  w_c · y_ic · log(ŷ_ic)          (Eq.6)

static weights (Eq.7-8):
    w_c(freq) = (1/n_c) / Σ_j (1/n_j)

dynamic weights, updated after each epoch (Eq.9):
    w_c(dynamic) = 1 / (1 + exp(-β · loss_c))
```

The dynamic weight is normalized to unit mean so the overall loss scale stays stable during training. It is enabled by default (`config.USE_DYNAMIC_WEIGHT = True`).

## Notes

- Backbone weights are initialized from ImageNet pre-training (`ResNet50_Weights.IMAGENET1K_V2`). Set `PRETRAINED_BACKBONE = False` in `config.py` to train from scratch.
- Bilinear interpolation (align-corner-free) is used for the 2× upsampling, following the paper.
- All results (CSV logs, best checkpoints) are saved under `outputs/`.
