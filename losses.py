# -*- coding: utf-8 -*-
"""Loss functions: CrossEntropy, Weighted CE, Focal Loss and the proposed
Aore-Loss (Eq.(6)-(9) of the paper)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class AoreLoss(nn.Module):
    """Aore-Loss: a difficulty-aware re-weighted cross entropy.

    LAore = -(1/N) * sum_i sum_c w_c * y_ic * log(y_hat_ic)          (Eq.6)

    Two weighting strategies are combined:

    * static class weights from inverse class frequency, normalized
      over all classes (Eq.7 / Eq.8):
          w_c(freq)  = (1/n_c) / sum_j (1/n_j)
    * dynamic per-class weights updated at the end of every epoch from the
      average loss of each class in the previous epoch (Eq.9):
          w_c(dynamic) = 1 / (1 + exp(-beta * loss_c))
      normalized so that mean(w) == 1 to keep the loss scale stable.
    """

    def __init__(self, num_classes: int, class_counts=None, beta: float = 1.0,
                 use_dynamic_weight: bool = True, device=None):
        super().__init__()
        self.num_classes = num_classes
        self.beta = beta
        self.use_dynamic_weight = use_dynamic_weight

        if class_counts is not None:
            counts = torch.as_tensor(class_counts, dtype=torch.float)
        else:
            counts = torch.ones(num_classes)

        # Eq.(7)-(8): normalized inverse-frequency weights
        inv_freq = 1.0 / counts.clamp(min=1)
        static_w = inv_freq / inv_freq.sum() * num_classes

        if device is not None:
            static_w = static_w.to(device)
        self.register_buffer("static_weight", static_w)
        self.register_buffer("dynamic_weight", torch.ones(num_classes))
        if device is not None:
            self.dynamic_weight = self.dynamic_weight.to(device)

        self._per_class_loss_sum = torch.zeros(num_classes)
        self._per_class_count = torch.zeros(num_classes)

    # ------------------------------------------------------------------
    def forward(self, logits, targets):
        weight = self.static_weight
        if self.use_dynamic_weight:
            weight = weight * self.dynamic_weight
        weight = weight.to(logits.device)

        log_probs = F.log_softmax(logits, dim=1)
        loss = -(weight[targets] * log_probs.gather(1, targets.unsqueeze(1)).squeeze(1))

        with torch.no_grad():
            one_hot = F.one_hot(targets, self.num_classes).float()
            probs = torch.exp(log_probs)  # predicted probability of gt class
            nll = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            self._per_class_loss_sum += (one_hot * nll.unsqueeze(1)).sum(0).cpu()
            self._per_class_count += one_hot.sum(0).cpu()
        return loss.mean()

    # ------------------------------------------------------------------
    @torch.no_grad()
    def update_dynamic_weights(self):
        """Called at the end of each training epoch. Uses the average loss of
        each class from the previous epoch to re-compute w_c (Eq.9)."""
        if not self.use_dynamic_weight:
            return
        seen = self._per_class_count > 0
        if seen.sum() == 0:
            return
        avg_loss = torch.zeros(self.num_classes)
        avg_loss[seen] = self._per_class_loss_sum[seen] / self._per_class_count[seen]

        w = 1.0 / (1.0 + torch.exp(-self.beta * avg_loss))  # Eq.(9)
        w = w / w.mean().clamp(min=1e-8)                    # normalize to mean 1

        device = self.dynamic_weight.device
        self.dynamic_weight = w.to(device)
        self._per_class_loss_sum.zero_()
        self._per_class_count.zero_()


# ----------------------------------------------------------------------
def build_criterion(name: str, num_classes: int, class_counts=None, device=None):
    """Factory used by the experiment scripts.

    name in {"ce", "weighted_ce", "focal", "aore"}
    """
    name = name.lower()
    if name == "ce":
        return nn.CrossEntropyLoss()
    if name == "weighted_ce":
        counts = class_counts if class_counts is not None else [1] * num_classes
        w = torch.tensor(counts, dtype=torch.float)
        w = 1.0 / w.clamp(min=1)
        w = w / w.sum() * num_classes
        return nn.CrossEntropyLoss(weight=w.to(device) if device is not None else w)
    if name == "focal":
        return FocalLoss(num_classes=num_classes)
    if name == "aore":
        return AoreLoss(num_classes, class_counts=class_counts,
                        beta=config.AORE_BETA,
                        use_dynamic_weight=config.USE_DYNAMIC_WEIGHT,
                        device=device)
    raise ValueError(f"Unknown loss: {name}")


class FocalLoss(nn.Module):
    """Focal Loss (Lin et al., 2017) used as a comparison baseline."""

    def __init__(self, num_classes: int, gamma: float = 2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        if alpha is None:
            self.alpha = None
        else:
            self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float))

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal = -((1 - pt) ** self.gamma) * log_probs.gather(
            1, targets.unsqueeze(1)).squeeze(1)
        if self.alpha is not None:
            focal = self.alpha[targets] * focal
        return focal.mean()
