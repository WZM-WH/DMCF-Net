import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        valid_mask = (target != -1).float()
        target = torch.where(target == -1, torch.tensor(0.0, device=target.device), target)
        pt = torch.where(target == 1, pred, 1 - pred)
        alpha_t = torch.where(target == 1, 1 - self.alpha, self.alpha)
        focal_weight = alpha_t * (1 - pt).pow(self.gamma)
        bce_loss = -torch.log(pt + 1e-6)
        focal_loss = focal_weight * bce_loss
        focal_loss = focal_loss * valid_mask
        if self.reduction == 'mean':
            return focal_loss.sum() / (valid_mask.sum() + 1e-6)
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            raise NotImplementedError(f"Reduction type {self.reduction} not implemented")