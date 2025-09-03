"""
Custom loss functions for engineering applications.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Union, Tuple
import math


class EngineeringLoss(nn.Module):
    """
    Base engineering loss function.
    """
    
    def __init__(self, alpha: float = 1.0, beta: float = 0.0, gamma: float = 1.0):
        """
        Initialize engineering loss.
        
        Args:
            alpha: Primary weight
            beta: Secondary weight
            gamma: Tertiary weight
        """
        super(EngineeringLoss, self).__init__()
        
        self.logger = logging.getLogger(__name__)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        raise NotImplementedError("Subclasses must implement forward method")


class FocalLoss(EngineeringLoss):
    """
    Focal Loss for addressing class imbalance.
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor
            gamma: Focusing parameter
            reduction: Reduction method
        """
        super(FocalLoss, self).__init__(alpha=alpha, gamma=gamma)
        
        self.reduction = reduction
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Compute cross entropy
        ce_loss = F.cross_entropy(input, target, reduction='none')
        
        # Compute p_t
        pt = torch.exp(-ce_loss)
        
        # Compute focal loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DiceLoss(EngineeringLoss):
    """
    Dice Loss for segmentation tasks.
    """
    
    def __init__(self, alpha: float = 1.0, smooth: float = 1e-5, reduction: str = 'mean'):
        """
        Initialize Dice Loss.
        
        Args:
            alpha: Weighting factor
            smooth: Smoothing factor
            reduction: Reduction method
        """
        super(DiceLoss, self).__init__(alpha=alpha)
        
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Apply sigmoid to input
        input = torch.sigmoid(input)
        
        # Flatten tensors
        input = input.view(-1)
        target = target.view(-1)
        
        # Compute intersection
        intersection = (input * target).sum()
        
        # Compute dice coefficient
        dice = (2. * intersection + self.smooth) / (input.sum() + target.sum() + self.smooth)
        
        # Compute dice loss
        dice_loss = 1 - dice
        
        return dice_loss


class IoULoss(EngineeringLoss):
    """
    Intersection over Union (IoU) Loss.
    """
    
    def __init__(self, alpha: float = 1.0, smooth: float = 1e-5, reduction: str = 'mean'):
        """
        Initialize IoU Loss.
        
        Args:
            alpha: Weighting factor
            smooth: Smoothing factor
            reduction: Reduction method
        """
        super(IoULoss, self).__init__(alpha=alpha)
        
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Apply sigmoid to input
        input = torch.sigmoid(input)
        
        # Flatten tensors
        input = input.view(-1)
        target = target.view(-1)
        
        # Compute intersection
        intersection = (input * target).sum()
        
        # Compute union
        union = input.sum() + target.sum() - intersection
        
        # Compute IoU
        iou = (intersection + self.smooth) / (union + self.smooth)
        
        # Compute IoU loss
        iou_loss = 1 - iou
        
        return iou_loss


class TverskyLoss(EngineeringLoss):
    """
    Tversky Loss for segmentation tasks.
    """
    
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1e-5, reduction: str = 'mean'):
        """
        Initialize Tversky Loss.
        
        Args:
            alpha: False negative weight
            beta: False positive weight
            smooth: Smoothing factor
            reduction: Reduction method
        """
        super(TverskyLoss, self).__init__(alpha=alpha, beta=beta)
        
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Apply sigmoid to input
        input = torch.sigmoid(input)
        
        # Flatten tensors
        input = input.view(-1)
        target = target.view(-1)
        
        # Compute intersection
        intersection = (input * target).sum()
        
        # Compute false negatives and false positives
        false_negatives = (target * (1 - input)).sum()
        false_positives = ((1 - target) * input).sum()
        
        # Compute Tversky coefficient
        tversky = (intersection + self.smooth) / (intersection + self.alpha * false_negatives + self.beta * false_positives + self.smooth)
        
        # Compute Tversky loss
        tversky_loss = 1 - tversky
        
        return tversky_loss


class FocalTverskyLoss(EngineeringLoss):
    """
    Focal Tversky Loss combining Focal and Tversky losses.
    """
    
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, gamma: float = 1.0, smooth: float = 1e-5, reduction: str = 'mean'):
        """
        Initialize Focal Tversky Loss.
        
        Args:
            alpha: False negative weight
            beta: False positive weight
            gamma: Focusing parameter
            smooth: Smoothing factor
            reduction: Reduction method
        """
        super(FocalTverskyLoss, self).__init__(alpha=alpha, beta=beta, gamma=gamma)
        
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Apply sigmoid to input
        input = torch.sigmoid(input)
        
        # Flatten tensors
        input = input.view(-1)
        target = target.view(-1)
        
        # Compute intersection
        intersection = (input * target).sum()
        
        # Compute false negatives and false positives
        false_negatives = (target * (1 - input)).sum()
        false_positives = ((1 - target) * input).sum()
        
        # Compute Tversky coefficient
        tversky = (intersection + self.smooth) / (intersection + self.alpha * false_negatives + self.beta * false_positives + self.smooth)
        
        # Compute Focal Tversky loss
        focal_tversky_loss = (1 - tversky) ** self.gamma
        
        return focal_tversky_loss


class LovaszLoss(EngineeringLoss):
    """
    Lovász Loss for segmentation tasks.
    """
    
    def __init__(self, alpha: float = 1.0, reduction: str = 'mean'):
        """
        Initialize Lovász Loss.
        
        Args:
            alpha: Weighting factor
            reduction: Reduction method
        """
        super(LovaszLoss, self).__init__(alpha=alpha)
        
        self.reduction = reduction
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Apply sigmoid to input
        input = torch.sigmoid(input)
        
        # Flatten tensors
        input = input.view(-1)
        target = target.view(-1)
        
        # Compute Lovász loss
        lovasz_loss = self._lovasz_hinge(input, target)
        
        return lovasz_loss
    
    def _lovasz_hinge(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute Lovász hinge loss."""
        if len(labels) == 0:
            return logits.sum() * 0
        
        signs = 2. * labels - 1.
        errors = (1. - logits * signs)
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        perm = perm.data
        gt_sorted = labels[perm]
        grad = self._lovasz_grad(gt_sorted)
        loss = torch.dot(F.relu(errors_sorted), grad)
        return loss
    
    def _lovasz_grad(self, gt_sorted: torch.Tensor) -> torch.Tensor:
        """Compute Lovász gradient."""
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        if gts == 0:
            return gt_sorted.new_zeros(p)
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1. - intersection / union
        if p > 1:
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
        return jaccard


class BoundaryLoss(EngineeringLoss):
    """
    Boundary Loss for segmentation tasks.
    """
    
    def __init__(self, alpha: float = 1.0, reduction: str = 'mean'):
        """
        Initialize Boundary Loss.
        
        Args:
            alpha: Weighting factor
            reduction: Reduction method
        """
        super(BoundaryLoss, self).__init__(alpha=alpha)
        
        self.reduction = reduction
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Apply sigmoid to input
        input = torch.sigmoid(input)
        
        # Compute boundary loss
        boundary_loss = self._compute_boundary_loss(input, target)
        
        return boundary_loss
    
    def _compute_boundary_loss(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute boundary loss."""
        # Compute gradients
        input_grad_x = torch.abs(input[:, :, :, 1:] - input[:, :, :, :-1])
        input_grad_y = torch.abs(input[:, :, 1:, :] - input[:, :, :-1, :])
        
        target_grad_x = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
        target_grad_y = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])
        
        # Compute boundary loss
        boundary_loss = torch.mean(input_grad_x * target_grad_x) + torch.mean(input_grad_y * target_grad_y)
        
        return boundary_loss


class HausdorffLoss(EngineeringLoss):
    """
    Hausdorff Loss for segmentation tasks.
    """
    
    def __init__(self, alpha: float = 1.0, reduction: str = 'mean'):
        """
        Initialize Hausdorff Loss.
        
        Args:
            alpha: Weighting factor
            reduction: Reduction method
        """
        super(HausdorffLoss, self).__init__(alpha=alpha)
        
        self.reduction = reduction
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Apply sigmoid to input
        input = torch.sigmoid(input)
        
        # Compute Hausdorff loss
        hausdorff_loss = self._compute_hausdorff_loss(input, target)
        
        return hausdorff_loss
    
    def _compute_hausdorff_loss(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Hausdorff loss."""
        # Compute distance transform
        input_dt = self._distance_transform(input)
        target_dt = self._distance_transform(target)
        
        # Compute Hausdorff loss
        hausdorff_loss = torch.mean(input_dt * target) + torch.mean(target_dt * input)
        
        return hausdorff_loss
    
    def _distance_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Compute distance transform."""
        # Simple distance transform implementation
        # In practice, you might want to use a more efficient implementation
        return torch.abs(x - 1.0)


class ContrastiveLoss(EngineeringLoss):
    """
    Contrastive Loss for representation learning.
    """
    
    def __init__(self, alpha: float = 1.0, margin: float = 1.0, reduction: str = 'mean'):
        """
        Initialize Contrastive Loss.
        
        Args:
            alpha: Weighting factor
            margin: Margin for negative pairs
            reduction: Reduction method
        """
        super(ContrastiveLoss, self).__init__(alpha=alpha)
        
        self.margin = margin
        self.reduction = reduction
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Compute pairwise distances
        distances = torch.pdist(input, p=2)
        
        # Compute contrastive loss
        positive_loss = distances ** 2
        negative_loss = F.relu(self.margin - distances) ** 2
        
        # Combine losses
        contrastive_loss = positive_loss + negative_loss
        
        if self.reduction == 'mean':
            return contrastive_loss.mean()
        elif self.reduction == 'sum':
            return contrastive_loss.sum()
        else:
            return contrastive_loss


class TripletLoss(EngineeringLoss):
    """
    Triplet Loss for representation learning.
    """
    
    def __init__(self, alpha: float = 1.0, margin: float = 1.0, reduction: str = 'mean'):
        """
        Initialize Triplet Loss.
        
        Args:
            alpha: Weighting factor
            margin: Margin for negative pairs
            reduction: Reduction method
        """
        super(TripletLoss, self).__init__(alpha=alpha)
        
        self.margin = margin
        self.reduction = reduction
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Compute pairwise distances
        distances = torch.pdist(input, p=2)
        
        # Compute triplet loss
        positive_distances = distances[::2]  # Even indices
        negative_distances = distances[1::2]  # Odd indices
        
        triplet_loss = F.relu(positive_distances - negative_distances + self.margin)
        
        if self.reduction == 'mean':
            return triplet_loss.mean()
        elif self.reduction == 'sum':
            return triplet_loss.sum()
        else:
            return triplet_loss


class CenterLoss(EngineeringLoss):
    """
    Center Loss for representation learning.
    """
    
    def __init__(self, alpha: float = 1.0, num_classes: int = 10, feature_dim: int = 2, reduction: str = 'mean'):
        """
        Initialize Center Loss.
        
        Args:
            alpha: Weighting factor
            num_classes: Number of classes
            feature_dim: Feature dimension
            reduction: Reduction method
        """
        super(CenterLoss, self).__init__(alpha=alpha)
        
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.reduction = reduction
        
        # Initialize centers
        self.centers = nn.Parameter(torch.randn(num_classes, feature_dim))
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Compute center loss
        center_loss = torch.mean(torch.sum((input - self.centers[target]) ** 2, dim=1))
        
        return center_loss


class AngularLoss(EngineeringLoss):
    """
    Angular Loss for representation learning.
    """
    
    def __init__(self, alpha: float = 1.0, margin: float = 0.5, reduction: str = 'mean'):
        """
        Initialize Angular Loss.
        
        Args:
            alpha: Weighting factor
            margin: Margin for negative pairs
            reduction: Reduction method
        """
        super(AngularLoss, self).__init__(alpha=alpha)
        
        self.margin = margin
        self.reduction = reduction
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Normalize input
        input = F.normalize(input, p=2, dim=1)
        
        # Compute pairwise cosine similarities
        similarities = torch.mm(input, input.t())
        
        # Compute angular loss
        angular_loss = torch.mean(torch.acos(torch.clamp(similarities, -1, 1)))
        
        return angular_loss
