"""
Neural Networks module for engineering applications.
"""

from .neural_module import NeuralModule
from .architectures import EngineeringNet, StructuralNet, FluidNet, MaterialNet, ControlNet
from .layers import EngineeringLayer, AdaptiveLayer, ResidualLayer, AttentionLayer
from .activations import EngineeringActivation, Swish, Mish, GELU
from .optimizers import EngineeringOptimizer, AdamW, RAdam, AdaBelief
from .losses import EngineeringLoss, FocalLoss, DiceLoss, IoULoss
from .training import NeuralTrainer, DistributedTrainer, FederatedTrainer

__all__ = [
    "NeuralModule",
    "EngineeringNet",
    "StructuralNet",
    "FluidNet",
    "MaterialNet",
    "ControlNet",
    "EngineeringLayer",
    "AdaptiveLayer",
    "ResidualLayer",
    "AttentionLayer",
    "EngineeringActivation",
    "Swish",
    "Mish",
    "GELU",
    "EngineeringOptimizer",
    "AdamW",
    "RAdam",
    "AdaBelief",
    "EngineeringLoss",
    "FocalLoss",
    "DiceLoss",
    "IoULoss",
    "NeuralTrainer",
    "DistributedTrainer",
    "FederatedTrainer"
]
