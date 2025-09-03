"""
Custom activation functions for engineering applications.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Union, Tuple
import math


class EngineeringActivation(nn.Module):
    """
    Base engineering activation function.
    """
    
    def __init__(self, alpha: float = 1.0, beta: float = 0.0):
        """
        Initialize engineering activation.
        
        Args:
            alpha: Scaling parameter
            beta: Shift parameter
        """
        super(EngineeringActivation, self).__init__()
        
        self.logger = logging.getLogger(__name__)
        self.alpha = alpha
        self.beta = beta
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        raise NotImplementedError("Subclasses must implement forward method")


class Swish(EngineeringActivation):
    """
    Swish activation function: x * sigmoid(x)
    """
    
    def __init__(self, alpha: float = 1.0, beta: float = 0.0):
        """
        Initialize Swish activation.
        
        Args:
            alpha: Scaling parameter
            beta: Shift parameter
        """
        super(Swish, self).__init__(alpha, beta)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return x * torch.sigmoid(self.alpha * x + self.beta)


class Mish(EngineeringActivation):
    """
    Mish activation function: x * tanh(softplus(x))
    """
    
    def __init__(self, alpha: float = 1.0, beta: float = 0.0):
        """
        Initialize Mish activation.
        
        Args:
            alpha: Scaling parameter
            beta: Shift parameter
        """
        super(Mish, self).__init__(alpha, beta)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return x * torch.tanh(F.softplus(self.alpha * x + self.beta))


class GELU(EngineeringActivation):
    """
    Gaussian Error Linear Unit activation function.
    """
    
    def __init__(self, alpha: float = 1.0, beta: float = 0.0):
        """
        Initialize GELU activation.
        
        Args:
            alpha: Scaling parameter
            beta: Shift parameter
        """
        super(GELU, self).__init__(alpha, beta)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (self.alpha * x + self.beta + 0.044715 * torch.pow(self.alpha * x + self.beta, 3))))


class ELU(EngineeringActivation):
    """
    Exponential Linear Unit activation function.
    """
    
    def __init__(self, alpha: float = 1.0, beta: float = 0.0):
        """
        Initialize ELU activation.
        
        Args:
            alpha: Scaling parameter
            beta: Shift parameter
        """
        super(ELU, self).__init__(alpha, beta)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return torch.where(x > 0, self.alpha * x + self.beta, self.alpha * (torch.exp(x) - 1) + self.beta)


class SELU(EngineeringActivation):
    """
    Scaled Exponential Linear Unit activation function.
    """
    
    def __init__(self, alpha: float = 1.6732632423543772848170429916717, beta: float = 1.0507009873554804934193349852946):
        """
        Initialize SELU activation.
        
        Args:
            alpha: Scaling parameter
            beta: Shift parameter
        """
        super(SELU, self).__init__(alpha, beta)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.beta * torch.where(x > 0, x, self.alpha * (torch.exp(x) - 1))


class LeakyReLU(EngineeringActivation):
    """
    Leaky ReLU activation function.
    """
    
    def __init__(self, alpha: float = 0.01, beta: float = 0.0):
        """
        Initialize LeakyReLU activation.
        
        Args:
            alpha: Negative slope
            beta: Shift parameter
        """
        super(LeakyReLU, self).__init__(alpha, beta)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return torch.where(x > 0, x + self.beta, self.alpha * x + self.beta)


class PReLU(EngineeringActivation):
    """
    Parametric ReLU activation function.
    """
    
    def __init__(self, alpha: float = 0.25, beta: float = 0.0):
        """
        Initialize PReLU activation.
        
        Args:
            alpha: Learnable parameter
            beta: Shift parameter
        """
        super(PReLU, self).__init__(alpha, beta)
        self.alpha = nn.Parameter(torch.tensor(alpha))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return torch.where(x > 0, x + self.beta, self.alpha * x + self.beta)


class RReLU(EngineeringActivation):
    """
    Randomized ReLU activation function.
    """
    
    def __init__(self, alpha: float = 0.125, beta: float = 0.0, training: bool = True):
        """
        Initialize RReLU activation.
        
        Args:
            alpha: Upper bound for random parameter
            beta: Shift parameter
            training: Whether in training mode
        """
        super(RReLU, self).__init__(alpha, beta)
        self.training = training
        self.alpha = alpha
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if self.training:
            random_alpha = torch.rand_like(x) * self.alpha
            return torch.where(x > 0, x + self.beta, random_alpha * x + self.beta)
        else:
            return torch.where(x > 0, x + self.beta, (self.alpha / 2) * x + self.beta)


class HardSwish(EngineeringActivation):
    """
    Hard Swish activation function.
    """
    
    def __init__(self, alpha: float = 1.0, beta: float = 0.0):
        """
        Initialize HardSwish activation.
        
        Args:
            alpha: Scaling parameter
            beta: Shift parameter
        """
        super(HardSwish, self).__init__(alpha, beta)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return x * torch.clamp((self.alpha * x + self.beta + 3) / 6, 0, 1)


class HardSigmoid(EngineeringActivation):
    """
    Hard Sigmoid activation function.
    """
    
    def __init__(self, alpha: float = 1.0, beta: float = 0.0):
        """
        Initialize HardSigmoid activation.
        
        Args:
            alpha: Scaling parameter
            beta: Shift parameter
        """
        super(HardSigmoid, self).__init__(alpha, beta)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return torch.clamp((self.alpha * x + self.beta + 3) / 6, 0, 1)


class HardTanh(EngineeringActivation):
    """
    Hard Tanh activation function.
    """
    
    def __init__(self, alpha: float = 1.0, beta: float = 0.0):
        """
        Initialize HardTanh activation.
        
        Args:
            alpha: Scaling parameter
            beta: Shift parameter
        """
        super(HardTanh, self).__init__(alpha, beta)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return torch.clamp(self.alpha * x + self.beta, -1, 1)


class Softplus(EngineeringActivation):
    """
    Softplus activation function.
    """
    
    def __init__(self, alpha: float = 1.0, beta: float = 0.0):
        """
        Initialize Softplus activation.
        
        Args:
            alpha: Scaling parameter
            beta: Shift parameter
        """
        super(Softplus, self).__init__(alpha, beta)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return F.softplus(self.alpha * x + self.beta)


class Softsign(EngineeringActivation):
    """
    Softsign activation function.
    """
    
    def __init__(self, alpha: float = 1.0, beta: float = 0.0):
        """
        Initialize Softsign activation.
        
        Args:
            alpha: Scaling parameter
            beta: Shift parameter
        """
        super(Softsign, self).__init__(alpha, beta)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return (self.alpha * x + self.beta) / (1 + torch.abs(self.alpha * x + self.beta))


class TanhShrink(EngineeringActivation):
    """
    TanhShrink activation function.
    """
    
    def __init__(self, alpha: float = 1.0, beta: float = 0.0):
        """
        Initialize TanhShrink activation.
        
        Args:
            alpha: Scaling parameter
            beta: Shift parameter
        """
        super(TanhShrink, self).__init__(alpha, beta)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return (self.alpha * x + self.beta) - torch.tanh(self.alpha * x + self.beta)


class Softmin(EngineeringActivation):
    """
    Softmin activation function.
    """
    
    def __init__(self, alpha: float = 1.0, beta: float = 0.0):
        """
        Initialize Softmin activation.
        
        Args:
            alpha: Scaling parameter
            beta: Shift parameter
        """
        super(Softmin, self).__init__(alpha, beta)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return F.softmin(self.alpha * x + self.beta, dim=-1)


class Softmax(EngineeringActivation):
    """
    Softmax activation function.
    """
    
    def __init__(self, alpha: float = 1.0, beta: float = 0.0):
        """
        Initialize Softmax activation.
        
        Args:
            alpha: Scaling parameter
            beta: Shift parameter
        """
        super(Softmax, self).__init__(alpha, beta)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return F.softmax(self.alpha * x + self.beta, dim=-1)


class LogSoftmax(EngineeringActivation):
    """
    Log Softmax activation function.
    """
    
    def __init__(self, alpha: float = 1.0, beta: float = 0.0):
        """
        Initialize LogSoftmax activation.
        
        Args:
            alpha: Scaling parameter
            beta: Shift parameter
        """
        super(LogSoftmax, self).__init__(alpha, beta)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return F.log_softmax(self.alpha * x + self.beta, dim=-1)


class LogSigmoid(EngineeringActivation):
    """
    Log Sigmoid activation function.
    """
    
    def __init__(self, alpha: float = 1.0, beta: float = 0.0):
        """
        Initialize LogSigmoid activation.
        
        Args:
            alpha: Scaling parameter
            beta: Shift parameter
        """
        super(LogSigmoid, self).__init__(alpha, beta)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return F.logsigmoid(self.alpha * x + self.beta)


class Hardshrink(EngineeringActivation):
    """
    Hardshrink activation function.
    """
    
    def __init__(self, alpha: float = 0.5, beta: float = 0.0):
        """
        Initialize Hardshrink activation.
        
        Args:
            alpha: Threshold parameter
            beta: Shift parameter
        """
        super(Hardshrink, self).__init__(alpha, beta)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return torch.where(torch.abs(x + self.beta) > self.alpha, x + self.beta, torch.zeros_like(x))


class Softshrink(EngineeringActivation):
    """
    Softshrink activation function.
    """
    
    def __init__(self, alpha: float = 0.5, beta: float = 0.0):
        """
        Initialize Softshrink activation.
        
        Args:
            alpha: Threshold parameter
            beta: Shift parameter
        """
        super(Softshrink, self).__init__(alpha, beta)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return torch.sign(x + self.beta) * torch.clamp(torch.abs(x + self.beta) - self.alpha, min=0)


class Tanh(EngineeringActivation):
    """
    Tanh activation function.
    """
    
    def __init__(self, alpha: float = 1.0, beta: float = 0.0):
        """
        Initialize Tanh activation.
        
        Args:
            alpha: Scaling parameter
            beta: Shift parameter
        """
        super(Tanh, self).__init__(alpha, beta)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return torch.tanh(self.alpha * x + self.beta)


class Sigmoid(EngineeringActivation):
    """
    Sigmoid activation function.
    """
    
    def __init__(self, alpha: float = 1.0, beta: float = 0.0):
        """
        Initialize Sigmoid activation.
        
        Args:
            alpha: Scaling parameter
            beta: Shift parameter
        """
        super(Sigmoid, self).__init__(alpha, beta)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return torch.sigmoid(self.alpha * x + self.beta)


class ReLU(EngineeringActivation):
    """
    ReLU activation function.
    """
    
    def __init__(self, alpha: float = 1.0, beta: float = 0.0):
        """
        Initialize ReLU activation.
        
        Args:
            alpha: Scaling parameter
            beta: Shift parameter
        """
        super(ReLU, self).__init__(alpha, beta)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return torch.relu(self.alpha * x + self.beta)
