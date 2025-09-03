"""
Custom neural network architectures for engineering applications.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Union, Tuple
import math


class EngineeringNet(nn.Module):
    """
    Base engineering neural network with common engineering features.
    """
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int],
                 activation: str = "relu", dropout_rate: float = 0.1,
                 use_batch_norm: bool = True, use_residual: bool = False):
        """
        Initialize engineering network.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            hidden_dims: Hidden layer dimensions
            activation: Activation function
            dropout_rate: Dropout rate
            use_batch_norm: Whether to use batch normalization
            use_residual: Whether to use residual connections
        """
        super(EngineeringNet, self).__init__()
        
        self.logger = logging.getLogger(__name__)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.use_residual = use_residual
        
        # Activation function
        self.activation = self._get_activation(activation)
        
        # Build network
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
        if use_batch_norm:
            self.batch_norms.append(nn.BatchNorm1d(hidden_dims[0]))
        self.dropouts.append(nn.Dropout(dropout_rate))
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dims[i + 1]))
            self.dropouts.append(nn.Dropout(dropout_rate))
        
        # Output layer
        self.layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
        # Residual connections
        if use_residual:
            self.residual_layers = nn.ModuleList()
            for i in range(len(hidden_dims) - 1):
                if hidden_dims[i] != hidden_dims[i + 1]:
                    self.residual_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
                else:
                    self.residual_layers.append(nn.Identity())
        
        # Initialize weights
        self._initialize_weights()
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function."""
        if activation.lower() == "relu":
            return nn.ReLU()
        elif activation.lower() == "gelu":
            return nn.GELU()
        elif activation.lower() == "swish":
            return nn.SiLU()
        elif activation.lower() == "tanh":
            return nn.Tanh()
        elif activation.lower() == "sigmoid":
            return nn.Sigmoid()
        else:
            return nn.ReLU()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Input layer
        x = self.layers[0](x)
        if len(self.batch_norms) > 0:
            x = self.batch_norms[0](x)
        x = self.activation(x)
        x = self.dropouts[0](x)
        
        # Hidden layers
        for i in range(1, len(self.layers) - 1):
            residual = x
            
            x = self.layers[i](x)
            if len(self.batch_norms) > i:
                x = self.batch_norms[i](x)
            x = self.activation(x)
            x = self.dropouts[i](x)
            
            # Residual connection
            if self.use_residual and i > 1:
                if len(self.residual_layers) > i - 2:
                    residual = self.residual_layers[i - 2](residual)
                    x = x + residual
        
        # Output layer
        x = self.layers[-1](x)
        
        return x


class StructuralNet(nn.Module):
    """
    Neural network for structural engineering applications.
    """
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int] = [128, 64, 32],
                 num_attention_heads: int = 4, dropout_rate: float = 0.1):
        """
        Initialize structural network.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            hidden_dims: Hidden layer dimensions
            num_attention_heads: Number of attention heads
            dropout_rate: Dropout rate
        """
        super(StructuralNet, self).__init__()
        
        self.logger = logging.getLogger(__name__)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dims[0])
        
        # Multi-head attention for structural relationships
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dims[0],
            num_heads=num_attention_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Structural feature extractor
        self.structural_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.structural_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            self.structural_layers.append(nn.BatchNorm1d(hidden_dims[i + 1]))
            self.structural_layers.append(nn.ReLU())
            self.structural_layers.append(nn.Dropout(dropout_rate))
        
        # Output layers for different structural properties
        self.stress_head = nn.Linear(hidden_dims[-1], output_dim // 3)
        self.strain_head = nn.Linear(hidden_dims[-1], output_dim // 3)
        self.displacement_head = nn.Linear(hidden_dims[-1], output_dim - 2 * (output_dim // 3))
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Input projection
        x = self.input_projection(x)
        
        # Add sequence dimension for attention
        x = x.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        # Multi-head attention
        attn_output, _ = self.attention(x, x, x)
        x = attn_output.squeeze(1)  # Remove sequence dimension
        
        # Structural feature extraction
        for layer in self.structural_layers:
            x = layer(x)
        
        # Multiple output heads
        stress = self.stress_head(x)
        strain = self.strain_head(x)
        displacement = self.displacement_head(x)
        
        # Concatenate outputs
        output = torch.cat([stress, strain, displacement], dim=1)
        
        return output


class FluidNet(nn.Module):
    """
    Neural network for fluid dynamics applications.
    """
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int] = [128, 64, 32],
                 num_convolutions: int = 3, dropout_rate: float = 0.1):
        """
        Initialize fluid network.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            hidden_dims: Hidden layer dimensions
            num_convolutions: Number of convolution layers
            dropout_rate: Dropout rate
        """
        super(FluidNet, self).__init__()
        
        self.logger = logging.getLogger(__name__)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dims[0])
        
        # Convolutional layers for spatial relationships
        self.conv_layers = nn.ModuleList()
        for i in range(num_convolutions):
            self.conv_layers.append(nn.Conv1d(1, 1, kernel_size=3, padding=1))
            self.conv_layers.append(nn.BatchNorm1d(1))
            self.conv_layers.append(nn.ReLU())
            self.conv_layers.append(nn.Dropout(dropout_rate))
        
        # Fluid dynamics feature extractor
        self.fluid_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.fluid_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            self.fluid_layers.append(nn.LayerNorm(hidden_dims[i + 1]))
            self.fluid_layers.append(nn.SiLU())  # Swish activation
            self.fluid_layers.append(nn.Dropout(dropout_rate))
        
        # Output layers for different fluid properties
        self.velocity_head = nn.Linear(hidden_dims[-1], output_dim // 3)
        self.pressure_head = nn.Linear(hidden_dims[-1], output_dim // 3)
        self.temperature_head = nn.Linear(hidden_dims[-1], output_dim - 2 * (output_dim // 3))
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Input projection
        x = self.input_projection(x)
        
        # Add channel dimension for convolution
        x = x.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        # Convolutional layers
        for layer in self.conv_layers:
            x = layer(x)
        
        # Remove channel dimension
        x = x.squeeze(1)
        
        # Fluid dynamics feature extraction
        for layer in self.fluid_layers:
            x = layer(x)
        
        # Multiple output heads
        velocity = self.velocity_head(x)
        pressure = self.pressure_head(x)
        temperature = self.temperature_head(x)
        
        # Concatenate outputs
        output = torch.cat([velocity, pressure, temperature], dim=1)
        
        return output


class MaterialNet(nn.Module):
    """
    Neural network for material science applications.
    """
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int] = [128, 64, 32],
                 num_attention_heads: int = 4, dropout_rate: float = 0.1):
        """
        Initialize material network.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            hidden_dims: Hidden layer dimensions
            num_attention_heads: Number of attention heads
            dropout_rate: Dropout rate
        """
        super(MaterialNet, self).__init__()
        
        self.logger = logging.getLogger(__name__)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dims[0])
        
        # Material property attention
        self.material_attention = nn.MultiheadAttention(
            embed_dim=hidden_dims[0],
            num_heads=num_attention_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Material feature extractor
        self.material_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.material_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            self.material_layers.append(nn.BatchNorm1d(hidden_dims[i + 1]))
            self.material_layers.append(nn.GELU())
            self.material_layers.append(nn.Dropout(dropout_rate))
        
        # Output layers for different material properties
        self.elastic_head = nn.Linear(hidden_dims[-1], output_dim // 3)
        self.plastic_head = nn.Linear(hidden_dims[-1], output_dim // 3)
        self.thermal_head = nn.Linear(hidden_dims[-1], output_dim - 2 * (output_dim // 3))
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Input projection
        x = self.input_projection(x)
        
        # Add sequence dimension for attention
        x = x.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        # Material property attention
        attn_output, _ = self.material_attention(x, x, x)
        x = attn_output.squeeze(1)  # Remove sequence dimension
        
        # Material feature extraction
        for layer in self.material_layers:
            x = layer(x)
        
        # Multiple output heads
        elastic = self.elastic_head(x)
        plastic = self.plastic_head(x)
        thermal = self.thermal_head(x)
        
        # Concatenate outputs
        output = torch.cat([elastic, plastic, thermal], dim=1)
        
        return output


class ControlNet(nn.Module):
    """
    Neural network for control systems applications.
    """
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int] = [128, 64, 32],
                 num_lstm_layers: int = 2, dropout_rate: float = 0.1):
        """
        Initialize control network.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            hidden_dims: Hidden layer dimensions
            num_lstm_layers: Number of LSTM layers
            dropout_rate: Dropout rate
        """
        super(ControlNet, self).__init__()
        
        self.logger = logging.getLogger(__name__)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dims[0])
        
        # LSTM for temporal relationships
        self.lstm = nn.LSTM(
            input_size=hidden_dims[0],
            hidden_size=hidden_dims[0],
            num_layers=num_lstm_layers,
            dropout=dropout_rate if num_lstm_layers > 1 else 0,
            batch_first=True
        )
        
        # Control feature extractor
        self.control_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.control_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            self.control_layers.append(nn.BatchNorm1d(hidden_dims[i + 1]))
            self.control_layers.append(nn.Tanh())
            self.control_layers.append(nn.Dropout(dropout_rate))
        
        # Output layers for different control signals
        self.position_head = nn.Linear(hidden_dims[-1], output_dim // 3)
        self.velocity_head = nn.Linear(hidden_dims[-1], output_dim // 3)
        self.acceleration_head = nn.Linear(hidden_dims[-1], output_dim - 2 * (output_dim // 3))
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Input projection
        x = self.input_projection(x)
        
        # Add sequence dimension for LSTM
        x = x.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        # LSTM processing
        lstm_output, _ = self.lstm(x)
        x = lstm_output.squeeze(1)  # Remove sequence dimension
        
        # Control feature extraction
        for layer in self.control_layers:
            x = layer(x)
        
        # Multiple output heads
        position = self.position_head(x)
        velocity = self.velocity_head(x)
        acceleration = self.acceleration_head(x)
        
        # Concatenate outputs
        output = torch.cat([position, velocity, acceleration], dim=1)
        
        return output


class OptimizationNet(nn.Module):
    """
    Neural network for optimization applications.
    """
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int] = [128, 64, 32],
                 num_attention_heads: int = 4, dropout_rate: float = 0.1):
        """
        Initialize optimization network.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            hidden_dims: Hidden layer dimensions
            num_attention_heads: Number of attention heads
            dropout_rate: Dropout rate
        """
        super(OptimizationNet, self).__init__()
        
        self.logger = logging.getLogger(__name__)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dims[0])
        
        # Optimization attention
        self.optimization_attention = nn.MultiheadAttention(
            embed_dim=hidden_dims[0],
            num_heads=num_attention_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Optimization feature extractor
        self.optimization_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.optimization_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            self.optimization_layers.append(nn.BatchNorm1d(hidden_dims[i + 1]))
            self.optimization_layers.append(nn.ReLU())
            self.optimization_layers.append(nn.Dropout(dropout_rate))
        
        # Output layers for different optimization objectives
        self.objective_head = nn.Linear(hidden_dims[-1], output_dim // 2)
        self.constraint_head = nn.Linear(hidden_dims[-1], output_dim - output_dim // 2)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Input projection
        x = self.input_projection(x)
        
        # Add sequence dimension for attention
        x = x.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        # Optimization attention
        attn_output, _ = self.optimization_attention(x, x, x)
        x = attn_output.squeeze(1)  # Remove sequence dimension
        
        # Optimization feature extraction
        for layer in self.optimization_layers:
            x = layer(x)
        
        # Multiple output heads
        objective = self.objective_head(x)
        constraint = self.constraint_head(x)
        
        # Concatenate outputs
        output = torch.cat([objective, constraint], dim=1)
        
        return output
