"""
Custom neural network layers for engineering applications.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Union, Tuple
import math


class EngineeringLayer(nn.Module):
    """
    Base engineering layer with common engineering features.
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 activation: str = "relu", dropout_rate: float = 0.1,
                 use_batch_norm: bool = True):
        """
        Initialize engineering layer.
        
        Args:
            in_features: Input features
            out_features: Output features
            bias: Whether to use bias
            activation: Activation function
            dropout_rate: Dropout rate
            use_batch_norm: Whether to use batch normalization
        """
        super(EngineeringLayer, self).__init__()
        
        self.logger = logging.getLogger(__name__)
        self.in_features = in_features
        self.out_features = out_features
        
        # Linear layer
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        # Batch normalization
        if use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(out_features)
        else:
            self.batch_norm = None
        
        # Activation function
        self.activation = self._get_activation(activation)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
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
        elif activation.lower() == "leaky_relu":
            return nn.LeakyReLU()
        else:
            return nn.ReLU()
    
    def _initialize_weights(self):
        """Initialize layer weights."""
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.linear(x)
        
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        
        x = self.activation(x)
        x = self.dropout(x)
        
        return x


class AdaptiveLayer(nn.Module):
    """
    Adaptive layer that adjusts its behavior based on input characteristics.
    """
    
    def __init__(self, in_features: int, out_features: int, adaptation_dim: int = 32,
                 bias: bool = True, activation: str = "relu", dropout_rate: float = 0.1):
        """
        Initialize adaptive layer.
        
        Args:
            in_features: Input features
            out_features: Output features
            adaptation_dim: Adaptation dimension
            bias: Whether to use bias
            activation: Activation function
            dropout_rate: Dropout rate
        """
        super(AdaptiveLayer, self).__init__()
        
        self.logger = logging.getLogger(__name__)
        self.in_features = in_features
        self.out_features = out_features
        self.adaptation_dim = adaptation_dim
        
        # Main linear layer
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        # Adaptation network
        self.adaptation_net = nn.Sequential(
            nn.Linear(in_features, adaptation_dim),
            nn.ReLU(),
            nn.Linear(adaptation_dim, out_features),
            nn.Sigmoid()
        )
        
        # Activation function
        self.activation = self._get_activation(activation)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
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
        """Initialize layer weights."""
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)
        
        for module in self.adaptation_net.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Main transformation
        main_output = self.linear(x)
        
        # Adaptation
        adaptation = self.adaptation_net(x)
        
        # Apply adaptation
        adapted_output = main_output * adaptation
        
        # Activation and dropout
        output = self.activation(adapted_output)
        output = self.dropout(output)
        
        return output


class ResidualLayer(nn.Module):
    """
    Residual layer with skip connection.
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 activation: str = "relu", dropout_rate: float = 0.1,
                 use_batch_norm: bool = True):
        """
        Initialize residual layer.
        
        Args:
            in_features: Input features
            out_features: Output features
            bias: Whether to use bias
            activation: Activation function
            dropout_rate: Dropout rate
            use_batch_norm: Whether to use batch normalization
        """
        super(ResidualLayer, self).__init__()
        
        self.logger = logging.getLogger(__name__)
        self.in_features = in_features
        self.out_features = out_features
        
        # Main transformation
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        # Batch normalization
        if use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(out_features)
        else:
            self.batch_norm = None
        
        # Activation function
        self.activation = self._get_activation(activation)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Skip connection
        if in_features != out_features:
            self.skip_connection = nn.Linear(in_features, out_features, bias=False)
        else:
            self.skip_connection = nn.Identity()
        
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
        """Initialize layer weights."""
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)
        
        if isinstance(self.skip_connection, nn.Linear):
            nn.init.xavier_uniform_(self.skip_connection.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Main transformation
        main_output = self.linear(x)
        
        if self.batch_norm is not None:
            main_output = self.batch_norm(main_output)
        
        main_output = self.activation(main_output)
        main_output = self.dropout(main_output)
        
        # Skip connection
        skip_output = self.skip_connection(x)
        
        # Residual connection
        output = main_output + skip_output
        
        return output


class AttentionLayer(nn.Module):
    """
    Attention layer for engineering applications.
    """
    
    def __init__(self, in_features: int, out_features: int, num_heads: int = 4,
                 bias: bool = True, activation: str = "relu", dropout_rate: float = 0.1):
        """
        Initialize attention layer.
        
        Args:
            in_features: Input features
            out_features: Output features
            num_heads: Number of attention heads
            bias: Whether to use bias
            activation: Activation function
            dropout_rate: Dropout rate
        """
        super(AttentionLayer, self).__init__()
        
        self.logger = logging.getLogger(__name__)
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=in_features,
            num_heads=num_heads,
            dropout=dropout_rate,
            bias=bias,
            batch_first=True
        )
        
        # Output projection
        self.output_projection = nn.Linear(in_features, out_features, bias=bias)
        
        # Activation function
        self.activation = self._get_activation(activation)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
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
        """Initialize layer weights."""
        nn.init.xavier_uniform_(self.output_projection.weight)
        if self.output_projection.bias is not None:
            nn.init.zeros_(self.output_projection.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Add sequence dimension for attention
        x = x.unsqueeze(1)  # [batch_size, 1, in_features]
        
        # Multi-head attention
        attn_output, _ = self.attention(x, x, x)
        
        # Remove sequence dimension
        attn_output = attn_output.squeeze(1)  # [batch_size, in_features]
        
        # Output projection
        output = self.output_projection(attn_output)
        
        # Activation and dropout
        output = self.activation(output)
        output = self.dropout(output)
        
        return output


class ConvolutionalLayer(nn.Module):
    """
    Convolutional layer for engineering applications.
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1, bias: bool = True,
                 activation: str = "relu", dropout_rate: float = 0.1,
                 use_batch_norm: bool = True):
        """
        Initialize convolutional layer.
        
        Args:
            in_channels: Input channels
            out_channels: Output channels
            kernel_size: Kernel size
            stride: Stride
            padding: Padding
            bias: Whether to use bias
            activation: Activation function
            dropout_rate: Dropout rate
            use_batch_norm: Whether to use batch normalization
        """
        super(ConvolutionalLayer, self).__init__()
        
        self.logger = logging.getLogger(__name__)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Convolutional layer
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )
        
        # Batch normalization
        if use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(out_channels)
        else:
            self.batch_norm = None
        
        # Activation function
        self.activation = self._get_activation(activation)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
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
        """Initialize layer weights."""
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.conv(x)
        
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        
        x = self.activation(x)
        x = self.dropout(x)
        
        return x


class RecurrentLayer(nn.Module):
    """
    Recurrent layer for engineering applications.
    """
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1,
                 rnn_type: str = "lstm", bias: bool = True, dropout_rate: float = 0.1,
                 bidirectional: bool = False):
        """
        Initialize recurrent layer.
        
        Args:
            input_size: Input size
            hidden_size: Hidden size
            num_layers: Number of layers
            rnn_type: Type of RNN (lstm, gru, rnn)
            bias: Whether to use bias
            dropout_rate: Dropout rate
            bidirectional: Whether to use bidirectional RNN
        """
        super(RecurrentLayer, self).__init__()
        
        self.logger = logging.getLogger(__name__)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type.lower()
        self.bidirectional = bidirectional
        
        # RNN layer
        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bias=bias,
                dropout=dropout_rate if num_layers > 1 else 0,
                bidirectional=bidirectional,
                batch_first=True
            )
        elif self.rnn_type == "gru":
            self.rnn = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bias=bias,
                dropout=dropout_rate if num_layers > 1 else 0,
                bidirectional=bidirectional,
                batch_first=True
            )
        elif self.rnn_type == "rnn":
            self.rnn = nn.RNN(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bias=bias,
                dropout=dropout_rate if num_layers > 1 else 0,
                bidirectional=bidirectional,
                batch_first=True
            )
        else:
            raise ValueError(f"Unsupported RNN type: {rnn_type}")
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize layer weights."""
        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Add sequence dimension if needed
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # [batch_size, 1, input_size]
        
        # RNN forward pass
        rnn_output, _ = self.rnn(x)
        
        # Remove sequence dimension
        rnn_output = rnn_output.squeeze(1)  # [batch_size, hidden_size]
        
        # Dropout
        output = self.dropout(rnn_output)
        
        return output


class TransformerLayer(nn.Module):
    """
    Transformer layer for engineering applications.
    """
    
    def __init__(self, d_model: int, nhead: int = 8, dim_feedforward: int = 2048,
                 dropout_rate: float = 0.1, activation: str = "relu", bias: bool = True):
        """
        Initialize transformer layer.
        
        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            dim_feedforward: Feedforward dimension
            dropout_rate: Dropout rate
            activation: Activation function
            bias: Whether to use bias
        """
        super(TransformerLayer, self).__init__()
        
        self.logger = logging.getLogger(__name__)
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        
        # Transformer encoder layer
        self.transformer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout_rate,
            activation=activation,
            bias=bias,
            batch_first=True
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize layer weights."""
        for module in self.transformer.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Add sequence dimension if needed
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # [batch_size, 1, d_model]
        
        # Transformer forward pass
        transformer_output = self.transformer(x)
        
        # Remove sequence dimension
        output = transformer_output.squeeze(1)  # [batch_size, d_model]
        
        return output
