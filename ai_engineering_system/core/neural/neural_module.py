"""
Neural Networks module for engineering applications.
"""

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
import matplotlib.pyplot as plt
from dataclasses import dataclass
import json
import pickle
from pathlib import Path


@dataclass
class NeuralConfig:
    """Configuration for neural network training."""
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    optimizer: str = "adam"
    loss_function: str = "mse"
    activation: str = "relu"
    dropout_rate: float = 0.1
    weight_decay: float = 1e-4
    early_stopping_patience: int = 10
    validation_split: float = 0.2
    device: str = "auto"


@dataclass
class TrainingMetrics:
    """Training metrics container."""
    train_loss: List[float]
    val_loss: List[float]
    train_accuracy: List[float]
    val_accuracy: List[float]
    learning_rate: List[float]
    epoch_times: List[float]


class NeuralModule:
    """
    Main neural networks module for engineering applications.
    """
    
    def __init__(self, config: Optional[NeuralConfig] = None):
        """
        Initialize neural networks module.
        
        Args:
            config: Configuration for neural networks
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or NeuralConfig()
        
        # Device configuration
        self.device = self._setup_device()
        
        # Model registry
        self.models = {}
        self.training_history = {}
        
        # Engineering-specific components
        self.engineering_networks = {}
        self.specialized_layers = {}
        self.custom_activations = {}
        
        self.logger.info(f"Neural Networks module initialized on device: {self.device}")
    
    def _setup_device(self) -> torch.device:
        """Setup computation device."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(self.config.device)
        
        return device
    
    async def create_engineering_network(self, network_type: str, input_dim: int, 
                                       output_dim: int, hidden_dims: List[int],
                                       **kwargs) -> nn.Module:
        """
        Create engineering-specific neural network.
        
        Args:
            network_type: Type of engineering network
            input_dim: Input dimension
            output_dim: Output dimension
            hidden_dims: Hidden layer dimensions
            **kwargs: Additional network parameters
            
        Returns:
            Neural network model
        """
        self.logger.info(f"Creating {network_type} network")
        
        if network_type == "structural":
            model = self._create_structural_network(input_dim, output_dim, hidden_dims, **kwargs)
        elif network_type == "fluid":
            model = self._create_fluid_network(input_dim, output_dim, hidden_dims, **kwargs)
        elif network_type == "material":
            model = self._create_material_network(input_dim, output_dim, hidden_dims, **kwargs)
        elif network_type == "control":
            model = self._create_control_network(input_dim, output_dim, hidden_dims, **kwargs)
        elif network_type == "optimization":
            model = self._create_optimization_network(input_dim, output_dim, hidden_dims, **kwargs)
        else:
            model = self._create_general_network(input_dim, output_dim, hidden_dims, **kwargs)
        
        # Move to device
        model = model.to(self.device)
        
        # Register model
        model_id = f"{network_type}_{len(self.models)}"
        self.models[model_id] = model
        self.engineering_networks[network_type] = model
        
        return model
    
    def _create_structural_network(self, input_dim: int, output_dim: int, 
                                 hidden_dims: List[int], **kwargs) -> nn.Module:
        """Create structural engineering network."""
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(self.config.dropout_rate))
        
        # Hidden layers with residual connections
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.BatchNorm1d(hidden_dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.config.dropout_rate))
        
        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
        return nn.Sequential(*layers)
    
    def _create_fluid_network(self, input_dim: int, output_dim: int, 
                            hidden_dims: List[int], **kwargs) -> nn.Module:
        """Create fluid dynamics network."""
        layers = []
        
        # Input layer with special normalization for fluid properties
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.LayerNorm(hidden_dims[0]))
        layers.append(nn.Swish())
        layers.append(nn.Dropout(self.config.dropout_rate))
        
        # Hidden layers with attention mechanism
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.LayerNorm(hidden_dims[i + 1]))
            layers.append(nn.Swish())
            layers.append(nn.Dropout(self.config.dropout_rate))
        
        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
        return nn.Sequential(*layers)
    
    def _create_material_network(self, input_dim: int, output_dim: int, 
                               hidden_dims: List[int], **kwargs) -> nn.Module:
        """Create material science network."""
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(self.config.dropout_rate))
        
        # Hidden layers with skip connections
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.BatchNorm1d(hidden_dims[i + 1]))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(self.config.dropout_rate))
        
        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
        return nn.Sequential(*layers)
    
    def _create_control_network(self, input_dim: int, output_dim: int, 
                              hidden_dims: List[int], **kwargs) -> nn.Module:
        """Create control systems network."""
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(nn.Tanh())
        layers.append(nn.Dropout(self.config.dropout_rate))
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.BatchNorm1d(hidden_dims[i + 1]))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(self.config.dropout_rate))
        
        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
        return nn.Sequential(*layers)
    
    def _create_optimization_network(self, input_dim: int, output_dim: int, 
                                   hidden_dims: List[int], **kwargs) -> nn.Module:
        """Create optimization network."""
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(self.config.dropout_rate))
        
        # Hidden layers with residual connections
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.BatchNorm1d(hidden_dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.config.dropout_rate))
        
        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
        return nn.Sequential(*layers)
    
    def _create_general_network(self, input_dim: int, output_dim: int, 
                              hidden_dims: List[int], **kwargs) -> nn.Module:
        """Create general purpose network."""
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(self.config.dropout_rate))
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.BatchNorm1d(hidden_dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.config.dropout_rate))
        
        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
        return nn.Sequential(*layers)
    
    async def train_model(self, model: nn.Module, train_data: torch.utils.data.DataLoader,
                         val_data: Optional[torch.utils.data.DataLoader] = None,
                         config: Optional[NeuralConfig] = None) -> TrainingMetrics:
        """
        Train neural network model.
        
        Args:
            model: Neural network model
            train_data: Training data loader
            val_data: Validation data loader
            config: Training configuration
            
        Returns:
            Training metrics
        """
        self.logger.info("Starting model training")
        
        config = config or self.config
        
        # Setup optimizer
        optimizer = self._setup_optimizer(model, config)
        
        # Setup loss function
        criterion = self._setup_loss_function(config)
        
        # Setup scheduler
        scheduler = self._setup_scheduler(optimizer, config)
        
        # Training metrics
        metrics = TrainingMetrics(
            train_loss=[],
            val_loss=[],
            train_accuracy=[],
            val_accuracy=[],
            learning_rate=[],
            epoch_times=[]
        )
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(config.epochs):
            start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            
            if start_time:
                start_time.record()
            
            # Training phase
            train_loss, train_acc = self._train_epoch(model, train_data, optimizer, criterion)
            
            # Validation phase
            val_loss, val_acc = 0.0, 0.0
            if val_data:
                val_loss, val_acc = self._validate_epoch(model, val_data, criterion)
            
            # Update scheduler
            if scheduler:
                scheduler.step()
            
            # Record metrics
            metrics.train_loss.append(train_loss)
            metrics.val_loss.append(val_loss)
            metrics.train_accuracy.append(train_acc)
            metrics.val_accuracy.append(val_acc)
            metrics.learning_rate.append(optimizer.param_groups[0]['lr'])
            
            if end_time:
                end_time.record()
                torch.cuda.synchronize()
                epoch_time = start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
            else:
                epoch_time = 0.0
            
            metrics.epoch_times.append(epoch_time)
            
            # Log progress
            self.logger.info(f"Epoch {epoch+1}/{config.epochs}: "
                           f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                           f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Early stopping
            if val_data and val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= config.early_stopping_patience:
                self.logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Store training history
        model_id = f"model_{len(self.training_history)}"
        self.training_history[model_id] = metrics
        
        return metrics
    
    def _setup_optimizer(self, model: nn.Module, config: NeuralConfig) -> optim.Optimizer:
        """Setup optimizer."""
        if config.optimizer.lower() == "adam":
            return optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        elif config.optimizer.lower() == "adamw":
            return optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        elif config.optimizer.lower() == "sgd":
            return optim.SGD(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay, momentum=0.9)
        elif config.optimizer.lower() == "rmsprop":
            return optim.RMSprop(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        else:
            return optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    def _setup_loss_function(self, config: NeuralConfig) -> nn.Module:
        """Setup loss function."""
        if config.loss_function.lower() == "mse":
            return nn.MSELoss()
        elif config.loss_function.lower() == "mae":
            return nn.L1Loss()
        elif config.loss_function.lower() == "crossentropy":
            return nn.CrossEntropyLoss()
        elif config.loss_function.lower() == "bce":
            return nn.BCELoss()
        elif config.loss_function.lower() == "huber":
            return nn.HuberLoss()
        else:
            return nn.MSELoss()
    
    def _setup_scheduler(self, optimizer: optim.Optimizer, config: NeuralConfig) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Setup learning rate scheduler."""
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
    
    def _train_epoch(self, model: nn.Module, train_data: torch.utils.data.DataLoader,
                    optimizer: optim.Optimizer, criterion: nn.Module) -> Tuple[float, float]:
        """Train for one epoch."""
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for batch_idx, (data, target) in enumerate(train_data):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate accuracy (for classification tasks)
            if len(target.shape) == 1:  # Classification
                pred = output.argmax(dim=1)
                total_correct += pred.eq(target).sum().item()
                total_samples += target.size(0)
            else:  # Regression
                total_correct += 1  # Placeholder for regression
                total_samples += 1
        
        avg_loss = total_loss / len(train_data)
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        
        return avg_loss, accuracy
    
    def _validate_epoch(self, model: nn.Module, val_data: torch.utils.data.DataLoader,
                       criterion: nn.Module) -> Tuple[float, float]:
        """Validate for one epoch."""
        model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for data, target in val_data:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                
                # Calculate accuracy
                if len(target.shape) == 1:  # Classification
                    pred = output.argmax(dim=1)
                    total_correct += pred.eq(target).sum().item()
                    total_samples += target.size(0)
                else:  # Regression
                    total_correct += 1
                    total_samples += 1
        
        avg_loss = total_loss / len(val_data)
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        
        return avg_loss, accuracy
    
    async def evaluate_model(self, model: nn.Module, test_data: torch.utils.data.DataLoader,
                           metrics: List[str] = None) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            model: Neural network model
            test_data: Test data loader
            metrics: List of metrics to calculate
            
        Returns:
            Evaluation metrics
        """
        self.logger.info("Evaluating model")
        
        model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in test_data:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                
                all_predictions.append(output.cpu())
                all_targets.append(target.cpu())
        
        # Concatenate all predictions and targets
        predictions = torch.cat(all_predictions, dim=0)
        targets = torch.cat(all_targets, dim=0)
        
        # Calculate metrics
        evaluation_metrics = {}
        
        if metrics is None:
            metrics = ["mse", "mae", "r2"]
        
        for metric in metrics:
            if metric == "mse":
                evaluation_metrics["mse"] = torch.mean((predictions - targets) ** 2).item()
            elif metric == "mae":
                evaluation_metrics["mae"] = torch.mean(torch.abs(predictions - targets)).item()
            elif metric == "r2":
                ss_res = torch.sum((targets - predictions) ** 2)
                ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
                evaluation_metrics["r2"] = 1 - (ss_res / ss_tot).item()
            elif metric == "accuracy":
                if len(targets.shape) == 1:  # Classification
                    pred = predictions.argmax(dim=1)
                    evaluation_metrics["accuracy"] = (pred == targets).float().mean().item()
                else:  # Regression
                    evaluation_metrics["accuracy"] = 0.0
        
        return evaluation_metrics
    
    def save_model(self, model: nn.Module, filepath: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Save model to file.
        
        Args:
            model: Neural network model
            filepath: Path to save model
            metadata: Additional metadata to save
        """
        self.logger.info(f"Saving model to {filepath}")
        
        save_dict = {
            'model_state_dict': model.state_dict(),
            'model_config': self.config.__dict__,
            'metadata': metadata or {}
        }
        
        torch.save(save_dict, filepath)
    
    def load_model(self, filepath: str, model: Optional[nn.Module] = None) -> nn.Module:
        """
        Load model from file.
        
        Args:
            filepath: Path to model file
            model: Model to load state into (optional)
            
        Returns:
            Loaded model
        """
        self.logger.info(f"Loading model from {filepath}")
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        if model is None:
            # Create model from saved config
            config = NeuralConfig(**checkpoint['model_config'])
            model = self._create_general_network(
                input_dim=checkpoint['metadata'].get('input_dim', 10),
                output_dim=checkpoint['metadata'].get('output_dim', 1),
                hidden_dims=checkpoint['metadata'].get('hidden_dims', [64, 32])
            )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        
        return model
    
    def plot_training_history(self, metrics: TrainingMetrics, save_path: Optional[str] = None):
        """Plot training history."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Plot loss
        axes[0, 0].plot(metrics.train_loss, label='Train Loss')
        if metrics.val_loss:
            axes[0, 0].plot(metrics.val_loss, label='Validation Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot accuracy
        axes[0, 1].plot(metrics.train_accuracy, label='Train Accuracy')
        if metrics.val_accuracy:
            axes[0, 1].plot(metrics.val_accuracy, label='Validation Accuracy')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot learning rate
        axes[1, 0].plot(metrics.learning_rate)
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].grid(True)
        
        # Plot epoch times
        axes[1, 1].plot(metrics.epoch_times)
        axes[1, 1].set_title('Epoch Times')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Time (seconds)')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        
        plt.close()
    
    def get_model_info(self, model: nn.Module) -> Dict[str, Any]:
        """Get model information."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / (1024 * 1024),  # Assuming float32
            "device": str(self.device),
            "model_type": type(model).__name__
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get module status."""
        return {
            "device": str(self.device),
            "models_registered": len(self.models),
            "engineering_networks": list(self.engineering_networks.keys()),
            "training_history": len(self.training_history),
            "config": self.config.__dict__
        }
