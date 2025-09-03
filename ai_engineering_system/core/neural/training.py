"""
Training utilities for neural networks in engineering applications.
"""

import logging
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
import matplotlib.pyplot as plt
from dataclasses import dataclass
import time
import json
import pickle
from pathlib import Path
import numpy as np


@dataclass
class TrainingConfig:
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
    gradient_clipping: bool = True
    clip_value: float = 1.0
    scheduler: str = "reduce_on_plateau"
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    warmup_epochs: int = 0
    warmup_factor: float = 0.1


@dataclass
class TrainingMetrics:
    """Training metrics container."""
    train_loss: List[float]
    val_loss: List[float]
    train_accuracy: List[float]
    val_accuracy: List[float]
    learning_rate: List[float]
    epoch_times: List[float]
    best_epoch: int
    best_val_loss: float


class NeuralTrainer:
    """
    Neural network trainer for engineering applications.
    """
    
    def __init__(self, config: Optional[TrainingConfig] = None):
        """
        Initialize neural trainer.
        
        Args:
            config: Training configuration
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or TrainingConfig()
        
        # Device configuration
        self.device = self._setup_device()
        
        # Training state
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.training_history = {}
        
        self.logger.info(f"Neural trainer initialized on device: {self.device}")
    
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
    
    def setup_training(self, model: nn.Module, train_data: torch.utils.data.DataLoader,
                      val_data: Optional[torch.utils.data.DataLoader] = None):
        """
        Setup training components.
        
        Args:
            model: Neural network model
            train_data: Training data loader
            val_data: Validation data loader
        """
        self.logger.info("Setting up training components")
        
        # Move model to device
        self.model = model.to(self.device)
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        
        # Setup loss function
        self.criterion = self._setup_loss_function()
        
        # Setup scheduler
        self.scheduler = self._setup_scheduler()
        
        # Store data loaders
        self.train_data = train_data
        self.val_data = val_data
    
    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer."""
        if self.config.optimizer.lower() == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == "adamw":
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                momentum=0.9
            )
        elif self.config.optimizer.lower() == "rmsprop":
            return optim.RMSprop(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        else:
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
    
    def _setup_loss_function(self) -> nn.Module:
        """Setup loss function."""
        if self.config.loss_function.lower() == "mse":
            return nn.MSELoss()
        elif self.config.loss_function.lower() == "mae":
            return nn.L1Loss()
        elif self.config.loss_function.lower() == "crossentropy":
            return nn.CrossEntropyLoss()
        elif self.config.loss_function.lower() == "bce":
            return nn.BCELoss()
        elif self.config.loss_function.lower() == "huber":
            return nn.HuberLoss()
        else:
            return nn.MSELoss()
    
    def _setup_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Setup learning rate scheduler."""
        if self.config.scheduler.lower() == "reduce_on_plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.config.scheduler_factor,
                patience=self.config.scheduler_patience,
                verbose=True
            )
        elif self.config.scheduler.lower() == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=20,
                gamma=0.5
            )
        elif self.config.scheduler.lower() == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs
            )
        else:
            return None
    
    async def train(self) -> TrainingMetrics:
        """
        Train the neural network.
        
        Returns:
            Training metrics
        """
        self.logger.info("Starting training")
        
        # Initialize metrics
        metrics = TrainingMetrics(
            train_loss=[],
            val_loss=[],
            train_accuracy=[],
            val_accuracy=[],
            learning_rate=[],
            epoch_times=[],
            best_epoch=0,
            best_val_loss=float('inf')
        )
        
        # Training loop
        patience_counter = 0
        
        for epoch in range(self.config.epochs):
            start_time = time.time()
            
            # Training phase
            train_loss, train_acc = self._train_epoch()
            
            # Validation phase
            val_loss, val_acc = 0.0, 0.0
            if self.val_data:
                val_loss, val_acc = self._validate_epoch()
            
            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Record metrics
            epoch_time = time.time() - start_time
            metrics.train_loss.append(train_loss)
            metrics.val_loss.append(val_loss)
            metrics.train_accuracy.append(train_acc)
            metrics.val_accuracy.append(val_acc)
            metrics.learning_rate.append(self.optimizer.param_groups[0]['lr'])
            metrics.epoch_times.append(epoch_time)
            
            # Log progress
            self.logger.info(f"Epoch {epoch+1}/{self.config.epochs}: "
                           f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                           f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Early stopping
            if self.val_data and val_loss < metrics.best_val_loss:
                metrics.best_val_loss = val_loss
                metrics.best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= self.config.early_stopping_patience:
                self.logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Store training history
        self.training_history['latest'] = metrics
        
        return metrics
    
    def _train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for batch_idx, (data, target) in enumerate(self.train_data):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            
            # Gradient clipping
            if self.config.gradient_clipping:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_value)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate accuracy
            if len(target.shape) == 1:  # Classification
                pred = output.argmax(dim=1)
                total_correct += pred.eq(target).sum().item()
                total_samples += target.size(0)
            else:  # Regression
                total_correct += 1
                total_samples += 1
        
        avg_loss = total_loss / len(self.train_data)
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        
        return avg_loss, accuracy
    
    def _validate_epoch(self) -> Tuple[float, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for data, target in self.val_data:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                
                # Calculate accuracy
                if len(target.shape) == 1:  # Classification
                    pred = output.argmax(dim=1)
                    total_correct += pred.eq(target).sum().item()
                    total_samples += target.size(0)
                else:  # Regression
                    total_correct += 1
                    total_samples += 1
        
        avg_loss = total_loss / len(self.val_data)
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        
        return avg_loss, accuracy
    
    async def evaluate(self, test_data: torch.utils.data.DataLoader,
                      metrics: List[str] = None) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            test_data: Test data loader
            metrics: List of metrics to calculate
            
        Returns:
            Evaluation metrics
        """
        self.logger.info("Evaluating model")
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in test_data:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
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
    
    def save_model(self, filepath: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Save model to file.
        
        Args:
            filepath: Path to save model
            metadata: Additional metadata to save
        """
        self.logger.info(f"Saving model to {filepath}")
        
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'training_config': self.config.__dict__,
            'training_history': self.training_history,
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
            config = TrainingConfig(**checkpoint['training_config'])
            # You would need to implement model creation based on config
            raise NotImplementedError("Model creation from config not implemented")
        
        self.model = model.to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'training_history' in checkpoint:
            self.training_history = checkpoint['training_history']
        
        return self.model
    
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
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        if self.model is None:
            return {}
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / (1024 * 1024),  # Assuming float32
            "device": str(self.device),
            "model_type": type(self.model).__name__
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get trainer status."""
        return {
            "device": str(self.device),
            "model_loaded": self.model is not None,
            "optimizer_loaded": self.optimizer is not None,
            "scheduler_loaded": self.scheduler is not None,
            "training_history": len(self.training_history),
            "config": self.config.__dict__
        }


class DistributedTrainer(NeuralTrainer):
    """
    Distributed neural network trainer for engineering applications.
    """
    
    def __init__(self, config: Optional[TrainingConfig] = None, world_size: int = 1, rank: int = 0):
        """
        Initialize distributed trainer.
        
        Args:
            config: Training configuration
            world_size: Number of processes
            rank: Process rank
        """
        super(DistributedTrainer, self).__init__(config)
        
        self.world_size = world_size
        self.rank = rank
        
        # Initialize distributed training
        if world_size > 1:
            torch.distributed.init_process_group(
                backend='nccl' if torch.cuda.is_available() else 'gloo',
                world_size=world_size,
                rank=rank
            )
        
        self.logger.info(f"Distributed trainer initialized with world_size={world_size}, rank={rank}")
    
    def setup_training(self, model: nn.Module, train_data: torch.utils.data.DataLoader,
                      val_data: Optional[torch.utils.data.DataLoader] = None):
        """
        Setup distributed training components.
        
        Args:
            model: Neural network model
            train_data: Training data loader
            val_data: Validation data loader
        """
        super().setup_training(model, train_data, val_data)
        
        # Wrap model for distributed training
        if self.world_size > 1:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.rank] if torch.cuda.is_available() else None
            )
    
    def _train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch with distributed training."""
        if self.world_size > 1:
            # Set epoch for distributed sampler
            self.train_data.sampler.set_epoch(self.current_epoch)
        
        return super()._train_epoch()


class FederatedTrainer(NeuralTrainer):
    """
    Federated learning trainer for engineering applications.
    """
    
    def __init__(self, config: Optional[TrainingConfig] = None, num_clients: int = 10):
        """
        Initialize federated trainer.
        
        Args:
            config: Training configuration
            num_clients: Number of clients
        """
        super(FederatedTrainer, self).__init__(config)
        
        self.num_clients = num_clients
        self.client_models = {}
        self.client_data = {}
        
        self.logger.info(f"Federated trainer initialized with {num_clients} clients")
    
    def add_client(self, client_id: str, model: nn.Module, data: torch.utils.data.DataLoader):
        """
        Add client to federated training.
        
        Args:
            client_id: Client identifier
            model: Client model
            data: Client data
        """
        self.client_models[client_id] = model.to(self.device)
        self.client_data[client_id] = data
        
        self.logger.info(f"Added client {client_id}")
    
    async def federated_train(self, rounds: int = 10) -> Dict[str, TrainingMetrics]:
        """
        Perform federated training.
        
        Args:
            rounds: Number of federated rounds
            
        Returns:
            Training metrics for each client
        """
        self.logger.info(f"Starting federated training for {rounds} rounds")
        
        client_metrics = {}
        
        for round_num in range(rounds):
            self.logger.info(f"Federated round {round_num + 1}/{rounds}")
            
            # Train each client
            for client_id in self.client_models:
                client_model = self.client_models[client_id]
                client_data = self.client_data[client_id]
                
                # Train client model
                client_trainer = NeuralTrainer(self.config)
                client_trainer.setup_training(client_model, client_data)
                metrics = await client_trainer.train()
                
                if client_id not in client_metrics:
                    client_metrics[client_id] = []
                client_metrics[client_id].append(metrics)
            
            # Aggregate models (simple averaging)
            self._aggregate_models()
        
        return client_metrics
    
    def _aggregate_models(self):
        """Aggregate client models using federated averaging."""
        if not self.client_models:
            return
        
        # Get reference model
        reference_model = list(self.client_models.values())[0]
        
        # Initialize aggregated parameters
        aggregated_params = {}
        for name, param in reference_model.named_parameters():
            aggregated_params[name] = torch.zeros_like(param.data)
        
        # Sum parameters from all clients
        for client_model in self.client_models.values():
            for name, param in client_model.named_parameters():
                aggregated_params[name] += param.data
        
        # Average parameters
        for name, param in aggregated_params.items():
            aggregated_params[name] /= len(self.client_models)
        
        # Update all client models
        for client_model in self.client_models.values():
            for name, param in client_model.named_parameters():
                param.data = aggregated_params[name]
        
        self.logger.info("Models aggregated using federated averaging")
