"""
Main trainer for the AI Engineering System.
"""

import logging
import asyncio
import time
import torch
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
import json
import os
from pathlib import Path

from ..core.orchestrator import AIEngineeringOrchestrator, SystemConfig
from ..core.integration import AIIntegrationFramework, IntegrationConfig
from .data_generator import EngineeringDataGenerator
from .model_trainer import ModelTrainer


@dataclass
class TrainingConfig:
    """Configuration for AI training."""
    
    # General training settings
    device: str = "auto"
    batch_size: int = 32
    learning_rate: float = 0.001
    num_epochs: int = 100
    validation_split: float = 0.2
    test_split: float = 0.1
    
    # Data generation settings
    num_training_samples: int = 10000
    num_validation_samples: int = 2000
    num_test_samples: int = 1000
    
    # Model-specific settings
    ml_models: List[str] = None
    nlp_models: List[str] = None
    vision_models: List[str] = None
    rl_models: List[str] = None
    neural_models: List[str] = None
    
    # Training optimization
    early_stopping_patience: int = 10
    save_best_model: bool = True
    checkpoint_frequency: int = 10
    
    # Output settings
    output_dir: str = "trained_models"
    log_frequency: int = 10
    
    def __post_init__(self):
        if self.ml_models is None:
            self.ml_models = ["random_forest", "svm", "neural_network", "gradient_boosting"]
        if self.nlp_models is None:
            self.nlp_models = ["bert", "transformer", "lstm", "cnn"]
        if self.vision_models is None:
            self.vision_models = ["resnet", "vgg", "efficientnet", "custom_cnn"]
        if self.rl_models is None:
            self.rl_models = ["dqn", "ppo", "a2c", "sac"]
        if self.neural_models is None:
            self.neural_models = ["structural_net", "fluid_net", "material_net", "control_net"]


@dataclass
class TrainingResult:
    """Result of training process."""
    
    success: bool
    training_time: float
    models_trained: Dict[str, List[str]]
    performance_metrics: Dict[str, Any]
    best_models: Dict[str, str]
    training_history: Dict[str, Any]
    error: Optional[str] = None


class AITrainer:
    """
    Main trainer for the AI Engineering System.
    """
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize AI trainer.
        
        Args:
            config: Training configuration
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Initialize components
        self.data_generator = EngineeringDataGenerator(config)
        self.model_trainer = ModelTrainer(config)
        
        # Training state
        self.training_history = {}
        self.best_models = {}
        self.performance_metrics = {}
        
        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("AI Trainer initialized")
    
    async def train_all_models(self) -> TrainingResult:
        """
        Train all AI models.
        
        Returns:
            Training result
        """
        self.logger.info("Starting comprehensive AI model training")
        start_time = time.time()
        
        try:
            # Generate training data
            self.logger.info("Generating training data...")
            training_data = await self.data_generator.generate_all_training_data()
            
            # Train ML models
            self.logger.info("Training ML models...")
            ml_results = await self._train_ml_models(training_data)
            
            # Train NLP models
            self.logger.info("Training NLP models...")
            nlp_results = await self._train_nlp_models(training_data)
            
            # Train Vision models
            self.logger.info("Training Vision models...")
            vision_results = await self._train_vision_models(training_data)
            
            # Train RL models
            self.logger.info("Training RL models...")
            rl_results = await self._train_rl_models(training_data)
            
            # Train Neural models
            self.logger.info("Training Neural models...")
            neural_results = await self._train_neural_models(training_data)
            
            # Compile results
            training_time = time.time() - start_time
            models_trained = {
                "ml": list(ml_results.keys()),
                "nlp": list(nlp_results.keys()),
                "vision": list(vision_results.keys()),
                "rl": list(rl_results.keys()),
                "neural": list(neural_results.keys())
            }
            
            performance_metrics = {
                "ml": ml_results,
                "nlp": nlp_results,
                "vision": vision_results,
                "rl": rl_results,
                "neural": neural_results
            }
            
            # Save training results
            await self._save_training_results(models_trained, performance_metrics)
            
            self.logger.info(f"Training completed successfully in {training_time:.2f} seconds")
            
            return TrainingResult(
                success=True,
                training_time=training_time,
                models_trained=models_trained,
                performance_metrics=performance_metrics,
                best_models=self.best_models,
                training_history=self.training_history
            )
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            return TrainingResult(
                success=False,
                training_time=time.time() - start_time,
                models_trained={},
                performance_metrics={},
                best_models={},
                training_history={},
                error=str(e)
            )
    
    async def _train_ml_models(self, training_data: Dict[str, Any]) -> Dict[str, Any]:
        """Train ML models."""
        results = {}
        
        for model_name in self.config.ml_models:
            try:
                self.logger.info(f"Training ML model: {model_name}")
                
                # Get training data for ML
                X_train = training_data["ml"]["X_train"]
                y_train = training_data["ml"]["y_train"]
                X_val = training_data["ml"]["X_val"]
                y_val = training_data["ml"]["y_val"]
                
                # Train model
                model, metrics = await self.model_trainer.train_ml_model(
                    model_name, X_train, y_train, X_val, y_val
                )
                
                results[model_name] = metrics
                
                # Save best model
                if metrics.get("accuracy", 0) > self.best_models.get("ml", {}).get("accuracy", 0):
                    self.best_models["ml"] = model_name
                    await self._save_model(model, f"ml_{model_name}_best.pth")
                
                self.logger.info(f"ML model {model_name} trained with accuracy: {metrics.get('accuracy', 0):.3f}")
                
            except Exception as e:
                self.logger.error(f"Failed to train ML model {model_name}: {e}")
                results[model_name] = {"error": str(e)}
        
        return results
    
    async def _train_nlp_models(self, training_data: Dict[str, Any]) -> Dict[str, Any]:
        """Train NLP models."""
        results = {}
        
        for model_name in self.config.nlp_models:
            try:
                self.logger.info(f"Training NLP model: {model_name}")
                
                # Get training data for NLP
                text_data = training_data["nlp"]["text_data"]
                labels = training_data["nlp"]["labels"]
                
                # Train model
                model, metrics = await self.model_trainer.train_nlp_model(
                    model_name, text_data, labels
                )
                
                results[model_name] = metrics
                
                # Save best model
                if metrics.get("f1_score", 0) > self.best_models.get("nlp", {}).get("f1_score", 0):
                    self.best_models["nlp"] = model_name
                    await self._save_model(model, f"nlp_{model_name}_best.pth")
                
                self.logger.info(f"NLP model {model_name} trained with F1: {metrics.get('f1_score', 0):.3f}")
                
            except Exception as e:
                self.logger.error(f"Failed to train NLP model {model_name}: {e}")
                results[model_name] = {"error": str(e)}
        
        return results
    
    async def _train_vision_models(self, training_data: Dict[str, Any]) -> Dict[str, Any]:
        """Train Vision models."""
        results = {}
        
        for model_name in self.config.vision_models:
            try:
                self.logger.info(f"Training Vision model: {model_name}")
                
                # Get training data for Vision
                images = training_data["vision"]["images"]
                labels = training_data["vision"]["labels"]
                
                # Train model
                model, metrics = await self.model_trainer.train_vision_model(
                    model_name, images, labels
                )
                
                results[model_name] = metrics
                
                # Save best model
                if metrics.get("accuracy", 0) > self.best_models.get("vision", {}).get("accuracy", 0):
                    self.best_models["vision"] = model_name
                    await self._save_model(model, f"vision_{model_name}_best.pth")
                
                self.logger.info(f"Vision model {model_name} trained with accuracy: {metrics.get('accuracy', 0):.3f}")
                
            except Exception as e:
                self.logger.error(f"Failed to train Vision model {model_name}: {e}")
                results[model_name] = {"error": str(e)}
        
        return results
    
    async def _train_rl_models(self, training_data: Dict[str, Any]) -> Dict[str, Any]:
        """Train RL models."""
        results = {}
        
        for model_name in self.config.rl_models:
            try:
                self.logger.info(f"Training RL model: {model_name}")
                
                # Get training data for RL
                environment = training_data["rl"]["environment"]
                
                # Train model
                model, metrics = await self.model_trainer.train_rl_model(
                    model_name, environment
                )
                
                results[model_name] = metrics
                
                # Save best model
                if metrics.get("reward", 0) > self.best_models.get("rl", {}).get("reward", 0):
                    self.best_models["rl"] = model_name
                    await self._save_model(model, f"rl_{model_name}_best.pth")
                
                self.logger.info(f"RL model {model_name} trained with reward: {metrics.get('reward', 0):.3f}")
                
            except Exception as e:
                self.logger.error(f"Failed to train RL model {model_name}: {e}")
                results[model_name] = {"error": str(e)}
        
        return results
    
    async def _train_neural_models(self, training_data: Dict[str, Any]) -> Dict[str, Any]:
        """Train Neural models."""
        results = {}
        
        for model_name in self.config.neural_models:
            try:
                self.logger.info(f"Training Neural model: {model_name}")
                
                # Get training data for Neural
                data = training_data["neural"][model_name]
                
                # Train model
                model, metrics = await self.model_trainer.train_neural_model(
                    model_name, data
                )
                
                results[model_name] = metrics
                
                # Save best model
                if metrics.get("loss", float('inf')) < self.best_models.get("neural", {}).get("loss", float('inf')):
                    self.best_models["neural"] = model_name
                    await self._save_model(model, f"neural_{model_name}_best.pth")
                
                self.logger.info(f"Neural model {model_name} trained with loss: {metrics.get('loss', 0):.3f}")
                
            except Exception as e:
                self.logger.error(f"Failed to train Neural model {model_name}: {e}")
                results[model_name] = {"error": str(e)}
        
        return results
    
    async def _save_model(self, model: Any, filename: str):
        """Save trained model."""
        try:
            model_path = self.output_dir / filename
            if hasattr(model, 'state_dict'):
                torch.save(model.state_dict(), model_path)
            else:
                torch.save(model, model_path)
            self.logger.info(f"Model saved to {model_path}")
        except Exception as e:
            self.logger.error(f"Failed to save model {filename}: {e}")
    
    async def _save_training_results(self, models_trained: Dict[str, List[str]], performance_metrics: Dict[str, Any]):
        """Save training results."""
        try:
            results = {
                "models_trained": models_trained,
                "performance_metrics": performance_metrics,
                "best_models": self.best_models,
                "training_history": self.training_history,
                "config": self.config.__dict__
            }
            
            results_path = self.output_dir / "training_results.json"
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            self.logger.info(f"Training results saved to {results_path}")
        except Exception as e:
            self.logger.error(f"Failed to save training results: {e}")
    
    async def load_trained_models(self) -> Dict[str, Any]:
        """Load trained models."""
        models = {}
        
        try:
            for model_type in ["ml", "nlp", "vision", "rl", "neural"]:
                if model_type in self.best_models:
                    model_name = self.best_models[model_type]
                    model_path = self.output_dir / f"{model_type}_{model_name}_best.pth"
                    
                    if model_path.exists():
                        model = torch.load(model_path, map_location=self.config.device)
                        models[model_type] = model
                        self.logger.info(f"Loaded {model_type} model: {model_name}")
            
            return models
        except Exception as e:
            self.logger.error(f"Failed to load trained models: {e}")
            return {}
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status."""
        return {
            "config": self.config.__dict__,
            "best_models": self.best_models,
            "training_history": self.training_history,
            "output_dir": str(self.output_dir)
        }
