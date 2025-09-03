"""
Advanced Machine Learning module for engineering applications.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
import joblib
import optuna
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim

from .models import EngineeringClassifier, EngineeringRegressor, EnsembleModel
from .preprocessing import DataPreprocessor, FeatureSelector
from .validation import ModelValidator, CrossValidator
from ..utils.config import Config


class MLModule:
    """
    Advanced Machine Learning module for engineering applications.
    
    Provides comprehensive ML capabilities including:
    - Classification and regression models
    - Feature engineering and selection
    - Model ensemble and optimization
    - Hyperparameter tuning
    - Model validation and evaluation
    """
    
    def __init__(self, config: Config, device: str = "cpu"):
        """
        Initialize the ML module.
        
        Args:
            config: Configuration object
            device: Device to use for computations
        """
        self.config = config
        self.device = device
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.preprocessor = DataPreprocessor()
        self.feature_selector = FeatureSelector()
        self.validator = ModelValidator()
        self.cross_validator = CrossValidator()
        
        # Model storage
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        
        # Engineering-specific models
        self.classifier = EngineeringClassifier(device)
        self.regressor = EngineeringRegressor(device)
        self.ensemble = EnsembleModel()
        
        self.logger.info("ML Module initialized")
    
    async def analyze(self, data: np.ndarray, task_type: str = "auto") -> Dict[str, Any]:
        """
        Analyze data using appropriate ML models.
        
        Args:
            data: Input data array
            task_type: Type of task ('classification', 'regression', 'auto')
            
        Returns:
            Analysis results
        """
        self.logger.info(f"Analyzing data with shape {data.shape}, task: {task_type}")
        
        # Preprocess data
        processed_data = self.preprocessor.preprocess(data)
        
        # Auto-detect task type if needed
        if task_type == "auto":
            task_type = self._detect_task_type(processed_data)
        
        # Select appropriate model
        if task_type == "classification":
            result = await self._classify(processed_data)
        elif task_type == "regression":
            result = await self._regress(processed_data)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
        
        return result
    
    async def _classify(self, data: np.ndarray) -> Dict[str, Any]:
        """Perform classification analysis."""
        # Use ensemble of classifiers
        predictions = await self.classifier.predict_ensemble(data)
        
        # Calculate confidence
        confidence = np.mean(predictions['confidence_scores'])
        
        return {
            "task_type": "classification",
            "predictions": predictions['predictions'],
            "confidence_scores": predictions['confidence_scores'],
            "overall_confidence": confidence,
            "model_contributions": predictions['model_contributions']
        }
    
    async def _regress(self, data: np.ndarray) -> Dict[str, Any]:
        """Perform regression analysis."""
        # Use ensemble of regressors
        predictions = await self.regressor.predict_ensemble(data)
        
        # Calculate confidence based on prediction variance
        confidence = 1.0 / (1.0 + np.var(predictions['predictions']))
        
        return {
            "task_type": "regression",
            "predictions": predictions['predictions'],
            "confidence": confidence,
            "uncertainty": np.std(predictions['predictions']),
            "model_contributions": predictions['model_contributions']
        }
    
    def _detect_task_type(self, data: np.ndarray) -> str:
        """Auto-detect whether this is a classification or regression task."""
        # Simple heuristic: if target values are discrete, it's classification
        if len(data.shape) > 1 and data.shape[1] > 1:
            target = data[:, -1]  # Assume last column is target
            unique_values = len(np.unique(target))
            if unique_values < 10:  # Arbitrary threshold
                return "classification"
        return "regression"
    
    async def train_model(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        model_type: str = "ensemble",
        task_type: str = "auto"
    ) -> Dict[str, Any]:
        """
        Train a machine learning model.
        
        Args:
            X: Training features
            y: Training targets
            model_type: Type of model to train
            task_type: Type of task
            
        Returns:
            Training results
        """
        self.logger.info(f"Training {model_type} model for {task_type} task")
        
        # Preprocess data
        X_processed = self.preprocessor.preprocess(X)
        y_processed = self.preprocessor.preprocess_targets(y, task_type)
        
        # Feature selection
        X_selected = self.feature_selector.select_features(X_processed, y_processed)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y_processed, test_size=0.2, random_state=42
        )
        
        # Train model based on type
        if model_type == "ensemble":
            model = self.ensemble
        elif model_type == "classifier":
            model = self.classifier
        elif model_type == "regressor":
            model = self.regressor
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Train the model
        training_result = await model.train(X_train, y_train, task_type)
        
        # Validate model
        validation_result = self.validator.validate(model, X_test, y_test, task_type)
        
        # Store model
        model_id = f"{model_type}_{task_type}_{len(self.models)}"
        self.models[model_id] = model
        
        return {
            "model_id": model_id,
            "training_result": training_result,
            "validation_result": validation_result,
            "feature_importance": model.get_feature_importance() if hasattr(model, 'get_feature_importance') else None
        }
    
    async def optimize_hyperparameters(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_type: str,
        task_type: str,
        n_trials: int = 100
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Optuna.
        
        Args:
            X: Training features
            y: Training targets
            model_type: Type of model
            task_type: Type of task
            n_trials: Number of optimization trials
            
        Returns:
            Optimization results
        """
        self.logger.info(f"Optimizing hyperparameters for {model_type}")
        
        def objective(trial):
            # Define hyperparameter space
            if model_type == "random_forest":
                if task_type == "classification":
                    model = RandomForestClassifier(
                        n_estimators=trial.suggest_int('n_estimators', 10, 200),
                        max_depth=trial.suggest_int('max_depth', 3, 20),
                        min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
                        min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10)
                    )
                else:
                    model = RandomForestRegressor(
                        n_estimators=trial.suggest_int('n_estimators', 10, 200),
                        max_depth=trial.suggest_int('max_depth', 3, 20),
                        min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
                        min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10)
                    )
            elif model_type == "svm":
                if task_type == "classification":
                    model = SVC(
                        C=trial.suggest_float('C', 1e-4, 1e4, log=True),
                        gamma=trial.suggest_float('gamma', 1e-4, 1e4, log=True),
                        kernel=trial.suggest_categorical('kernel', ['rbf', 'poly', 'sigmoid'])
                    )
                else:
                    model = SVR(
                        C=trial.suggest_float('C', 1e-4, 1e4, log=True),
                        gamma=trial.suggest_float('gamma', 1e-4, 1e4, log=True),
                        kernel=trial.suggest_categorical('kernel', ['rbf', 'poly', 'sigmoid'])
                    )
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Cross-validation score
            scores = cross_val_score(model, X, y, cv=5, scoring='accuracy' if task_type == 'classification' else 'r2')
            return scores.mean()
        
        # Run optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        return {
            "best_params": study.best_params,
            "best_score": study.best_value,
            "optimization_history": study.trials
        }
    
    def predict(self, X: np.ndarray, model_id: str) -> Dict[str, Any]:
        """
        Make predictions using a trained model.
        
        Args:
            X: Input features
            model_id: ID of the trained model
            
        Returns:
            Predictions
        """
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        model = self.models[model_id]
        X_processed = self.preprocessor.preprocess(X)
        X_selected = self.feature_selector.transform(X_processed)
        
        predictions = model.predict(X_selected)
        
        return {
            "predictions": predictions,
            "model_id": model_id,
            "confidence": getattr(model, 'confidence', 0.0)
        }
    
    def validate_design(self, design_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate engineering design using ML models.
        
        Args:
            design_parameters: Design parameters to validate
            
        Returns:
            Validation results
        """
        self.logger.info("Validating engineering design")
        
        # Convert design parameters to feature vector
        features = self._design_to_features(design_parameters)
        
        # Use appropriate models for validation
        validation_results = {}
        
        # Structural validation
        if "structural" in self.models:
            structural_result = self.predict(features, "structural")
            validation_results["structural"] = structural_result
        
        # Performance validation
        if "performance" in self.models:
            performance_result = self.predict(features, "performance")
            validation_results["performance"] = performance_result
        
        # Safety validation
        if "safety" in self.models:
            safety_result = self.predict(features, "safety")
            validation_results["safety"] = safety_result
        
        # Calculate overall score
        overall_score = np.mean([r.get("confidence", 0.0) for r in validation_results.values()])
        
        return {
            "validation_results": validation_results,
            "overall_score": overall_score,
            "recommendations": self._generate_recommendations(validation_results)
        }
    
    def _design_to_features(self, design_parameters: Dict[str, Any]) -> np.ndarray:
        """Convert design parameters to feature vector."""
        # This would be implemented based on specific engineering domain
        # For now, return a placeholder
        return np.array([list(design_parameters.values())])
    
    def _generate_recommendations(self, validation_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        for category, result in validation_results.items():
            confidence = result.get("confidence", 0.0)
            if confidence < 0.7:
                recommendations.append(f"Improve {category} design - confidence: {confidence:.2f}")
        
        return recommendations
    
    def get_feature_importance(self, model_id: str) -> Dict[str, float]:
        """Get feature importance for a trained model."""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        model = self.models[model_id]
        if hasattr(model, 'get_feature_importance'):
            return model.get_feature_importance()
        else:
            return {}
    
    def save_model(self, model_id: str, filepath: str):
        """Save a trained model to disk."""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        model = self.models[model_id]
        joblib.dump(model, filepath)
        self.logger.info(f"Model {model_id} saved to {filepath}")
    
    def load_model(self, model_id: str, filepath: str):
        """Load a trained model from disk."""
        model = joblib.load(filepath)
        self.models[model_id] = model
        self.logger.info(f"Model {model_id} loaded from {filepath}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the ML module."""
        return {
            "device": self.device,
            "models_loaded": len(self.models),
            "model_ids": list(self.models.keys()),
            "preprocessor_status": self.preprocessor.get_status(),
            "feature_selector_status": self.feature_selector.get_status()
        }
    
    def cleanup(self):
        """Cleanup resources."""
        self.models.clear()
        self.scalers.clear()
        self.encoders.clear()
        self.logger.info("ML Module cleanup complete")
