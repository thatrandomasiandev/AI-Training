"""
Advanced ML models for engineering applications.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, List, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, VotingClassifier, VotingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import logging


class EngineeringClassifier:
    """
    Advanced classifier for engineering applications.
    """
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.ensemble_model = None
        self.is_trained = False
    
    async def train(self, X: np.ndarray, y: np.ndarray, task_type: str = "classification") -> Dict[str, Any]:
        """Train ensemble of classifiers."""
        self.logger.info("Training engineering classifier ensemble")
        
        # Initialize individual models
        self.models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'svm': SVC(probability=True, random_state=42),
            'mlp': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42),
            'logistic': LogisticRegression(random_state=42),
            'decision_tree': DecisionTreeClassifier(random_state=42)
        }
        
        # Train individual models
        training_results = {}
        for name, model in self.models.items():
            try:
                model.fit(X, y)
                # Calculate training accuracy
                train_pred = model.predict(X)
                accuracy = accuracy_score(y, train_pred)
                training_results[name] = {
                    'accuracy': accuracy,
                    'status': 'success'
                }
            except Exception as e:
                self.logger.error(f"Error training {name}: {e}")
                training_results[name] = {
                    'accuracy': 0.0,
                    'status': 'failed',
                    'error': str(e)
                }
        
        # Create ensemble
        successful_models = [(name, model) for name, model in self.models.items() 
                           if training_results[name]['status'] == 'success']
        
        if successful_models:
            self.ensemble_model = VotingClassifier(
                estimators=successful_models,
                voting='soft'
            )
            self.ensemble_model.fit(X, y)
            self.is_trained = True
        
        return {
            'training_results': training_results,
            'ensemble_created': self.ensemble_model is not None,
            'models_trained': len(successful_models)
        }
    
    async def predict_ensemble(self, X: np.ndarray) -> Dict[str, Any]:
        """Make ensemble predictions."""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        # Individual model predictions
        individual_predictions = {}
        confidence_scores = []
        
        for name, model in self.models.items():
            try:
                pred = model.predict(X)
                proba = model.predict_proba(X) if hasattr(model, 'predict_proba') else None
                individual_predictions[name] = {
                    'predictions': pred,
                    'probabilities': proba
                }
                if proba is not None:
                    confidence_scores.append(np.max(proba, axis=1))
            except Exception as e:
                self.logger.error(f"Error predicting with {name}: {e}")
        
        # Ensemble prediction
        ensemble_pred = self.ensemble_model.predict(X)
        ensemble_proba = self.ensemble_model.predict_proba(X)
        ensemble_confidence = np.max(ensemble_proba, axis=1)
        
        return {
            'predictions': ensemble_pred,
            'probabilities': ensemble_proba,
            'confidence_scores': ensemble_confidence,
            'individual_predictions': individual_predictions,
            'model_contributions': self._calculate_contributions(individual_predictions, ensemble_pred)
        }
    
    def _calculate_contributions(self, individual_predictions: Dict, ensemble_pred: np.ndarray) -> Dict[str, float]:
        """Calculate contribution of each model to ensemble prediction."""
        contributions = {}
        for name, pred_data in individual_predictions.items():
            individual_pred = pred_data['predictions']
            agreement = np.mean(individual_pred == ensemble_pred)
            contributions[name] = agreement
        return contributions
    
    def get_feature_importance(self) -> Dict[str, np.ndarray]:
        """Get feature importance from models that support it."""
        importance = {}
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importance[name] = model.feature_importances_
        return importance


class EngineeringRegressor:
    """
    Advanced regressor for engineering applications.
    """
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.ensemble_model = None
        self.is_trained = False
    
    async def train(self, X: np.ndarray, y: np.ndarray, task_type: str = "regression") -> Dict[str, Any]:
        """Train ensemble of regressors."""
        self.logger.info("Training engineering regressor ensemble")
        
        # Initialize individual models
        self.models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'svr': SVR(),
            'mlp': MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42),
            'linear': LinearRegression(),
            'ridge': Ridge(random_state=42),
            'lasso': Lasso(random_state=42),
            'decision_tree': DecisionTreeRegressor(random_state=42)
        }
        
        # Train individual models
        training_results = {}
        for name, model in self.models.items():
            try:
                model.fit(X, y)
                # Calculate training R²
                train_pred = model.predict(X)
                r2 = r2_score(y, train_pred)
                mse = mean_squared_error(y, train_pred)
                training_results[name] = {
                    'r2': r2,
                    'mse': mse,
                    'status': 'success'
                }
            except Exception as e:
                self.logger.error(f"Error training {name}: {e}")
                training_results[name] = {
                    'r2': 0.0,
                    'mse': float('inf'),
                    'status': 'failed',
                    'error': str(e)
                }
        
        # Create ensemble
        successful_models = [(name, model) for name, model in self.models.items() 
                           if training_results[name]['status'] == 'success']
        
        if successful_models:
            self.ensemble_model = VotingRegressor(
                estimators=successful_models
            )
            self.ensemble_model.fit(X, y)
            self.is_trained = True
        
        return {
            'training_results': training_results,
            'ensemble_created': self.ensemble_model is not None,
            'models_trained': len(successful_models)
        }
    
    async def predict_ensemble(self, X: np.ndarray) -> Dict[str, Any]:
        """Make ensemble predictions."""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        # Individual model predictions
        individual_predictions = {}
        
        for name, model in self.models.items():
            try:
                pred = model.predict(X)
                individual_predictions[name] = {
                    'predictions': pred
                }
            except Exception as e:
                self.logger.error(f"Error predicting with {name}: {e}")
        
        # Ensemble prediction
        ensemble_pred = self.ensemble_model.predict(X)
        
        # Calculate prediction variance as uncertainty measure
        pred_array = np.array([pred_data['predictions'] for pred_data in individual_predictions.values()])
        prediction_variance = np.var(pred_array, axis=0)
        
        return {
            'predictions': ensemble_pred,
            'uncertainty': prediction_variance,
            'individual_predictions': individual_predictions,
            'model_contributions': self._calculate_contributions(individual_predictions, ensemble_pred)
        }
    
    def _calculate_contributions(self, individual_predictions: Dict, ensemble_pred: np.ndarray) -> Dict[str, float]:
        """Calculate contribution of each model to ensemble prediction."""
        contributions = {}
        for name, pred_data in individual_predictions.items():
            individual_pred = pred_data['predictions']
            # Calculate correlation with ensemble prediction
            correlation = np.corrcoef(individual_pred, ensemble_pred)[0, 1]
            contributions[name] = abs(correlation) if not np.isnan(correlation) else 0.0
        return contributions
    
    def get_feature_importance(self) -> Dict[str, np.ndarray]:
        """Get feature importance from models that support it."""
        importance = {}
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importance[name] = model.feature_importances_
        return importance


class EnsembleModel:
    """
    Advanced ensemble model combining multiple ML approaches.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.classifier = EngineeringClassifier()
        self.regressor = EngineeringRegressor()
        self.is_trained = False
    
    async def train(self, X: np.ndarray, y: np.ndarray, task_type: str) -> Dict[str, Any]:
        """Train ensemble model."""
        if task_type == "classification":
            result = await self.classifier.train(X, y, task_type)
        elif task_type == "regression":
            result = await self.regressor.train(X, y, task_type)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
        
        self.is_trained = True
        return result
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        # This would need to be implemented based on the trained task type
        # For now, return placeholder
        return np.zeros(X.shape[0])
    
    def get_feature_importance(self) -> Dict[str, np.ndarray]:
        """Get feature importance."""
        importance = {}
        if hasattr(self.classifier, 'get_feature_importance'):
            importance.update(self.classifier.get_feature_importance())
        if hasattr(self.regressor, 'get_feature_importance'):
            importance.update(self.regressor.get_feature_importance())
        return importance


class FeatureEngineer:
    """
    Advanced feature engineering for engineering applications.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.feature_transforms = {}
    
    def create_engineering_features(self, X: np.ndarray, feature_names: List[str] = None) -> np.ndarray:
        """Create engineering-specific features."""
        features = []
        
        # Basic statistical features
        features.append(np.mean(X, axis=1))  # Mean
        features.append(np.std(X, axis=1))   # Standard deviation
        features.append(np.max(X, axis=1))   # Maximum
        features.append(np.min(X, axis=1))   # Minimum
        
        # Engineering-specific features
        if X.shape[1] >= 2:
            # Ratio features
            for i in range(X.shape[1]):
                for j in range(i+1, X.shape[1]):
                    ratio = X[:, i] / (X[:, j] + 1e-8)  # Avoid division by zero
                    features.append(ratio)
            
            # Product features
            for i in range(X.shape[1]):
                for j in range(i+1, X.shape[1]):
                    product = X[:, i] * X[:, j]
                    features.append(product)
        
        # Polynomial features
        for i in range(X.shape[1]):
            features.append(X[:, i] ** 2)  # Squared
            features.append(np.sqrt(np.abs(X[:, i])))  # Square root
        
        return np.column_stack(features)
    
    def create_interaction_features(self, X: np.ndarray) -> np.ndarray:
        """Create interaction features between variables."""
        n_features = X.shape[1]
        interaction_features = []
        
        for i in range(n_features):
            for j in range(i+1, n_features):
                # Multiplicative interaction
                interaction_features.append(X[:, i] * X[:, j])
                # Additive interaction
                interaction_features.append(X[:, i] + X[:, j])
                # Difference interaction
                interaction_features.append(X[:, i] - X[:, j])
        
        return np.column_stack(interaction_features) if interaction_features else X


class ModelSelector:
    """
    Intelligent model selection for engineering problems.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.model_performance = {}
    
    def select_best_model(self, X: np.ndarray, y: np.ndarray, task_type: str) -> str:
        """Select the best model based on data characteristics."""
        # Analyze data characteristics
        data_characteristics = self._analyze_data(X, y)
        
        # Model selection logic based on data characteristics
        if task_type == "classification":
            if data_characteristics['n_samples'] < 1000:
                return "decision_tree"
            elif data_characteristics['n_features'] > data_characteristics['n_samples']:
                return "logistic"
            else:
                return "random_forest"
        else:  # regression
            if data_characteristics['n_samples'] < 1000:
                return "linear"
            elif data_characteristics['n_features'] > data_characteristics['n_samples']:
                return "ridge"
            else:
                return "random_forest"
    
    def _analyze_data(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Analyze data characteristics."""
        return {
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'feature_variance': np.var(X, axis=0),
            'target_variance': np.var(y) if len(y.shape) == 1 else np.var(y, axis=0),
            'missing_values': np.isnan(X).sum(),
            'outliers': self._detect_outliers(X)
        }
    
    def _detect_outliers(self, X: np.ndarray) -> int:
        """Detect outliers using IQR method."""
        Q1 = np.percentile(X, 25, axis=0)
        Q3 = np.percentile(X, 75, axis=0)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = ((X < lower_bound) | (X > upper_bound)).any(axis=1)
        return np.sum(outliers)
