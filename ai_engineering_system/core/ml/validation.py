"""
Model validation and evaluation for engineering ML applications.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
from sklearn.model_selection import (
    cross_val_score, StratifiedKFold, KFold, 
    validation_curve, learning_curve, permutation_test_score
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
import logging


class ModelValidator:
    """
    Advanced model validation for engineering applications.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.validation_results = {}
    
    def validate(
        self, 
        model: Any, 
        X_test: np.ndarray, 
        y_test: np.ndarray, 
        task_type: str,
        X_train: Optional[np.ndarray] = None,
        y_train: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive model validation.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            task_type: Type of task ('classification' or 'regression')
            X_train: Training features (optional, for additional analysis)
            y_train: Training targets (optional, for additional analysis)
            
        Returns:
            Validation results
        """
        self.logger.info(f"Validating {task_type} model")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Basic validation metrics
        if task_type == "classification":
            validation_results = self._validate_classifier(y_test, y_pred, model, X_test)
        else:
            validation_results = self._validate_regressor(y_test, y_pred, model, X_test)
        
        # Additional validation if training data is available
        if X_train is not None and y_train is not None:
            additional_results = self._additional_validation(model, X_train, y_train, X_test, y_test, task_type)
            validation_results.update(additional_results)
        
        # Store results
        model_id = f"{type(model).__name__}_{len(self.validation_results)}"
        self.validation_results[model_id] = validation_results
        
        return validation_results
    
    def _validate_classifier(self, y_true: np.ndarray, y_pred: np.ndarray, model: Any, X_test: np.ndarray) -> Dict[str, Any]:
        """Validate classification model."""
        results = {
            "task_type": "classification",
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
            "recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
            "f1_score": f1_score(y_true, y_pred, average='weighted', zero_division=0),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            "classification_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        }
        
        # ROC AUC if model supports probability prediction
        try:
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)
                if y_proba.shape[1] == 2:  # Binary classification
                    results["roc_auc"] = roc_auc_score(y_true, y_proba[:, 1])
                else:  # Multi-class
                    results["roc_auc"] = roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
        except Exception as e:
            self.logger.warning(f"Could not calculate ROC AUC: {e}")
            results["roc_auc"] = None
        
        return results
    
    def _validate_regressor(self, y_true: np.ndarray, y_pred: np.ndarray, model: Any, X_test: np.ndarray) -> Dict[str, Any]:
        """Validate regression model."""
        results = {
            "task_type": "regression",
            "mse": mean_squared_error(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "mae": mean_absolute_error(y_true, y_pred),
            "r2_score": r2_score(y_true, y_pred),
            "mape": self._calculate_mape(y_true, y_pred)
        }
        
        # Additional regression metrics
        results["explained_variance"] = 1 - np.var(y_true - y_pred) / np.var(y_true)
        results["max_error"] = np.max(np.abs(y_true - y_pred))
        
        return results
    
    def _calculate_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error."""
        return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    
    def _additional_validation(
        self, 
        model: Any, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_test: np.ndarray, 
        y_test: np.ndarray, 
        task_type: str
    ) -> Dict[str, Any]:
        """Perform additional validation analyses."""
        additional_results = {}
        
        # Cross-validation
        cv_scores = self._cross_validate(model, X_train, y_train, task_type)
        additional_results["cross_validation"] = cv_scores
        
        # Learning curve
        learning_curve_data = self._learning_curve(model, X_train, y_train, task_type)
        additional_results["learning_curve"] = learning_curve_data
        
        # Validation curve (if applicable)
        if hasattr(model, 'get_params'):
            validation_curve_data = self._validation_curve(model, X_train, y_train, task_type)
            additional_results["validation_curve"] = validation_curve_data
        
        return additional_results
    
    def _cross_validate(self, model: Any, X: np.ndarray, y: np.ndarray, task_type: str) -> Dict[str, Any]:
        """Perform cross-validation."""
        if task_type == "classification":
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scoring = 'accuracy'
        else:
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            scoring = 'r2'
        
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        
        return {
            "scores": scores.tolist(),
            "mean_score": np.mean(scores),
            "std_score": np.std(scores),
            "cv_folds": len(scores)
        }
    
    def _learning_curve(self, model: Any, X: np.ndarray, y: np.ndarray, task_type: str) -> Dict[str, Any]:
        """Generate learning curve data."""
        if task_type == "classification":
            scoring = 'accuracy'
        else:
            scoring = 'r2'
        
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, 
            train_sizes=np.linspace(0.1, 1.0, 10),
            cv=3,
            scoring=scoring,
            n_jobs=-1
        )
        
        return {
            "train_sizes": train_sizes.tolist(),
            "train_scores_mean": np.mean(train_scores, axis=1).tolist(),
            "train_scores_std": np.std(train_scores, axis=1).tolist(),
            "val_scores_mean": np.mean(val_scores, axis=1).tolist(),
            "val_scores_std": np.std(val_scores, axis=1).tolist()
        }
    
    def _validation_curve(self, model: Any, X: np.ndarray, y: np.ndarray, task_type: str) -> Dict[str, Any]:
        """Generate validation curve data for hyperparameter tuning."""
        # This would be implemented based on the specific model type
        # For now, return placeholder
        return {"message": "Validation curve analysis not implemented for this model type"}
    
    def permutation_importance(self, model: Any, X: np.ndarray, y: np.ndarray, task_type: str) -> Dict[str, Any]:
        """Calculate permutation importance."""
        if task_type == "classification":
            scoring = 'accuracy'
        else:
            scoring = 'r2'
        
        score, permutation_scores, pvalue = permutation_test_score(
            model, X, y, scoring=scoring, n_permutations=100, random_state=42
        )
        
        return {
            "original_score": score,
            "permutation_scores": permutation_scores.tolist(),
            "p_value": pvalue,
            "importance": score - np.mean(permutation_scores)
        }
    
    def calibration_analysis(self, model: Any, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Analyze model calibration for classification."""
        if not hasattr(model, 'predict_proba'):
            return {"error": "Model does not support probability prediction"}
        
        y_proba = model.predict_proba(X_test)
        
        # For binary classification
        if y_proba.shape[1] == 2:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_test, y_proba[:, 1], n_bins=10
            )
            
            return {
                "fraction_of_positives": fraction_of_positives.tolist(),
                "mean_predicted_value": mean_predicted_value.tolist(),
                "calibration_error": np.mean(np.abs(fraction_of_positives - mean_predicted_value))
            }
        
        return {"message": "Calibration analysis only implemented for binary classification"}
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of all validation results."""
        if not self.validation_results:
            return {"message": "No validation results available"}
        
        summary = {}
        for model_id, results in self.validation_results.items():
            summary[model_id] = {
                "task_type": results.get("task_type"),
                "primary_metric": self._get_primary_metric(results),
                "validation_date": results.get("validation_date", "unknown")
            }
        
        return summary
    
    def _get_primary_metric(self, results: Dict[str, Any]) -> float:
        """Get the primary metric for the model."""
        if results.get("task_type") == "classification":
            return results.get("accuracy", 0.0)
        else:
            return results.get("r2_score", 0.0)


class CrossValidator:
    """
    Advanced cross-validation for engineering applications.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.cv_results = {}
    
    def time_series_cv(
        self, 
        model: Any, 
        X: np.ndarray, 
        y: np.ndarray, 
        n_splits: int = 5
    ) -> Dict[str, Any]:
        """Time series cross-validation for temporal engineering data."""
        from sklearn.model_selection import TimeSeriesSplit
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = []
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            scores.append(score)
        
        return {
            "scores": scores,
            "mean_score": np.mean(scores),
            "std_score": np.std(scores),
            "cv_type": "time_series"
        }
    
    def group_cv(
        self, 
        model: Any, 
        X: np.ndarray, 
        y: np.ndarray, 
        groups: np.ndarray,
        n_splits: int = 5
    ) -> Dict[str, Any]:
        """Group-based cross-validation for engineering data with groups."""
        from sklearn.model_selection import GroupKFold
        
        gkf = GroupKFold(n_splits=n_splits)
        scores = []
        
        for train_idx, test_idx in gkf.split(X, y, groups):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            scores.append(score)
        
        return {
            "scores": scores,
            "mean_score": np.mean(scores),
            "std_score": np.std(scores),
            "cv_type": "group_based"
        }
    
    def nested_cv(
        self, 
        model: Any, 
        X: np.ndarray, 
        y: np.ndarray, 
        param_grid: Dict[str, Any],
        task_type: str = "classification"
    ) -> Dict[str, Any]:
        """Nested cross-validation for unbiased performance estimation."""
        from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold
        
        if task_type == "classification":
            inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        else:
            inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)
            outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
        
        # Inner CV for hyperparameter tuning
        grid_search = GridSearchCV(
            model, param_grid, cv=inner_cv, scoring='accuracy' if task_type == 'classification' else 'r2'
        )
        
        # Outer CV for performance estimation
        nested_scores = cross_val_score(grid_search, X, y, cv=outer_cv)
        
        return {
            "nested_scores": nested_scores.tolist(),
            "mean_score": np.mean(nested_scores),
            "std_score": np.std(nested_scores),
            "cv_type": "nested"
        }
