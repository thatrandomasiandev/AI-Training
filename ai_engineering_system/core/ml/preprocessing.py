"""
Data preprocessing and feature selection for engineering ML applications.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, SelectPercentile, RFE, SelectFromModel
from sklearn.feature_selection import f_classif, f_regression, mutual_info_classif, mutual_info_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE
import logging


class DataPreprocessor:
    """
    Advanced data preprocessing for engineering applications.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.scalers = {}
        self.imputers = {}
        self.encoders = {}
        self.transformers = {}
        self.is_fitted = False
    
    def preprocess(self, X: np.ndarray, method: str = "standard") -> np.ndarray:
        """
        Preprocess input data.
        
        Args:
            X: Input data
            method: Preprocessing method ('standard', 'minmax', 'robust')
            
        Returns:
            Preprocessed data
        """
        if not self.is_fitted:
            self._fit_preprocessors(X, method)
        
        # Handle missing values
        X_imputed = self._impute_missing_values(X)
        
        # Scale features
        X_scaled = self._scale_features(X_imputed, method)
        
        # Handle outliers
        X_cleaned = self._handle_outliers(X_scaled)
        
        return X_cleaned
    
    def preprocess_targets(self, y: np.ndarray, task_type: str) -> np.ndarray:
        """Preprocess target variables."""
        if task_type == "classification":
            # Encode categorical targets
            if not hasattr(self, 'target_encoder'):
                self.target_encoder = LabelEncoder()
                y_encoded = self.target_encoder.fit_transform(y)
            else:
                y_encoded = self.target_encoder.transform(y)
            return y_encoded
        else:
            # For regression, just return as is (could add scaling if needed)
            return y
    
    def _fit_preprocessors(self, X: np.ndarray, method: str):
        """Fit preprocessing components."""
        # Initialize scalers
        if method == "standard":
            self.scalers['main'] = StandardScaler()
        elif method == "minmax":
            self.scalers['main'] = MinMaxScaler()
        elif method == "robust":
            self.scalers['main'] = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        # Initialize imputers
        self.imputers['simple'] = SimpleImputer(strategy='mean')
        self.imputers['knn'] = KNNImputer(n_neighbors=5)
        
        # Fit scalers and imputers
        X_imputed = self.imputers['simple'].fit_transform(X)
        self.scalers['main'].fit(X_imputed)
        
        self.is_fitted = True
    
    def _impute_missing_values(self, X: np.ndarray) -> np.ndarray:
        """Handle missing values."""
        if np.isnan(X).any():
            # Use KNN imputation for better results
            return self.imputers['knn'].transform(X)
        return X
    
    def _scale_features(self, X: np.ndarray, method: str) -> np.ndarray:
        """Scale features."""
        return self.scalers['main'].transform(X)
    
    def _handle_outliers(self, X: np.ndarray, method: str = "iqr") -> np.ndarray:
        """Handle outliers in the data."""
        if method == "iqr":
            return self._iqr_outlier_handling(X)
        elif method == "zscore":
            return self._zscore_outlier_handling(X)
        else:
            return X
    
    def _iqr_outlier_handling(self, X: np.ndarray) -> np.ndarray:
        """Handle outliers using IQR method."""
        Q1 = np.percentile(X, 25, axis=0)
        Q3 = np.percentile(X, 75, axis=0)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Cap outliers instead of removing them
        X_capped = np.copy(X)
        X_capped = np.where(X_capped < lower_bound, lower_bound, X_capped)
        X_capped = np.where(X_capped > upper_bound, upper_bound, X_capped)
        
        return X_capped
    
    def _zscore_outlier_handling(self, X: np.ndarray, threshold: float = 3.0) -> np.ndarray:
        """Handle outliers using Z-score method."""
        z_scores = np.abs((X - np.mean(X, axis=0)) / np.std(X, axis=0))
        X_capped = np.copy(X)
        
        # Cap outliers
        outlier_mask = z_scores > threshold
        X_capped[outlier_mask] = np.where(
            X[outlier_mask] > np.mean(X, axis=0),
            np.mean(X, axis=0) + threshold * np.std(X, axis=0),
            np.mean(X, axis=0) - threshold * np.std(X, axis=0)
        )[outlier_mask]
        
        return X_capped
    
    def create_polynomial_features(self, X: np.ndarray, degree: int = 2) -> np.ndarray:
        """Create polynomial features."""
        from sklearn.preprocessing import PolynomialFeatures
        
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        return poly.fit_transform(X)
    
    def create_engineering_features(self, X: np.ndarray) -> np.ndarray:
        """Create engineering-specific features."""
        features = []
        
        # Basic statistical features
        features.append(np.mean(X, axis=1))  # Mean
        features.append(np.std(X, axis=1))   # Standard deviation
        features.append(np.max(X, axis=1))   # Maximum
        features.append(np.min(X, axis=1))   # Minimum
        features.append(np.median(X, axis=1))  # Median
        
        # Engineering ratios and interactions
        if X.shape[1] >= 2:
            for i in range(X.shape[1]):
                for j in range(i+1, X.shape[1]):
                    # Ratio features (avoid division by zero)
                    ratio = X[:, i] / (X[:, j] + 1e-8)
                    features.append(ratio)
                    
                    # Product features
                    product = X[:, i] * X[:, j]
                    features.append(product)
        
        # Non-linear transformations
        for i in range(X.shape[1]):
            features.append(X[:, i] ** 2)  # Squared
            features.append(np.sqrt(np.abs(X[:, i])))  # Square root
            features.append(np.log(np.abs(X[:, i]) + 1))  # Log transform
        
        return np.column_stack(features)
    
    def get_status(self) -> Dict[str, Any]:
        """Get preprocessing status."""
        return {
            "is_fitted": self.is_fitted,
            "scalers_available": list(self.scalers.keys()),
            "imputers_available": list(self.imputers.keys()),
            "encoders_available": list(self.encoders.keys())
        }


class FeatureSelector:
    """
    Advanced feature selection for engineering applications.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.selectors = {}
        self.selected_features = None
        self.feature_scores = {}
        self.is_fitted = False
    
    def select_features(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        method: str = "auto",
        n_features: Optional[int] = None,
        task_type: str = "auto"
    ) -> np.ndarray:
        """
        Select the most relevant features.
        
        Args:
            X: Input features
            y: Target variable
            method: Selection method ('auto', 'univariate', 'rfe', 'model_based', 'pca')
            n_features: Number of features to select
            task_type: Type of task ('classification', 'regression', 'auto')
            
        Returns:
            Selected features
        """
        if not self.is_fitted:
            self._fit_feature_selector(X, y, method, n_features, task_type)
        
        return self.transform(X)
    
    def _fit_feature_selector(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        method: str, 
        n_features: Optional[int],
        task_type: str
    ):
        """Fit the feature selector."""
        # Auto-detect task type
        if task_type == "auto":
            task_type = "classification" if len(np.unique(y)) < 10 else "regression"
        
        # Auto-determine number of features
        if n_features is None:
            n_features = min(X.shape[1], max(10, X.shape[1] // 2))
        
        # Auto-select method
        if method == "auto":
            if X.shape[1] > 100:
                method = "model_based"
            elif X.shape[1] > 50:
                method = "rfe"
            else:
                method = "univariate"
        
        # Initialize selector based on method
        if method == "univariate":
            if task_type == "classification":
                score_func = f_classif
            else:
                score_func = f_regression
            
            self.selectors['main'] = SelectKBest(score_func=score_func, k=n_features)
        
        elif method == "rfe":
            if task_type == "classification":
                estimator = RandomForestClassifier(n_estimators=50, random_state=42)
            else:
                estimator = RandomForestRegressor(n_estimators=50, random_state=42)
            
            self.selectors['main'] = RFE(estimator=estimator, n_features_to_select=n_features)
        
        elif method == "model_based":
            if task_type == "classification":
                estimator = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                estimator = RandomForestRegressor(n_estimators=100, random_state=42)
            
            self.selectors['main'] = SelectFromModel(estimator=estimator)
        
        elif method == "pca":
            self.selectors['main'] = PCA(n_components=n_features)
        
        else:
            raise ValueError(f"Unknown selection method: {method}")
        
        # Fit the selector
        self.selectors['main'].fit(X, y)
        
        # Store feature scores if available
        if hasattr(self.selectors['main'], 'scores_'):
            self.feature_scores = dict(zip(
                range(X.shape[1]), 
                self.selectors['main'].scores_
            ))
        
        # Store selected feature indices
        if hasattr(self.selectors['main'], 'get_support'):
            self.selected_features = self.selectors['main'].get_support(indices=True)
        elif hasattr(self.selectors['main'], 'components_'):
            self.selected_features = list(range(self.selectors['main'].components_.shape[0]))
        
        self.is_fitted = True
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using fitted selector."""
        if not self.is_fitted:
            raise ValueError("Feature selector not fitted")
        
        return self.selectors['main'].transform(X)
    
    def get_feature_importance(self) -> Dict[int, float]:
        """Get feature importance scores."""
        return self.feature_scores
    
    def get_selected_features(self) -> List[int]:
        """Get indices of selected features."""
        return self.selected_features.tolist() if self.selected_features is not None else []
    
    def select_features_multi_method(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        methods: List[str] = None,
        task_type: str = "auto"
    ) -> Dict[str, np.ndarray]:
        """Select features using multiple methods and return results."""
        if methods is None:
            methods = ["univariate", "rfe", "model_based"]
        
        results = {}
        for method in methods:
            try:
                selector = FeatureSelector()
                selected_features = selector.select_features(X, y, method=method, task_type=task_type)
                results[method] = selected_features
            except Exception as e:
                self.logger.error(f"Error with method {method}: {e}")
                results[method] = None
        
        return results
    
    def get_status(self) -> Dict[str, Any]:
        """Get feature selector status."""
        return {
            "is_fitted": self.is_fitted,
            "selectors_available": list(self.selectors.keys()),
            "selected_features_count": len(self.selected_features) if self.selected_features is not None else 0,
            "feature_scores_available": len(self.feature_scores) > 0
        }
