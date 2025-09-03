"""
Configuration management for the AI engineering system.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import logging


@dataclass
class MLConfig:
    """Machine Learning configuration."""
    default_task_type: str = "auto"
    ensemble_methods: list = field(default_factory=lambda: ["random_forest", "svm", "mlp"])
    cross_validation_folds: int = 5
    hyperparameter_optimization_trials: int = 100
    feature_selection_method: str = "auto"
    preprocessing_method: str = "standard"


@dataclass
class NLPConfig:
    """Natural Language Processing configuration."""
    model_name: str = "bert-base-uncased"
    max_sequence_length: int = 512
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 3
    language: str = "en"


@dataclass
class VisionConfig:
    """Computer Vision configuration."""
    model_name: str = "resnet50"
    input_size: tuple = (224, 224)
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 10
    augmentation: bool = True


@dataclass
class RLConfig:
    """Reinforcement Learning configuration."""
    algorithm: str = "PPO"
    learning_rate: float = 3e-4
    batch_size: int = 64
    buffer_size: int = 10000
    num_episodes: int = 1000
    exploration_rate: float = 0.1


@dataclass
class NeuralConfig:
    """Custom Neural Networks configuration."""
    hidden_layers: list = field(default_factory=lambda: [128, 64, 32])
    activation_function: str = "relu"
    dropout_rate: float = 0.2
    learning_rate: float = 1e-3
    batch_size: int = 32
    num_epochs: int = 100


@dataclass
class SystemConfig:
    """System configuration."""
    device: str = "auto"
    num_workers: int = 4
    memory_limit: str = "8GB"
    log_level: str = "INFO"
    cache_dir: str = "./cache"
    model_dir: str = "./models"
    data_dir: str = "./data"


class Config:
    """
    Main configuration class for the AI engineering system.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to configuration file (YAML format)
        """
        self.logger = logging.getLogger(__name__)
        
        # Default configurations
        self.ml = MLConfig()
        self.nlp = NLPConfig()
        self.vision = VisionConfig()
        self.rl = RLConfig()
        self.neural = NeuralConfig()
        self.system = SystemConfig()
        
        # Load configuration from file if provided
        if config_path and os.path.exists(config_path):
            self.load_from_file(config_path)
        elif config_path:
            self.logger.warning(f"Configuration file not found: {config_path}")
        
        # Create directories
        self._create_directories()
    
    def load_from_file(self, config_path: str):
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                config_data = yaml.safe_load(file)
            
            # Update configurations
            if 'ml' in config_data:
                self._update_dataclass(self.ml, config_data['ml'])
            if 'nlp' in config_data:
                self._update_dataclass(self.nlp, config_data['nlp'])
            if 'vision' in config_data:
                self._update_dataclass(self.vision, config_data['vision'])
            if 'rl' in config_data:
                self._update_dataclass(self.rl, config_data['rl'])
            if 'neural' in config_data:
                self._update_dataclass(self.neural, config_data['neural'])
            if 'system' in config_data:
                self._update_dataclass(self.system, config_data['system'])
            
            self.logger.info(f"Configuration loaded from {config_path}")
            
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
    
    def save_to_file(self, config_path: str):
        """Save configuration to YAML file."""
        try:
            config_data = {
                'ml': self._dataclass_to_dict(self.ml),
                'nlp': self._dataclass_to_dict(self.nlp),
                'vision': self._dataclass_to_dict(self.vision),
                'rl': self._dataclass_to_dict(self.rl),
                'neural': self._dataclass_to_dict(self.neural),
                'system': self._dataclass_to_dict(self.system)
            }
            
            with open(config_path, 'w') as file:
                yaml.dump(config_data, file, default_flow_style=False, indent=2)
            
            self.logger.info(f"Configuration saved to {config_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")
    
    def _update_dataclass(self, dataclass_instance, update_dict: Dict[str, Any]):
        """Update dataclass instance with dictionary values."""
        for key, value in update_dict.items():
            if hasattr(dataclass_instance, key):
                setattr(dataclass_instance, key, value)
    
    def _dataclass_to_dict(self, dataclass_instance) -> Dict[str, Any]:
        """Convert dataclass instance to dictionary."""
        return {
            field.name: getattr(dataclass_instance, field.name)
            for field in dataclass_instance.__dataclass_fields__.values()
        }
    
    def _create_directories(self):
        """Create necessary directories."""
        directories = [
            self.system.cache_dir,
            self.system.model_dir,
            self.system.data_dir
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def get_ml_config(self) -> MLConfig:
        """Get ML configuration."""
        return self.ml
    
    def get_nlp_config(self) -> NLPConfig:
        """Get NLP configuration."""
        return self.nlp
    
    def get_vision_config(self) -> VisionConfig:
        """Get Vision configuration."""
        return self.vision
    
    def get_rl_config(self) -> RLConfig:
        """Get RL configuration."""
        return self.rl
    
    def get_neural_config(self) -> NeuralConfig:
        """Get Neural configuration."""
        return self.neural
    
    def get_system_config(self) -> SystemConfig:
        """Get System configuration."""
        return self.system
    
    def update_config(self, section: str, key: str, value: Any):
        """Update a specific configuration value."""
        if hasattr(self, section):
            section_obj = getattr(self, section)
            if hasattr(section_obj, key):
                setattr(section_obj, key, value)
                self.logger.info(f"Updated {section}.{key} = {value}")
            else:
                self.logger.warning(f"Key {key} not found in {section}")
        else:
            self.logger.warning(f"Section {section} not found")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get configuration summary."""
        return {
            "ml": self._dataclass_to_dict(self.ml),
            "nlp": self._dataclass_to_dict(self.nlp),
            "vision": self._dataclass_to_dict(self.vision),
            "rl": self._dataclass_to_dict(self.rl),
            "neural": self._dataclass_to_dict(self.neural),
            "system": self._dataclass_to_dict(self.system)
        }
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return any issues."""
        issues = []
        
        # Validate ML config
        if self.ml.cross_validation_folds < 2:
            issues.append("ML cross_validation_folds must be >= 2")
        
        if self.ml.hyperparameter_optimization_trials < 1:
            issues.append("ML hyperparameter_optimization_trials must be >= 1")
        
        # Validate NLP config
        if self.nlp.max_sequence_length < 1:
            issues.append("NLP max_sequence_length must be >= 1")
        
        if self.nlp.learning_rate <= 0:
            issues.append("NLP learning_rate must be > 0")
        
        # Validate Vision config
        if len(self.vision.input_size) != 2:
            issues.append("Vision input_size must be a tuple of length 2")
        
        if self.vision.learning_rate <= 0:
            issues.append("Vision learning_rate must be > 0")
        
        # Validate RL config
        if self.rl.learning_rate <= 0:
            issues.append("RL learning_rate must be > 0")
        
        if self.rl.exploration_rate < 0 or self.rl.exploration_rate > 1:
            issues.append("RL exploration_rate must be between 0 and 1")
        
        # Validate Neural config
        if len(self.neural.hidden_layers) == 0:
            issues.append("Neural hidden_layers cannot be empty")
        
        if self.neural.dropout_rate < 0 or self.neural.dropout_rate > 1:
            issues.append("Neural dropout_rate must be between 0 and 1")
        
        # Validate System config
        if self.system.num_workers < 1:
            issues.append("System num_workers must be >= 1")
        
        return issues
