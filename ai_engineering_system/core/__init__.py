"""
Core AI modules for the engineering system.
"""

from .main import EngineeringAI
from .ml import MLModule
from .nlp import NLPModule
from .vision import VisionModule
from .rl import RLModule
from .neural import NeuralModule
from .integration import AIIntegrationFramework, IntegrationConfig, MultiModalResult, TaskResult
from .orchestrator import AIEngineeringOrchestrator, SystemConfig, EngineeringTask, TaskResult

__all__ = [
    "EngineeringAI",
    "MLModule",
    "NLPModule",
    "VisionModule", 
    "RLModule",
    "NeuralModule",
    "AIIntegrationFramework",
    "IntegrationConfig",
    "MultiModalResult",
    "TaskResult",
    "AIEngineeringOrchestrator",
    "SystemConfig",
    "EngineeringTask"
]
