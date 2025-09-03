"""
Advanced Multi-Modal AI Engineering System

A comprehensive AI system combining Machine Learning, Natural Language Processing,
Computer Vision, Reinforcement Learning, and Custom Neural Networks to solve
complex engineering problems.
"""

__version__ = "1.0.0"
__author__ = "Joshua Terranova"
__email__ = "joshua@example.com"

# Core imports
from .core import EngineeringAI
from .core.ml import MLModule
from .core.nlp import NLPModule
from .core.vision import VisionModule
from .core.rl import RLModule
from .core.neural import NeuralModule

# Application imports
from .applications.structural import StructuralAnalyzer
from .applications.fluid import FluidDynamicsAI
from .applications.materials import MaterialsAI
from .applications.manufacturing import ManufacturingOptimizer

__all__ = [
    "EngineeringAI",
    "MLModule",
    "NLPModule", 
    "VisionModule",
    "RLModule",
    "NeuralModule",
    "StructuralAnalyzer",
    "FluidDynamicsAI",
    "MaterialsAI",
    "ManufacturingOptimizer",
]
