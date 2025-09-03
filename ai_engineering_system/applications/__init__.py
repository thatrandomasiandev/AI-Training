"""
Engineering applications module for specialized problem solving.
"""

from .structural import StructuralAnalyzer, StructuralOptimizer, StructuralDesigner
from .fluid import FluidAnalyzer, FluidOptimizer, FluidDesigner
from .materials import MaterialAnalyzer, MaterialOptimizer, MaterialDesigner
from .manufacturing import ManufacturingAnalyzer, ManufacturingOptimizer, ManufacturingDesigner
from .control import ControlAnalyzer, ControlOptimizer, ControlDesigner
from .optimization import OptimizationAnalyzer, OptimizationOptimizer, OptimizationDesigner

__all__ = [
    "StructuralAnalyzer",
    "StructuralOptimizer",
    "StructuralDesigner",
    "FluidAnalyzer",
    "FluidOptimizer",
    "FluidDesigner",
    "MaterialAnalyzer",
    "MaterialOptimizer",
    "MaterialDesigner",
    "ManufacturingAnalyzer",
    "ManufacturingOptimizer",
    "ManufacturingDesigner",
    "ControlAnalyzer",
    "ControlOptimizer",
    "ControlDesigner",
    "OptimizationAnalyzer",
    "OptimizationOptimizer",
    "OptimizationDesigner"
]
