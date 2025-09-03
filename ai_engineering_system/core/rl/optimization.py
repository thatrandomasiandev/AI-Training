"""
Optimization utilities for engineering applications.
"""

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, List, Optional, Union, Tuple
import optuna
from scipy.optimize import minimize, differential_evolution
from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class OptimizationResult:
    """Result of optimization process."""
    optimal_parameters: Dict[str, float]
    optimal_value: float
    optimization_time: float
    iterations: int
    convergence: bool
    objective_history: List[float]
    parameter_history: List[Dict[str, float]]


class DesignOptimizer:
    """
    Design optimizer for engineering applications.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.optimization_history = []
        self.best_solutions = []
    
    def optimize_design(self, design_space: Dict[str, Any], objectives: List[str],
                       constraints: List[str], method: str = "genetic") -> OptimizationResult:
        """
        Optimize engineering design.
        
        Args:
            design_space: Design parameter space
            objectives: List of objectives to optimize
            constraints: List of constraints
            method: Optimization method
            
        Returns:
            Optimization result
        """
        self.logger.info(f"Starting design optimization using {method} method")
        
        if method == "genetic":
            return self._genetic_optimization(design_space, objectives, constraints)
        elif method == "gradient":
            return self._gradient_optimization(design_space, objectives, constraints)
        elif method == "random":
            return self._random_optimization(design_space, objectives, constraints)
        else:
            raise ValueError(f"Unknown optimization method: {method}")
    
    def _genetic_optimization(self, design_space: Dict[str, Any], objectives: List[str],
                             constraints: List[str]) -> OptimizationResult:
        """Perform genetic algorithm optimization."""
        # Define objective function
        def objective_function(params):
            return self._evaluate_objectives(params, objectives, constraints)
        
        # Define parameter bounds
        bounds = []
        param_names = []
        
        for param_name, param_info in design_space.items():
            if isinstance(param_info, dict):
                low = param_info.get("low", 0.0)
                high = param_info.get("high", 1.0)
            else:
                low, high = 0.0, 1.0
            
            bounds.append((low, high))
            param_names.append(param_name)
        
        # Run genetic algorithm
        result = differential_evolution(
            objective_function,
            bounds,
            maxiter=100,
            popsize=15,
            seed=42
        )
        
        # Create optimization result
        optimal_parameters = dict(zip(param_names, result.x))
        
        optimization_result = OptimizationResult(
            optimal_parameters=optimal_parameters,
            optimal_value=result.fun,
            optimization_time=0.0,  # Placeholder
            iterations=result.nit,
            convergence=result.success,
            objective_history=[],  # Placeholder
            parameter_history=[]  # Placeholder
        )
        
        self.optimization_history.append(optimization_result)
        
        return optimization_result
    
    def _gradient_optimization(self, design_space: Dict[str, Any], objectives: List[str],
                              constraints: List[str]) -> OptimizationResult:
        """Perform gradient-based optimization."""
        # Define objective function
        def objective_function(params):
            return self._evaluate_objectives(params, objectives, constraints)
        
        # Define parameter bounds
        bounds = []
        param_names = []
        initial_guess = []
        
        for param_name, param_info in design_space.items():
            if isinstance(param_info, dict):
                low = param_info.get("low", 0.0)
                high = param_info.get("high", 1.0)
                initial = param_info.get("initial", (low + high) / 2)
            else:
                low, high = 0.0, 1.0
                initial = 0.5
            
            bounds.append((low, high))
            param_names.append(param_name)
            initial_guess.append(initial)
        
        # Run gradient optimization
        result = minimize(
            objective_function,
            initial_guess,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 100}
        )
        
        # Create optimization result
        optimal_parameters = dict(zip(param_names, result.x))
        
        optimization_result = OptimizationResult(
            optimal_parameters=optimal_parameters,
            optimal_value=result.fun,
            optimization_time=0.0,  # Placeholder
            iterations=result.nit,
            convergence=result.success,
            objective_history=[],  # Placeholder
            parameter_history=[]  # Placeholder
        )
        
        self.optimization_history.append(optimization_result)
        
        return optimization_result
    
    def _random_optimization(self, design_space: Dict[str, Any], objectives: List[str],
                            constraints: List[str]) -> OptimizationResult:
        """Perform random search optimization."""
        # Define objective function
        def objective_function(params):
            return self._evaluate_objectives(params, objectives, constraints)
        
        # Define parameter bounds
        bounds = []
        param_names = []
        
        for param_name, param_info in design_space.items():
            if isinstance(param_info, dict):
                low = param_info.get("low", 0.0)
                high = param_info.get("high", 1.0)
            else:
                low, high = 0.0, 1.0
            
            bounds.append((low, high))
            param_names.append(param_name)
        
        # Random search
        best_value = float('inf')
        best_params = None
        objective_history = []
        parameter_history = []
        
        for i in range(1000):  # 1000 random samples
            # Generate random parameters
            params = []
            for low, high in bounds:
                params.append(np.random.uniform(low, high))
            
            # Evaluate objective
            value = objective_function(params)
            objective_history.append(value)
            parameter_history.append(dict(zip(param_names, params)))
            
            # Update best
            if value < best_value:
                best_value = value
                best_params = params
        
        # Create optimization result
        optimal_parameters = dict(zip(param_names, best_params))
        
        optimization_result = OptimizationResult(
            optimal_parameters=optimal_parameters,
            optimal_value=best_value,
            optimization_time=0.0,  # Placeholder
            iterations=1000,
            convergence=True,
            objective_history=objective_history,
            parameter_history=parameter_history
        )
        
        self.optimization_history.append(optimization_result)
        
        return optimization_result
    
    def _evaluate_objectives(self, params: List[float], objectives: List[str],
                           constraints: List[str]) -> float:
        """Evaluate objectives and constraints."""
        # Placeholder implementation
        # In practice, this would implement actual objective and constraint functions
        
        # Calculate objective value
        objective_value = 0.0
        
        for objective in objectives:
            if objective == "minimize_weight":
                # Example: minimize total weight
                weight = sum(params)
                objective_value += weight
            elif objective == "maximize_strength":
                # Example: maximize strength
                strength = np.prod(params)
                objective_value -= strength  # Negative for maximization
            elif objective == "minimize_cost":
                # Example: minimize cost
                cost = sum(param * (i + 1) for i, param in enumerate(params))
                objective_value += cost
            else:
                # Default objective
                objective_value += np.random.random()
        
        # Add constraint penalties
        for constraint in constraints:
            if constraint == "stress_limit":
                # Example: stress must be below limit
                stress = sum(params)
                limit = 10.0
                if stress > limit:
                    objective_value += (stress - limit) * 10.0  # Penalty
            elif constraint == "deflection_limit":
                # Example: deflection must be below limit
                deflection = np.mean(params)
                limit = 5.0
                if deflection > limit:
                    objective_value += (deflection - limit) * 10.0  # Penalty
        
        return objective_value
    
    def multi_objective_optimization(self, design_space: Dict[str, Any], objectives: List[str],
                                   constraints: List[str], weights: List[float] = None) -> List[OptimizationResult]:
        """
        Perform multi-objective optimization.
        
        Args:
            design_space: Design parameter space
            objectives: List of objectives to optimize
            constraints: List of constraints
            weights: Weights for objectives
            
        Returns:
            List of Pareto optimal solutions
        """
        self.logger.info("Starting multi-objective optimization")
        
        if weights is None:
            weights = [1.0] * len(objectives)
        
        # Define multi-objective function
        def multi_objective_function(params):
            return self._evaluate_multi_objectives(params, objectives, constraints, weights)
        
        # Define parameter bounds
        bounds = []
        param_names = []
        
        for param_name, param_info in design_space.items():
            if isinstance(param_info, dict):
                low = param_info.get("low", 0.0)
                high = param_info.get("high", 1.0)
            else:
                low, high = 0.0, 1.0
            
            bounds.append((low, high))
            param_names.append(param_name)
        
        # Run multi-objective optimization
        # Placeholder implementation - in practice, this would use NSGA-II or similar
        pareto_solutions = []
        
        for i in range(10):  # Generate 10 Pareto solutions
            # Generate random parameters
            params = []
            for low, high in bounds:
                params.append(np.random.uniform(low, high))
            
            # Evaluate objectives
            objective_values = self._evaluate_multi_objectives(params, objectives, constraints, weights)
            
            # Create optimization result
            optimal_parameters = dict(zip(param_names, params))
            
            optimization_result = OptimizationResult(
                optimal_parameters=optimal_parameters,
                optimal_value=objective_values,
                optimization_time=0.0,  # Placeholder
                iterations=1,
                convergence=True,
                objective_history=[objective_values],
                parameter_history=[optimal_parameters]
            )
            
            pareto_solutions.append(optimization_result)
        
        return pareto_solutions
    
    def _evaluate_multi_objectives(self, params: List[float], objectives: List[str],
                                 constraints: List[str], weights: List[float]) -> float:
        """Evaluate multiple objectives with weights."""
        # Placeholder implementation
        # In practice, this would implement actual multi-objective evaluation
        
        total_value = 0.0
        
        for i, objective in enumerate(objectives):
            weight = weights[i] if i < len(weights) else 1.0
            
            if objective == "minimize_weight":
                weight_value = sum(params)
            elif objective == "maximize_strength":
                weight_value = -np.prod(params)  # Negative for maximization
            elif objective == "minimize_cost":
                weight_value = sum(param * (j + 1) for j, param in enumerate(params))
            else:
                weight_value = np.random.random()
            
            total_value += weight * weight_value
        
        # Add constraint penalties
        for constraint in constraints:
            if constraint == "stress_limit":
                stress = sum(params)
                limit = 10.0
                if stress > limit:
                    total_value += (stress - limit) * 10.0
            elif constraint == "deflection_limit":
                deflection = np.mean(params)
                limit = 5.0
                if deflection > limit:
                    total_value += (deflection - limit) * 10.0
        
        return total_value
    
    def get_optimization_history(self) -> List[OptimizationResult]:
        """Get optimization history."""
        return self.optimization_history.copy()
    
    def plot_optimization_history(self, save_path: Optional[str] = None):
        """Plot optimization history."""
        if not self.optimization_history:
            self.logger.warning("No optimization history to plot")
            return
        
        # Plot objective values
        objective_values = [result.optimal_value for result in self.optimization_history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(objective_values)
        plt.title("Optimization History")
        plt.xlabel("Optimization Run")
        plt.ylabel("Objective Value")
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        
        plt.close()
    
    def get_status(self) -> Dict[str, Any]:
        """Get optimizer status."""
        return {
            "optimizer_type": "design",
            "optimization_runs": len(self.optimization_history),
            "best_solutions": len(self.best_solutions)
        }


class ParameterOptimizer:
    """
    Parameter optimizer for engineering applications.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.optimization_history = []
        self.parameter_space = {}
    
    def optimize_parameters(self, parameter_space: Dict[str, Any], objective_function: callable,
                          method: str = "bayesian") -> OptimizationResult:
        """
        Optimize system parameters.
        
        Args:
            parameter_space: Parameter space definition
            objective_function: Objective function to optimize
            method: Optimization method
            
        Returns:
            Optimization result
        """
        self.logger.info(f"Starting parameter optimization using {method} method")
        
        self.parameter_space = parameter_space
        
        if method == "bayesian":
            return self._bayesian_optimization(parameter_space, objective_function)
        elif method == "grid":
            return self._grid_optimization(parameter_space, objective_function)
        elif method == "random":
            return self._random_parameter_optimization(parameter_space, objective_function)
        else:
            raise ValueError(f"Unknown optimization method: {method}")
    
    def _bayesian_optimization(self, parameter_space: Dict[str, Any], objective_function: callable) -> OptimizationResult:
        """Perform Bayesian optimization."""
        # Define objective function for Optuna
        def objective(trial):
            params = {}
            
            for param_name, param_info in parameter_space.items():
                if isinstance(param_info, dict):
                    param_type = param_info.get("type", "float")
                    
                    if param_type == "float":
                        low = param_info.get("low", 0.0)
                        high = param_info.get("high", 1.0)
                        params[param_name] = trial.suggest_float(param_name, low, high)
                    elif param_type == "int":
                        low = param_info.get("low", 0)
                        high = param_info.get("high", 100)
                        params[param_name] = trial.suggest_int(param_name, low, high)
                    elif param_type == "categorical":
                        choices = param_info.get("choices", ["A", "B", "C"])
                        params[param_name] = trial.suggest_categorical(param_name, choices)
                else:
                    # Default float parameter
                    params[param_name] = trial.suggest_float(param_name, 0.0, 1.0)
            
            return objective_function(params)
        
        # Run Bayesian optimization
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=100)
        
        # Create optimization result
        optimization_result = OptimizationResult(
            optimal_parameters=study.best_params,
            optimal_value=study.best_value,
            optimization_time=0.0,  # Placeholder
            iterations=len(study.trials),
            convergence=True,
            objective_history=[trial.value for trial in study.trials],
            parameter_history=[trial.params for trial in study.trials]
        )
        
        self.optimization_history.append(optimization_result)
        
        return optimization_result
    
    def _grid_optimization(self, parameter_space: Dict[str, Any], objective_function: callable) -> OptimizationResult:
        """Perform grid search optimization."""
        # Create parameter grid
        param_grid = {}
        
        for param_name, param_info in parameter_space.items():
            if isinstance(param_info, dict):
                param_type = param_info.get("type", "float")
                
                if param_type == "float":
                    low = param_info.get("low", 0.0)
                    high = param_info.get("high", 1.0)
                    n_points = param_info.get("n_points", 10)
                    param_grid[param_name] = np.linspace(low, high, n_points)
                elif param_type == "int":
                    low = param_info.get("low", 0)
                    high = param_info.get("high", 100)
                    n_points = param_info.get("n_points", 10)
                    param_grid[param_name] = np.linspace(low, high, n_points, dtype=int)
                elif param_type == "categorical":
                    choices = param_info.get("choices", ["A", "B", "C"])
                    param_grid[param_name] = choices
            else:
                # Default float parameter
                param_grid[param_name] = np.linspace(0.0, 1.0, 10)
        
        # Generate parameter combinations
        param_combinations = list(ParameterGrid(param_grid))
        
        # Evaluate all combinations
        best_value = float('inf')
        best_params = None
        objective_history = []
        parameter_history = []
        
        for params in param_combinations:
            value = objective_function(params)
            objective_history.append(value)
            parameter_history.append(params)
            
            if value < best_value:
                best_value = value
                best_params = params
        
        # Create optimization result
        optimization_result = OptimizationResult(
            optimal_parameters=best_params,
            optimal_value=best_value,
            optimization_time=0.0,  # Placeholder
            iterations=len(param_combinations),
            convergence=True,
            objective_history=objective_history,
            parameter_history=parameter_history
        )
        
        self.optimization_history.append(optimization_result)
        
        return optimization_result
    
    def _random_parameter_optimization(self, parameter_space: Dict[str, Any], objective_function: callable) -> OptimizationResult:
        """Perform random parameter optimization."""
        # Define parameter bounds
        bounds = []
        param_names = []
        
        for param_name, param_info in parameter_space.items():
            if isinstance(param_info, dict):
                param_type = param_info.get("type", "float")
                
                if param_type == "float":
                    low = param_info.get("low", 0.0)
                    high = param_info.get("high", 1.0)
                elif param_type == "int":
                    low = param_info.get("low", 0)
                    high = param_info.get("high", 100)
                else:
                    low, high = 0.0, 1.0
            else:
                low, high = 0.0, 1.0
            
            bounds.append((low, high))
            param_names.append(param_name)
        
        # Random search
        best_value = float('inf')
        best_params = None
        objective_history = []
        parameter_history = []
        
        for i in range(1000):  # 1000 random samples
            # Generate random parameters
            params = {}
            for j, (param_name, (low, high)) in enumerate(zip(param_names, bounds)):
                if isinstance(parameter_space[param_name], dict):
                    param_type = parameter_space[param_name].get("type", "float")
                    
                    if param_type == "int":
                        params[param_name] = np.random.randint(low, high + 1)
                    elif param_type == "categorical":
                        choices = parameter_space[param_name].get("choices", ["A", "B", "C"])
                        params[param_name] = np.random.choice(choices)
                    else:
                        params[param_name] = np.random.uniform(low, high)
                else:
                    params[param_name] = np.random.uniform(low, high)
            
            # Evaluate objective
            value = objective_function(params)
            objective_history.append(value)
            parameter_history.append(params)
            
            # Update best
            if value < best_value:
                best_value = value
                best_params = params
        
        # Create optimization result
        optimization_result = OptimizationResult(
            optimal_parameters=best_params,
            optimal_value=best_value,
            optimization_time=0.0,  # Placeholder
            iterations=1000,
            convergence=True,
            objective_history=objective_history,
            parameter_history=parameter_history
        )
        
        self.optimization_history.append(optimization_result)
        
        return optimization_result
    
    def get_optimization_history(self) -> List[OptimizationResult]:
        """Get optimization history."""
        return self.optimization_history.copy()
    
    def plot_optimization_history(self, save_path: Optional[str] = None):
        """Plot optimization history."""
        if not self.optimization_history:
            self.logger.warning("No optimization history to plot")
            return
        
        # Plot objective values
        objective_values = [result.optimal_value for result in self.optimization_history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(objective_values)
        plt.title("Parameter Optimization History")
        plt.xlabel("Optimization Run")
        plt.ylabel("Objective Value")
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        
        plt.close()
    
    def get_status(self) -> Dict[str, Any]:
        """Get optimizer status."""
        return {
            "optimizer_type": "parameter",
            "optimization_runs": len(self.optimization_history),
            "parameter_space_size": len(self.parameter_space)
        }
