"""
Control system utilities for engineering applications.
"""

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, List, Optional, Union, Tuple
import gymnasium as gym
from stable_baselines3 import PPO, DQN, A2C, SAC
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class ControlSystem:
    """Represents a control system."""
    system_type: str
    parameters: Dict[str, float]
    reference: float
    current_state: np.ndarray
    control_output: float
    performance_metrics: Dict[str, float]


class AdaptiveController:
    """
    Adaptive controller for engineering applications.
    """
    
    def __init__(self, system_type: str = "general", learning_rate: float = 0.01):
        """
        Initialize adaptive controller.
        
        Args:
            system_type: Type of control system
            learning_rate: Learning rate for adaptation
        """
        self.logger = logging.getLogger(__name__)
        self.system_type = system_type
        self.learning_rate = learning_rate
        
        # Controller parameters
        self.controller_params = self._initialize_controller_params()
        
        # Performance tracking
        self.performance_history = []
        self.control_history = []
        self.reference_history = []
        
        # RL agent for adaptation
        self.rl_agent = None
        self.adaptation_enabled = True
        
    def _initialize_controller_params(self) -> Dict[str, float]:
        """Initialize controller parameters."""
        # Default PID parameters
        return {
            "kp": 1.0,  # Proportional gain
            "ki": 0.1,  # Integral gain
            "kd": 0.01,  # Derivative gain
            "alpha": 0.1,  # Adaptation rate
            "beta": 0.01,  # Learning rate
            "gamma": 0.99  # Discount factor
        }
    
    async def train(self, env: gym.Env, training_config: Any) -> Dict[str, Any]:
        """
        Train the adaptive controller.
        
        Args:
            env: Training environment
            training_config: Training configuration
            
        Returns:
            Training results
        """
        self.logger.info("Training adaptive controller")
        
        # Initialize RL agent
        self.rl_agent = PPO(
            "MlpPolicy",
            env,
            learning_rate=training_config.learning_rate,
            verbose=1
        )
        
        # Train agent
        self.rl_agent.learn(total_timesteps=training_config.total_timesteps)
        
        # Get training statistics
        training_results = {
            "total_timesteps": training_config.total_timesteps,
            "final_reward": self._evaluate_controller(env),
            "controller_type": "adaptive"
        }
        
        return training_results
    
    def _evaluate_controller(self, env: gym.Env) -> float:
        """Evaluate controller performance."""
        if self.rl_agent is None:
            return 0.0
        
        total_rewards = []
        
        for _ in range(10):  # Run 10 episodes
            obs, _ = env.reset()
            done = False
            episode_reward = 0.0
            
            while not done:
                action, _ = self.rl_agent.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                episode_reward += reward
                done = done or truncated
            
            total_rewards.append(episode_reward)
        
        return np.mean(total_rewards)
    
    def control(self, current_state: np.ndarray, reference: float, dt: float = 0.01) -> float:
        """
        Compute control output.
        
        Args:
            current_state: Current system state
            reference: Reference signal
            dt: Time step
            
        Returns:
            Control output
        """
        # Calculate error
        error = reference - current_state[0]  # Assuming first state is output
        
        # PID control
        control_output = self._pid_control(error, dt)
        
        # Adaptive adjustment
        if self.adaptation_enabled and self.rl_agent is not None:
            adaptive_adjustment = self._adaptive_adjustment(current_state, reference, error)
            control_output += adaptive_adjustment
        
        # Update performance tracking
        self._update_performance_tracking(current_state, reference, control_output)
        
        return control_output
    
    def _pid_control(self, error: float, dt: float) -> float:
        """PID control calculation."""
        # Get PID parameters
        kp = self.controller_params["kp"]
        ki = self.controller_params["ki"]
        kd = self.controller_params["kd"]
        
        # Calculate PID terms
        proportional = kp * error
        
        # Integral term (simplified)
        integral = ki * error * dt
        
        # Derivative term (simplified)
        if hasattr(self, '_prev_error'):
            derivative = kd * (error - self._prev_error) / dt
        else:
            derivative = 0.0
        
        self._prev_error = error
        
        # Total control output
        control_output = proportional + integral + derivative
        
        return control_output
    
    def _adaptive_adjustment(self, current_state: np.ndarray, reference: float, error: float) -> float:
        """Calculate adaptive adjustment using RL."""
        if self.rl_agent is None:
            return 0.0
        
        # Create observation for RL agent
        observation = np.concatenate([
            current_state,
            [reference, error]
        ])
        
        # Get action from RL agent
        action, _ = self.rl_agent.predict(observation, deterministic=True)
        
        # Convert action to control adjustment
        adjustment = action[0] * 0.1  # Scale action
        
        return adjustment
    
    def _update_performance_tracking(self, current_state: np.ndarray, reference: float, control_output: float):
        """Update performance tracking."""
        # Calculate performance metrics
        error = reference - current_state[0]
        performance_metrics = {
            "error": error,
            "control_output": control_output,
            "settling_time": self._calculate_settling_time(),
            "overshoot": self._calculate_overshoot(),
            "steady_state_error": self._calculate_steady_state_error()
        }
        
        # Update history
        self.performance_history.append(performance_metrics)
        self.control_history.append(control_output)
        self.reference_history.append(reference)
        
        # Keep only recent history
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
            self.control_history = self.control_history[-1000:]
            self.reference_history = self.reference_history[-1000:]
    
    def _calculate_settling_time(self) -> float:
        """Calculate settling time."""
        # Placeholder implementation
        # In practice, this would calculate actual settling time
        return 0.0
    
    def _calculate_overshoot(self) -> float:
        """Calculate overshoot."""
        # Placeholder implementation
        # In practice, this would calculate actual overshoot
        return 0.0
    
    def _calculate_steady_state_error(self) -> float:
        """Calculate steady state error."""
        # Placeholder implementation
        # In practice, this would calculate actual steady state error
        return 0.0
    
    def adapt_parameters(self, performance_feedback: Dict[str, float]):
        """
        Adapt controller parameters based on performance feedback.
        
        Args:
            performance_feedback: Performance feedback dictionary
        """
        # Get adaptation parameters
        alpha = self.controller_params["alpha"]
        beta = self.controller_params["beta"]
        
        # Adapt PID parameters based on performance
        if "error" in performance_feedback:
            error = performance_feedback["error"]
            
            # Adapt proportional gain
            if abs(error) > 0.1:  # Large error
                self.controller_params["kp"] += alpha * error * 0.1
            else:  # Small error
                self.controller_params["kp"] -= alpha * 0.01
        
        if "settling_time" in performance_feedback:
            settling_time = performance_feedback["settling_time"]
            
            # Adapt integral gain
            if settling_time > 1.0:  # Slow response
                self.controller_params["ki"] += beta * 0.1
            else:  # Fast response
                self.controller_params["ki"] -= beta * 0.01
        
        if "overshoot" in performance_feedback:
            overshoot = performance_feedback["overshoot"]
            
            # Adapt derivative gain
            if overshoot > 0.1:  # High overshoot
                self.controller_params["kd"] += beta * 0.1
            else:  # Low overshoot
                self.controller_params["kd"] -= beta * 0.01
        
        # Ensure parameters stay within reasonable bounds
        self.controller_params["kp"] = np.clip(self.controller_params["kp"], 0.1, 10.0)
        self.controller_params["ki"] = np.clip(self.controller_params["ki"], 0.01, 1.0)
        self.controller_params["kd"] = np.clip(self.controller_params["kd"], 0.001, 0.1)
    
    def get_optimal_policy(self) -> Dict[str, Any]:
        """Get optimal control policy."""
        return {
            "controller_type": "adaptive",
            "parameters": self.controller_params.copy(),
            "performance_metrics": self._get_performance_metrics(),
            "adaptation_enabled": self.adaptation_enabled
        }
    
    def _get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics."""
        if not self.performance_history:
            return {}
        
        # Calculate average performance metrics
        avg_error = np.mean([p["error"] for p in self.performance_history[-100:]])
        avg_control_output = np.mean([p["control_output"] for p in self.performance_history[-100:]])
        avg_settling_time = np.mean([p["settling_time"] for p in self.performance_history[-100:]])
        avg_overshoot = np.mean([p["overshoot"] for p in self.performance_history[-100:]])
        avg_steady_state_error = np.mean([p["steady_state_error"] for p in self.performance_history[-100:]])
        
        return {
            "average_error": avg_error,
            "average_control_output": avg_control_output,
            "average_settling_time": avg_settling_time,
            "average_overshoot": avg_overshoot,
            "average_steady_state_error": avg_steady_state_error
        }
    
    def plot_performance(self, save_path: Optional[str] = None):
        """Plot controller performance."""
        if not self.performance_history:
            self.logger.warning("No performance history to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Plot error
        errors = [p["error"] for p in self.performance_history]
        axes[0, 0].plot(errors)
        axes[0, 0].set_title("Control Error")
        axes[0, 0].set_xlabel("Time Step")
        axes[0, 0].set_ylabel("Error")
        axes[0, 0].grid(True)
        
        # Plot control output
        control_outputs = [p["control_output"] for p in self.performance_history]
        axes[0, 1].plot(control_outputs)
        axes[0, 1].set_title("Control Output")
        axes[0, 1].set_xlabel("Time Step")
        axes[0, 1].set_ylabel("Control Output")
        axes[0, 1].grid(True)
        
        # Plot reference vs actual
        if self.reference_history:
            axes[1, 0].plot(self.reference_history, label="Reference")
            # Assuming first state is output
            actual_outputs = [p["error"] + ref for p, ref in zip(self.performance_history, self.reference_history)]
            axes[1, 0].plot(actual_outputs, label="Actual")
            axes[1, 0].set_title("Reference vs Actual")
            axes[1, 0].set_xlabel("Time Step")
            axes[1, 0].set_ylabel("Value")
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Plot performance metrics
        settling_times = [p["settling_time"] for p in self.performance_history]
        overshoots = [p["overshoot"] for p in self.performance_history]
        axes[1, 1].plot(settling_times, label="Settling Time")
        axes[1, 1].plot(overshoots, label="Overshoot")
        axes[1, 1].set_title("Performance Metrics")
        axes[1, 1].set_xlabel("Time Step")
        axes[1, 1].set_ylabel("Value")
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        
        plt.close()
    
    def get_status(self) -> Dict[str, Any]:
        """Get controller status."""
        return {
            "controller_type": "adaptive",
            "system_type": self.system_type,
            "parameters": self.controller_params.copy(),
            "adaptation_enabled": self.adaptation_enabled,
            "performance_history_length": len(self.performance_history),
            "rl_agent_available": self.rl_agent is not None
        }


class PIDController:
    """
    PID controller for engineering applications.
    """
    
    def __init__(self, kp: float = 1.0, ki: float = 0.1, kd: float = 0.01):
        """
        Initialize PID controller.
        
        Args:
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
        """
        self.logger = logging.getLogger(__name__)
        
        # PID parameters
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
        # Controller state
        self.prev_error = 0.0
        self.integral = 0.0
        self.prev_time = None
        
        # Performance tracking
        self.performance_history = []
        self.control_history = []
        self.reference_history = []
        
    def control(self, current_state: np.ndarray, reference: float, dt: float = 0.01) -> float:
        """
        Compute PID control output.
        
        Args:
            current_state: Current system state
            reference: Reference signal
            dt: Time step
            
        Returns:
            Control output
        """
        # Calculate error
        error = reference - current_state[0]  # Assuming first state is output
        
        # Calculate PID terms
        proportional = self.kp * error
        
        # Integral term
        self.integral += error * dt
        integral = self.ki * self.integral
        
        # Derivative term
        derivative = self.kd * (error - self.prev_error) / dt
        
        # Total control output
        control_output = proportional + integral + derivative
        
        # Update state
        self.prev_error = error
        self.prev_time = dt
        
        # Update performance tracking
        self._update_performance_tracking(current_state, reference, control_output)
        
        return control_output
    
    def _update_performance_tracking(self, current_state: np.ndarray, reference: float, control_output: float):
        """Update performance tracking."""
        # Calculate performance metrics
        error = reference - current_state[0]
        performance_metrics = {
            "error": error,
            "control_output": control_output,
            "settling_time": self._calculate_settling_time(),
            "overshoot": self._calculate_overshoot(),
            "steady_state_error": self._calculate_steady_state_error()
        }
        
        # Update history
        self.performance_history.append(performance_metrics)
        self.control_history.append(control_output)
        self.reference_history.append(reference)
        
        # Keep only recent history
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
            self.control_history = self.control_history[-1000:]
            self.reference_history = self.reference_history[-1000:]
    
    def _calculate_settling_time(self) -> float:
        """Calculate settling time."""
        # Placeholder implementation
        # In practice, this would calculate actual settling time
        return 0.0
    
    def _calculate_overshoot(self) -> float:
        """Calculate overshoot."""
        # Placeholder implementation
        # In practice, this would calculate actual overshoot
        return 0.0
    
    def _calculate_steady_state_error(self) -> float:
        """Calculate steady state error."""
        # Placeholder implementation
        # In practice, this would calculate actual steady state error
        return 0.0
    
    def set_parameters(self, kp: float = None, ki: float = None, kd: float = None):
        """
        Set PID parameters.
        
        Args:
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
        """
        if kp is not None:
            self.kp = kp
        if ki is not None:
            self.ki = ki
        if kd is not None:
            self.kd = kd
        
        self.logger.info(f"PID parameters updated: Kp={self.kp}, Ki={self.ki}, Kd={self.kd}")
    
    def reset(self):
        """Reset controller state."""
        self.prev_error = 0.0
        self.integral = 0.0
        self.prev_time = None
        
        self.logger.info("PID controller reset")
    
    def get_parameters(self) -> Dict[str, float]:
        """Get current PID parameters."""
        return {
            "kp": self.kp,
            "ki": self.ki,
            "kd": self.kd
        }
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics."""
        if not self.performance_history:
            return {}
        
        # Calculate average performance metrics
        avg_error = np.mean([p["error"] for p in self.performance_history[-100:]])
        avg_control_output = np.mean([p["control_output"] for p in self.performance_history[-100:]])
        avg_settling_time = np.mean([p["settling_time"] for p in self.performance_history[-100:]])
        avg_overshoot = np.mean([p["overshoot"] for p in self.performance_history[-100:]])
        avg_steady_state_error = np.mean([p["steady_state_error"] for p in self.performance_history[-100:]])
        
        return {
            "average_error": avg_error,
            "average_control_output": avg_control_output,
            "average_settling_time": avg_settling_time,
            "average_overshoot": avg_overshoot,
            "average_steady_state_error": avg_steady_state_error
        }
    
    def plot_performance(self, save_path: Optional[str] = None):
        """Plot controller performance."""
        if not self.performance_history:
            self.logger.warning("No performance history to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Plot error
        errors = [p["error"] for p in self.performance_history]
        axes[0, 0].plot(errors)
        axes[0, 0].set_title("Control Error")
        axes[0, 0].set_xlabel("Time Step")
        axes[0, 0].set_ylabel("Error")
        axes[0, 0].grid(True)
        
        # Plot control output
        control_outputs = [p["control_output"] for p in self.performance_history]
        axes[0, 1].plot(control_outputs)
        axes[0, 1].set_title("Control Output")
        axes[0, 1].set_xlabel("Time Step")
        axes[0, 1].set_ylabel("Control Output")
        axes[0, 1].grid(True)
        
        # Plot reference vs actual
        if self.reference_history:
            axes[1, 0].plot(self.reference_history, label="Reference")
            # Assuming first state is output
            actual_outputs = [p["error"] + ref for p, ref in zip(self.performance_history, self.reference_history)]
            axes[1, 0].plot(actual_outputs, label="Actual")
            axes[1, 0].set_title("Reference vs Actual")
            axes[1, 0].set_xlabel("Time Step")
            axes[1, 0].set_ylabel("Value")
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Plot performance metrics
        settling_times = [p["settling_time"] for p in self.performance_history]
        overshoots = [p["overshoot"] for p in self.performance_history]
        axes[1, 1].plot(settling_times, label="Settling Time")
        axes[1, 1].plot(overshoots, label="Overshoot")
        axes[1, 1].set_title("Performance Metrics")
        axes[1, 1].set_xlabel("Time Step")
        axes[1, 1].set_ylabel("Value")
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        
        plt.close()
    
    def get_status(self) -> Dict[str, Any]:
        """Get controller status."""
        return {
            "controller_type": "PID",
            "parameters": self.get_parameters(),
            "performance_history_length": len(self.performance_history),
            "current_state": {
                "prev_error": self.prev_error,
                "integral": self.integral,
                "prev_time": self.prev_time
            }
        }
