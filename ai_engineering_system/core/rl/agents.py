"""
Reinforcement Learning agents for engineering applications.
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


class EngineeringAgent:
    """
    Base class for engineering RL agents.
    """
    
    def __init__(self, env: gym.Env, algorithm: str = "PPO"):
        """
        Initialize the engineering agent.
        
        Args:
            env: Environment
            algorithm: RL algorithm to use
        """
        self.logger = logging.getLogger(__name__)
        self.env = env
        self.algorithm = algorithm
        self.agent = None
        self.training_history = []
        
    def train(self, total_timesteps: int = 100000) -> Dict[str, Any]:
        """Train the agent."""
        if self.agent is None:
            raise ValueError("Agent not initialized")
        
        # Train agent
        self.agent.learn(total_timesteps=total_timesteps)
        
        # Get training statistics
        training_stats = {
            "total_timesteps": total_timesteps,
            "algorithm": self.algorithm,
            "final_reward": self.evaluate()
        }
        
        self.training_history.append(training_stats)
        
        return training_stats
    
    def evaluate(self, n_episodes: int = 10) -> float:
        """Evaluate the agent."""
        if self.agent is None:
            raise ValueError("Agent not initialized")
        
        total_rewards = []
        
        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            done = False
            episode_reward = 0.0
            
            while not done:
                action, _ = self.agent.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = self.env.step(action)
                episode_reward += reward
                done = done or truncated
            
            total_rewards.append(episode_reward)
        
        return np.mean(total_rewards)
    
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, Any]:
        """Predict action for given observation."""
        if self.agent is None:
            raise ValueError("Agent not initialized")
        
        return self.agent.predict(observation, deterministic=deterministic)
    
    def save(self, path: str):
        """Save the agent."""
        if self.agent is None:
            raise ValueError("Agent not initialized")
        
        self.agent.save(path)
        self.logger.info(f"Agent saved to {path}")
    
    def load(self, path: str):
        """Load the agent."""
        if self.algorithm == "PPO":
            self.agent = PPO.load(path)
        elif self.algorithm == "DQN":
            self.agent = DQN.load(path)
        elif self.algorithm == "A2C":
            self.agent = A2C.load(path)
        elif self.algorithm == "SAC":
            self.agent = SAC.load(path)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
        
        self.logger.info(f"Agent loaded from {path}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status."""
        return {
            "algorithm": self.algorithm,
            "trained": self.agent is not None,
            "training_episodes": len(self.training_history),
            "environment": str(self.env)
        }


class PPOAgent(EngineeringAgent):
    """
    Proximal Policy Optimization agent for engineering applications.
    """
    
    def __init__(self, env: gym.Env, learning_rate: float = 3e-4, batch_size: int = 64, 
                 n_epochs: int = 10, clip_range: float = 0.2):
        """
        Initialize PPO agent.
        
        Args:
            env: Environment
            learning_rate: Learning rate
            batch_size: Batch size
            n_epochs: Number of epochs per update
            clip_range: PPO clip range
        """
        super().__init__(env, "PPO")
        
        # Initialize PPO agent
        self.agent = PPO(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            batch_size=batch_size,
            n_epochs=n_epochs,
            clip_range=clip_range,
            verbose=1
        )
        
        self.logger.info("PPO agent initialized")
    
    def train(self, total_timesteps: int = 100000) -> Dict[str, Any]:
        """Train PPO agent."""
        # Train agent
        self.agent.learn(total_timesteps=total_timesteps)
        
        # Get training statistics
        training_stats = {
            "total_timesteps": total_timesteps,
            "algorithm": "PPO",
            "final_reward": self.evaluate(),
            "learning_rate": self.agent.learning_rate,
            "batch_size": self.agent.batch_size,
            "n_epochs": self.agent.n_epochs,
            "clip_range": self.agent.clip_range
        }
        
        self.training_history.append(training_stats)
        
        return training_stats


class DQNAgent(EngineeringAgent):
    """
    Deep Q-Network agent for engineering applications.
    """
    
    def __init__(self, env: gym.Env, learning_rate: float = 1e-3, buffer_size: int = 10000,
                 exploration_fraction: float = 0.1, exploration_final_eps: float = 0.02):
        """
        Initialize DQN agent.
        
        Args:
            env: Environment
            learning_rate: Learning rate
            buffer_size: Replay buffer size
            exploration_fraction: Exploration fraction
            exploration_final_eps: Final exploration epsilon
        """
        super().__init__(env, "DQN")
        
        # Initialize DQN agent
        self.agent = DQN(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            exploration_fraction=exploration_fraction,
            exploration_final_eps=exploration_final_eps,
            verbose=1
        )
        
        self.logger.info("DQN agent initialized")
    
    def train(self, total_timesteps: int = 100000) -> Dict[str, Any]:
        """Train DQN agent."""
        # Train agent
        self.agent.learn(total_timesteps=total_timesteps)
        
        # Get training statistics
        training_stats = {
            "total_timesteps": total_timesteps,
            "algorithm": "DQN",
            "final_reward": self.evaluate(),
            "learning_rate": self.agent.learning_rate,
            "buffer_size": self.agent.buffer_size,
            "exploration_fraction": self.agent.exploration_fraction,
            "exploration_final_eps": self.agent.exploration_final_eps
        }
        
        self.training_history.append(training_stats)
        
        return training_stats


class A2CAgent(EngineeringAgent):
    """
    Advantage Actor-Critic agent for engineering applications.
    """
    
    def __init__(self, env: gym.Env, learning_rate: float = 3e-4, n_steps: int = 5,
                 gamma: float = 0.99, gae_lambda: float = 1.0):
        """
        Initialize A2C agent.
        
        Args:
            env: Environment
            learning_rate: Learning rate
            n_steps: Number of steps per update
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
        """
        super().__init__(env, "A2C")
        
        # Initialize A2C agent
        self.agent = A2C(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            verbose=1
        )
        
        self.logger.info("A2C agent initialized")
    
    def train(self, total_timesteps: int = 100000) -> Dict[str, Any]:
        """Train A2C agent."""
        # Train agent
        self.agent.learn(total_timesteps=total_timesteps)
        
        # Get training statistics
        training_stats = {
            "total_timesteps": total_timesteps,
            "algorithm": "A2C",
            "final_reward": self.evaluate(),
            "learning_rate": self.agent.learning_rate,
            "n_steps": self.agent.n_steps,
            "gamma": self.agent.gamma,
            "gae_lambda": self.agent.gae_lambda
        }
        
        self.training_history.append(training_stats)
        
        return training_stats


class SACAgent(EngineeringAgent):
    """
    Soft Actor-Critic agent for engineering applications.
    """
    
    def __init__(self, env: gym.Env, learning_rate: float = 3e-4, buffer_size: int = 100000,
                 learning_starts: int = 100, batch_size: int = 256, tau: float = 0.005):
        """
        Initialize SAC agent.
        
        Args:
            env: Environment
            learning_rate: Learning rate
            buffer_size: Replay buffer size
            learning_starts: Learning starts
            batch_size: Batch size
            tau: Soft update coefficient
        """
        super().__init__(env, "SAC")
        
        # Initialize SAC agent
        self.agent = SAC(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            verbose=1
        )
        
        self.logger.info("SAC agent initialized")
    
    def train(self, total_timesteps: int = 100000) -> Dict[str, Any]:
        """Train SAC agent."""
        # Train agent
        self.agent.learn(total_timesteps=total_timesteps)
        
        # Get training statistics
        training_stats = {
            "total_timesteps": total_timesteps,
            "algorithm": "SAC",
            "final_reward": self.evaluate(),
            "learning_rate": self.agent.learning_rate,
            "buffer_size": self.agent.buffer_size,
            "learning_starts": self.agent.learning_starts,
            "batch_size": self.agent.batch_size,
            "tau": self.agent.tau
        }
        
        self.training_history.append(training_stats)
        
        return training_stats


class EngineeringFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for engineering applications.
    """
    
    def __init__(self, observation_space: gym.Space, features_dim: int = 64):
        """
        Initialize engineering feature extractor.
        
        Args:
            observation_space: Observation space
            features_dim: Feature dimension
        """
        super().__init__(observation_space, features_dim)
        
        # Define feature extraction network
        self.feature_net = nn.Sequential(
            nn.Linear(observation_space.shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Extract features from observations."""
        return self.feature_net(observations)


class EngineeringPolicy(BasePolicy):
    """
    Custom policy for engineering applications.
    """
    
    def __init__(self, observation_space: gym.Space, action_space: gym.Space, 
                 lr_schedule, net_arch: List[int] = None, activation_fn: nn.Module = None):
        """
        Initialize engineering policy.
        
        Args:
            observation_space: Observation space
            action_space: Action space
            lr_schedule: Learning rate schedule
            net_arch: Network architecture
            activation_fn: Activation function
        """
        super().__init__(observation_space, action_space, lr_schedule)
        
        # Set default network architecture
        if net_arch is None:
            net_arch = [64, 64]
        
        # Set default activation function
        if activation_fn is None:
            activation_fn = nn.ReLU
        
        # Build network
        self.net_arch = net_arch
        self.activation_fn = activation_fn
        
        # Create feature extractor
        self.features_extractor = EngineeringFeatureExtractor(observation_space)
        
        # Create policy network
        self.policy_net = self._build_policy_net()
        
        # Create value network
        self.value_net = self._build_value_net()
    
    def _build_policy_net(self) -> nn.Module:
        """Build policy network."""
        layers = []
        input_dim = self.features_extractor.features_dim
        
        for i, hidden_dim in enumerate(self.net_arch):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(self.activation_fn())
            input_dim = hidden_dim
        
        # Output layer
        if isinstance(self.action_space, gym.spaces.Box):
            # Continuous action space
            layers.append(nn.Linear(input_dim, self.action_space.shape[0]))
        else:
            # Discrete action space
            layers.append(nn.Linear(input_dim, self.action_space.n))
        
        return nn.Sequential(*layers)
    
    def _build_value_net(self) -> nn.Module:
        """Build value network."""
        layers = []
        input_dim = self.features_extractor.features_dim
        
        for i, hidden_dim in enumerate(self.net_arch):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(self.activation_fn())
            input_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(input_dim, 1))
        
        return nn.Sequential(*layers)
    
    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through policy."""
        # Extract features
        features = self.features_extractor(obs)
        
        # Get policy output
        policy_output = self.policy_net(features)
        
        # Get value output
        value = self.value_net(features)
        
        # Get action
        if isinstance(self.action_space, gym.spaces.Box):
            # Continuous action space
            action = policy_output
            if not deterministic:
                # Add noise for exploration
                noise = torch.randn_like(action) * 0.1
                action = action + noise
            action = torch.clamp(action, self.action_space.low[0], self.action_space.high[0])
        else:
            # Discrete action space
            if deterministic:
                action = torch.argmax(policy_output, dim=-1)
            else:
                action = torch.multinomial(torch.softmax(policy_output, dim=-1), 1).squeeze(-1)
        
        # Get log probability
        if isinstance(self.action_space, gym.spaces.Box):
            # Continuous action space - use normal distribution
            log_prob = -0.5 * ((action - policy_output) ** 2).sum(dim=-1)
        else:
            # Discrete action space
            log_prob = torch.log_softmax(policy_output, dim=-1)[torch.arange(action.size(0)), action]
        
        return action, value, log_prob
    
    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for given observations."""
        # Extract features
        features = self.features_extractor(obs)
        
        # Get policy output
        policy_output = self.policy_net(features)
        
        # Get value output
        value = self.value_net(features)
        
        # Get log probability
        if isinstance(self.action_space, gym.spaces.Box):
            # Continuous action space
            log_prob = -0.5 * ((actions - policy_output) ** 2).sum(dim=-1)
        else:
            # Discrete action space
            log_prob = torch.log_softmax(policy_output, dim=-1)[torch.arange(actions.size(0)), actions]
        
        return value, log_prob, torch.zeros_like(log_prob)  # entropy placeholder


class MultiObjectiveAgent(EngineeringAgent):
    """
    Multi-objective RL agent for engineering applications.
    """
    
    def __init__(self, env: gym.Env, objectives: List[str], weights: List[float] = None):
        """
        Initialize multi-objective agent.
        
        Args:
            env: Environment
            objectives: List of objectives
            weights: Weights for objectives
        """
        super().__init__(env, "MultiObjective")
        
        self.objectives = objectives
        self.weights = weights or [1.0] * len(objectives)
        
        # Initialize multiple agents for different objectives
        self.objective_agents = {}
        
        for i, objective in enumerate(objectives):
            weight = self.weights[i]
            agent = PPOAgent(env)
            self.objective_agents[objective] = {
                "agent": agent,
                "weight": weight
            }
        
        self.logger.info(f"Multi-objective agent initialized with {len(objectives)} objectives")
    
    def train(self, total_timesteps: int = 100000) -> Dict[str, Any]:
        """Train multi-objective agent."""
        training_results = {}
        
        # Train each objective agent
        for objective, agent_info in self.objective_agents.items():
            agent = agent_info["agent"]
            weight = agent_info["weight"]
            
            # Train agent
            result = agent.train(total_timesteps // len(self.objectives))
            training_results[objective] = result
        
        # Combine results
        combined_result = {
            "total_timesteps": total_timesteps,
            "algorithm": "MultiObjective",
            "objectives": self.objectives,
            "weights": self.weights,
            "objective_results": training_results,
            "final_reward": self.evaluate()
        }
        
        self.training_history.append(combined_result)
        
        return combined_result
    
    def evaluate(self, n_episodes: int = 10) -> Dict[str, float]:
        """Evaluate multi-objective agent."""
        evaluation_results = {}
        
        # Evaluate each objective agent
        for objective, agent_info in self.objective_agents.items():
            agent = agent_info["agent"]
            weight = agent_info["weight"]
            
            reward = agent.evaluate(n_episodes)
            evaluation_results[objective] = reward * weight
        
        return evaluation_results
    
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, Any]:
        """Predict action using weighted combination of objective agents."""
        actions = []
        weights = []
        
        # Get actions from each objective agent
        for objective, agent_info in self.objective_agents.items():
            agent = agent_info["agent"]
            weight = agent_info["weight"]
            
            action, _ = agent.predict(observation, deterministic)
            actions.append(action)
            weights.append(weight)
        
        # Weighted combination of actions
        actions = np.array(actions)
        weights = np.array(weights)
        weights = weights / np.sum(weights)  # Normalize weights
        
        combined_action = np.average(actions, axis=0, weights=weights)
        
        return combined_action, None
    
    def get_status(self) -> Dict[str, Any]:
        """Get multi-objective agent status."""
        base_status = super().get_status()
        base_status.update({
            "objectives": self.objectives,
            "weights": self.weights,
            "objective_agents": len(self.objective_agents)
        })
        return base_status
