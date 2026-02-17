"""
Curiosity-Driven Intrinsic Rewards: RND and ICM for MATWM

Implements two classic curiosity algorithms for multi-agent reinforcement learning:
1. RND (Random Network Distillation) - Burda et al. 2018
2. ICM (Intrinsic Curiosity Module) - Pathak et al. 2017

These serve as baselines to compare against Social Curiosity (MATWM's TeammatePredictor-based approach).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional
import os


@dataclass
class RNDConfig:
    """Configuration for RND (Random Network Distillation)"""
    # Network architecture
    obs_dim: int = 16
    hidden_dim: int = 256
    feature_dim: int = 128
    num_hidden_layers: int = 2

    # Training
    learning_rate: float = 1e-4
    weight: float = 1.0  # Intrinsic reward weight

    # Normalization
    normalize_observations: bool = True
    normalize_rewards: bool = True
    reward_clip: float = 5.0

    # Running statistics
    obs_mean_decay: float = 0.999
    obs_std_decay: float = 0.999
    reward_std_decay: float = 0.999


@dataclass
class ICMConfig:
    """Configuration for ICM (Intrinsic Curiosity Module)"""
    # Network architecture
    obs_dim: int = 16
    action_dim: int = 5
    hidden_dim: int = 256
    feature_dim: int = 128
    num_hidden_layers: int = 2

    # Training
    learning_rate: float = 1e-4
    forward_weight: float = 0.2  # Weight of forward loss in total ICM loss
    inverse_weight: float = 0.8  # Weight of inverse loss in total ICM loss
    intrinsic_reward_weight: float = 1.0  # Scale of intrinsic reward

    # Normalization
    normalize_observations: bool = True
    normalize_rewards: bool = True
    reward_clip: float = 5.0

    # Running statistics
    obs_mean_decay: float = 0.999
    obs_std_decay: float = 0.999
    reward_std_decay: float = 0.999


class MLP(nn.Module):
    """Simple multi-layer perceptron"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, activation='relu'):
        super().__init__()
        self.activation_fn = nn.ReLU() if activation == 'relu' else nn.Tanh()

        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(self.activation_fn)
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class RNDModule(nn.Module):
    """
    Random Network Distillation (Burda et al. 2018)

    Key idea: Train a network to predict the output of a fixed random network.
    The prediction error serves as an exploration bonus (novelty measure).

    Components:
    - Target network: Randomly initialized, fixed (never trained)
    - Predictor network: Trained to match target network output

    Intrinsic reward: ||predictor(s) - target(s)||^2
    """
    def __init__(self, config: RNDConfig):
        super().__init__()
        self.config = config

        # Target network (fixed random network)
        self.target_net = MLP(
            config.obs_dim,
            config.hidden_dim,
            config.feature_dim,
            config.num_hidden_layers
        )
        # Freeze target network
        for param in self.target_net.parameters():
            param.requires_grad = False

        # Predictor network (trainable)
        self.predictor_net = MLP(
            config.obs_dim,
            config.hidden_dim,
            config.feature_dim,
            config.num_hidden_layers
        )

        # Running statistics for observation normalization
        self.register_buffer('obs_mean', torch.zeros(config.obs_dim))
        self.register_buffer('obs_std', torch.ones(config.obs_dim))
        self.register_buffer('reward_std', torch.ones(1))

        # Optimizer
        self.optimizer = torch.optim.Adam(self.predictor_net.parameters(), lr=config.learning_rate)

    def normalize_obs(self, obs):
        """Normalize observations using running statistics"""
        if self.config.normalize_observations:
            return (obs - self.obs_mean) / (self.obs_std + 1e-8)
        return obs

    def update_obs_stats(self, obs):
        """Update running mean and std of observations"""
        if self.config.normalize_observations:
            batch_mean = obs.mean(dim=0)
            batch_std = obs.std(dim=0)

            self.obs_mean = self.config.obs_mean_decay * self.obs_mean + \
                           (1 - self.config.obs_mean_decay) * batch_mean
            self.obs_std = self.config.obs_std_decay * self.obs_std + \
                          (1 - self.config.obs_std_decay) * batch_std

    def compute_intrinsic_reward(self, obs):
        """
        Compute RND intrinsic reward (prediction error).

        Args:
            obs: Observation tensor (batch_size, obs_dim) or (obs_dim,)

        Returns:
            intrinsic_reward: Scalar or batch of scalars
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        obs_norm = self.normalize_obs(obs)

        with torch.no_grad():
            target_features = self.target_net(obs_norm)

        predictor_features = self.predictor_net(obs_norm)

        # Prediction error as intrinsic reward
        intrinsic_reward = F.mse_loss(predictor_features, target_features, reduction='none').mean(dim=1)

        # Normalize reward
        if self.config.normalize_rewards:
            intrinsic_reward = intrinsic_reward / (self.reward_std + 1e-8)
            if self.config.reward_clip > 0:
                intrinsic_reward = torch.clamp(intrinsic_reward, -self.config.reward_clip, self.config.reward_clip)

        return intrinsic_reward.squeeze() * self.config.weight

    def train_step(self, obs_batch):
        """
        Train predictor network to match target network.

        Args:
            obs_batch: Batch of observations (batch_size, obs_dim)

        Returns:
            loss: Training loss value
        """
        self.update_obs_stats(obs_batch)

        obs_norm = self.normalize_obs(obs_batch)

        with torch.no_grad():
            target_features = self.target_net(obs_norm)

        predictor_features = self.predictor_net(obs_norm)

        loss = F.mse_loss(predictor_features, target_features)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update reward std
        if self.config.normalize_rewards:
            with torch.no_grad():
                pred_error = F.mse_loss(predictor_features, target_features, reduction='none').mean(dim=1)
                batch_reward_std = pred_error.std()
                self.reward_std = self.config.reward_std_decay * self.reward_std + \
                                 (1 - self.config.reward_std_decay) * batch_reward_std

        return loss.item()


class ICMModule(nn.Module):
    """
    Intrinsic Curiosity Module (Pathak et al. 2017)

    Key idea: Learn a feature representation that captures information relevant for prediction.
    Use prediction error in this learned feature space as exploration bonus.

    Components:
    - Feature network: Encodes observations to feature space
    - Forward model: Predicts next state features from current features + action
    - Inverse model: Predicts action from current and next state features

    Intrinsic reward: ||forward(φ(s), a) - φ(s')||^2
    where φ is the feature network
    """
    def __init__(self, config: ICMConfig):
        super().__init__()
        self.config = config

        # Feature network (encoder)
        self.feature_net = MLP(
            config.obs_dim,
            config.hidden_dim,
            config.feature_dim,
            config.num_hidden_layers
        )

        # Inverse model: (φ(s), φ(s')) -> a
        self.inverse_model = MLP(
            config.feature_dim * 2,
            config.hidden_dim,
            config.action_dim,
            config.num_hidden_layers
        )

        # Forward model: (φ(s), a) -> φ(s')
        self.forward_model = MLP(
            config.feature_dim + config.action_dim,
            config.hidden_dim,
            config.feature_dim,
            config.num_hidden_layers
        )

        # Running statistics
        self.register_buffer('obs_mean', torch.zeros(config.obs_dim))
        self.register_buffer('obs_std', torch.ones(config.obs_dim))
        self.register_buffer('reward_std', torch.ones(1))

        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.learning_rate)

    def normalize_obs(self, obs):
        """Normalize observations using running statistics"""
        if self.config.normalize_observations:
            return (obs - self.obs_mean) / (self.obs_std + 1e-8)
        return obs

    def update_obs_stats(self, obs):
        """Update running mean and std of observations"""
        if self.config.normalize_observations:
            batch_mean = obs.mean(dim=0)
            batch_std = obs.std(dim=0)

            self.obs_mean = self.config.obs_mean_decay * self.obs_mean + \
                           (1 - self.config.obs_mean_decay) * batch_mean
            self.obs_std = self.config.obs_std_decay * self.obs_std + \
                          (1 - self.config.obs_std_decay) * batch_std

    def compute_intrinsic_reward(self, obs, action, next_obs):
        """
        Compute ICM intrinsic reward (forward model prediction error).

        Args:
            obs: Current observation (batch_size, obs_dim) or (obs_dim,)
            action: Action taken (batch_size,) or scalar
            next_obs: Next observation (batch_size, obs_dim) or (obs_dim,)

        Returns:
            intrinsic_reward: Scalar or batch of scalars
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
            next_obs = next_obs.unsqueeze(0)
            action = torch.tensor([action], dtype=torch.long, device=obs.device)

        obs_norm = self.normalize_obs(obs)
        next_obs_norm = self.normalize_obs(next_obs)

        # Encode observations
        with torch.no_grad():
            features = self.feature_net(obs_norm)
            next_features = self.feature_net(next_obs_norm)

        # One-hot encode action
        action_onehot = F.one_hot(action, num_classes=self.config.action_dim).float()

        # Forward model prediction
        forward_input = torch.cat([features, action_onehot], dim=-1)
        predicted_next_features = self.forward_model(forward_input)

        # Prediction error as intrinsic reward
        intrinsic_reward = F.mse_loss(predicted_next_features, next_features, reduction='none').mean(dim=1)

        # Normalize reward
        if self.config.normalize_rewards:
            intrinsic_reward = intrinsic_reward / (self.reward_std + 1e-8)
            if self.config.reward_clip > 0:
                intrinsic_reward = torch.clamp(intrinsic_reward, -self.config.reward_clip, self.config.reward_clip)

        return intrinsic_reward.squeeze() * self.config.intrinsic_reward_weight

    def train_step(self, obs_batch, action_batch, next_obs_batch):
        """
        Train ICM (feature net, forward model, inverse model).

        Args:
            obs_batch: Batch of observations (batch_size, obs_dim)
            action_batch: Batch of actions (batch_size,)
            next_obs_batch: Batch of next observations (batch_size, obs_dim)

        Returns:
            loss_dict: Dictionary with forward_loss, inverse_loss, total_loss
        """
        self.update_obs_stats(torch.cat([obs_batch, next_obs_batch], dim=0))

        obs_norm = self.normalize_obs(obs_batch)
        next_obs_norm = self.normalize_obs(next_obs_batch)

        # Encode observations
        features = self.feature_net(obs_norm)
        next_features = self.feature_net(next_obs_norm)

        # One-hot encode actions
        action_onehot = F.one_hot(action_batch, num_classes=self.config.action_dim).float()

        # Forward model loss
        forward_input = torch.cat([features, action_onehot], dim=-1)
        predicted_next_features = self.forward_model(forward_input)
        forward_loss = F.mse_loss(predicted_next_features, next_features.detach())

        # Inverse model loss
        inverse_input = torch.cat([features, next_features], dim=-1)
        predicted_action_logits = self.inverse_model(inverse_input)
        inverse_loss = F.cross_entropy(predicted_action_logits, action_batch)

        # Total loss
        total_loss = self.config.forward_weight * forward_loss + \
                     self.config.inverse_weight * inverse_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # Update reward std
        if self.config.normalize_rewards:
            with torch.no_grad():
                pred_error = F.mse_loss(predicted_next_features, next_features, reduction='none').mean(dim=1)
                batch_reward_std = pred_error.std()
                self.reward_std = self.config.reward_std_decay * self.reward_std + \
                                 (1 - self.config.reward_std_decay) * batch_reward_std

        return {
            'forward_loss': forward_loss.item(),
            'inverse_loss': inverse_loss.item(),
            'total_loss': total_loss.item(),
        }


class RNDCuriosityManager:
    """Manager for RND-based curiosity rewards in multi-agent setting"""
    def __init__(self, agent_name: str, config: RNDConfig, device: torch.device):
        self.agent_name = agent_name
        self.config = config
        self.device = device

        self.rnd = RNDModule(config).to(device)

        # Statistics
        self.episode_rewards = []
        self.total_intrinsic_reward = 0.0
        self.step_count = 0

    def reset_episode(self, episode_id: int):
        """Reset episode statistics"""
        self.total_intrinsic_reward = 0.0

    def compute_intrinsic_reward(self, obs):
        """
        Compute RND intrinsic reward for given observation.

        Args:
            obs: Observation array (numpy or torch tensor)

        Returns:
            intrinsic_reward: Scalar reward value
        """
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float().to(self.device)

        with torch.no_grad():
            reward = self.rnd.compute_intrinsic_reward(obs)

        reward_val = reward.item() if torch.is_tensor(reward) else reward
        self.total_intrinsic_reward += reward_val
        self.step_count += 1

        return reward_val

    def train(self, obs_batch):
        """Train RND predictor network"""
        if isinstance(obs_batch, np.ndarray):
            obs_batch = torch.from_numpy(obs_batch).float().to(self.device)

        loss = self.rnd.train_step(obs_batch)
        return {'rnd_loss': loss}

    def end_episode(self):
        """Called at episode end"""
        self.episode_rewards.append(self.total_intrinsic_reward)
        return {}

    def get_episode_summary(self):
        """Get episode statistics"""
        return {
            'total_intrinsic_reward': self.total_intrinsic_reward,
            'avg_intrinsic_reward_per_step': self.total_intrinsic_reward / max(1, self.step_count),
        }

    def print_stats(self):
        """Print curiosity statistics"""
        if self.episode_rewards:
            print(f'{self.agent_name} RND Stats:')
            print(f'  Episodes: {len(self.episode_rewards)}')
            print(f'  Mean intrinsic reward: {np.mean(self.episode_rewards):.2f}')
            print(f'  Std intrinsic reward: {np.std(self.episode_rewards):.2f}')


class ICMCuriosityManager:
    """Manager for ICM-based curiosity rewards in multi-agent setting"""
    def __init__(self, agent_name: str, config: ICMConfig, device: torch.device):
        self.agent_name = agent_name
        self.config = config
        self.device = device

        self.icm = ICMModule(config).to(device)

        # Statistics
        self.episode_rewards = []
        self.total_intrinsic_reward = 0.0
        self.step_count = 0

    def reset_episode(self, episode_id: int):
        """Reset episode statistics"""
        self.total_intrinsic_reward = 0.0

    def compute_intrinsic_reward(self, obs, action, next_obs):
        """
        Compute ICM intrinsic reward.

        Args:
            obs: Current observation (numpy or torch)
            action: Action taken (int)
            next_obs: Next observation (numpy or torch)

        Returns:
            intrinsic_reward: Scalar reward value
        """
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float().to(self.device)
        if isinstance(next_obs, np.ndarray):
            next_obs = torch.from_numpy(next_obs).float().to(self.device)
        if not torch.is_tensor(action):
            action = torch.tensor([action], dtype=torch.long, device=self.device)

        with torch.no_grad():
            reward = self.icm.compute_intrinsic_reward(obs, action, next_obs)

        reward_val = reward.item() if torch.is_tensor(reward) else reward
        self.total_intrinsic_reward += reward_val
        self.step_count += 1

        return reward_val

    def train(self, obs_batch, action_batch, next_obs_batch):
        """Train ICM networks"""
        if isinstance(obs_batch, np.ndarray):
            obs_batch = torch.from_numpy(obs_batch).float().to(self.device)
        if isinstance(next_obs_batch, np.ndarray):
            next_obs_batch = torch.from_numpy(next_obs_batch).float().to(self.device)
        if isinstance(action_batch, np.ndarray):
            action_batch = torch.from_numpy(action_batch).long().to(self.device)

        loss_dict = self.icm.train_step(obs_batch, action_batch, next_obs_batch)
        return loss_dict

    def end_episode(self):
        """Called at episode end"""
        self.episode_rewards.append(self.total_intrinsic_reward)
        return {}

    def get_episode_summary(self):
        """Get episode statistics"""
        return {
            'total_intrinsic_reward': self.total_intrinsic_reward,
            'avg_intrinsic_reward_per_step': self.total_intrinsic_reward / max(1, self.step_count),
        }

    def print_stats(self):
        """Print curiosity statistics"""
        if self.episode_rewards:
            print(f'{self.agent_name} ICM Stats:')
            print(f'  Episodes: {len(self.episode_rewards)}')
            print(f'  Mean intrinsic reward: {np.mean(self.episode_rewards):.2f}')
            print(f'  Std intrinsic reward: {np.std(self.episode_rewards):.2f}')


def create_rnd_managers(agent_names, config: RNDConfig, device):
    """Create RND curiosity managers for all agents"""
    managers = {}
    for name in agent_names:
        managers[name] = RNDCuriosityManager(name, config, device)
    return managers


def create_icm_managers(agent_names, config: ICMConfig, device):
    """Create ICM curiosity managers for all agents"""
    managers = {}
    for name in agent_names:
        managers[name] = ICMCuriosityManager(name, config, device)
    return managers


print('RND and ICM curiosity modules loaded successfully!')
print('Available classes:')
print('  - RNDModule: Random Network Distillation')
print('  - ICMModule: Intrinsic Curiosity Module')
print('  - RNDCuriosityManager: Manager for RND in multi-agent setting')
print('  - ICMCuriosityManager: Manager for ICM in multi-agent setting')
