import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
from abc import ABC, abstractmethod
from typing import Optional, List, Any, Dict

from agent.components.neural_networks import BaseNetwork, NetworkFactory
from agent.components.replay_buffer import ReplayBuffer
from config import AGENT_CONFIG

class BaseDQNAgent(ABC):
    """Abstract base class for DQN-based agents"""
    
    def __init__(self, network_type: str, **network_kwargs):
        self.state_dim = AGENT_CONFIG['state_dim']
        self.action_dim = AGENT_CONFIG['action_dim']
        self.gamma = AGENT_CONFIG['gamma']
        self.initial_epsilon = AGENT_CONFIG['epsilon_start']
        self.epsilon_end = AGENT_CONFIG['epsilon_end']
        self.epsilon = self.initial_epsilon
        self.batch_size = AGENT_CONFIG['batch_size']
        self.device = AGENT_CONFIG['device']
        self.tau = AGENT_CONFIG['tau']
        
        # Create networks
        self.q_net = self._create_network(network_type, **network_kwargs)
        self.target_q_net = self._create_network(network_type, **network_kwargs)
        self._sync_target_network()
        self.target_q_net.eval()
        
        # Initialize optimizer and loss function
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=AGENT_CONFIG['learning_rate'])
        self.criterion = self._get_loss_function()
        
        # Initialize replay buffer
        self.memory = ReplayBuffer(AGENT_CONFIG['buffer_size'])
        
        # Statistics tracking
        self.reward_history = []
        self.queue_length_history = []
    
    def _create_network(self, network_type: str, **kwargs) -> BaseNetwork:
        """Create neural network instance"""
        network = NetworkFactory.create_network(
            network_type, self.state_dim, self.action_dim, **kwargs
        )
        return network.to(self.device)
    
    def _get_loss_function(self) -> nn.Module:
        """Get loss function for training"""
        return nn.HuberLoss()
    
    def _sync_target_network(self) -> None:
        """Synchronize target network with main network"""
        self.target_q_net.load_state_dict(self.q_net.state_dict())
    
    def update_epsilon(self, episode: int, total_episodes: int) -> float:
        """Update exploration rate"""
        self.epsilon = self.initial_epsilon - (
            self.initial_epsilon - self.epsilon_end
        ) * (episode / total_episodes)
        return max(self.epsilon, self.epsilon_end)
    
    @abstractmethod
    def select_action(self, state: np.ndarray, *args, training: bool = True, **kwargs) -> np.ndarray:
        """Select action using agent-specific logic"""
        pass
    
    @abstractmethod
    def _compute_q_values(self, state: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Compute Q-values using network-specific forward pass"""
        pass
    
    @abstractmethod
    def _compute_target_q_values(self, next_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Compute target Q-values for training"""
        pass
    
    def remember(self, state: np.ndarray, action: np.ndarray, reward: Any, next_state: np.ndarray) -> None:
        """Store experience in replay buffer"""
        # Normalize shapes
        state = np.squeeze(state, axis=0) if state.shape[0] == 1 else state
        next_state = np.squeeze(next_state, axis=0) if next_state.shape[0] == 1 else next_state
        action = np.squeeze(action, axis=0) if action.shape[0] == 1 else action
        
        # Handle reward normalization
        if isinstance(reward, (list, np.ndarray)):
            reward = np.array(reward)
            if reward.ndim == 2 and reward.shape[0] == 1:
                reward = np.squeeze(reward, axis=0)
        
        self.memory.add(state, action, reward, next_state)
    
    def train(self, *args, **kwargs) -> None:
        """Train the agent"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states = self.memory.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.unsqueeze(-1).to(self.device)
        next_states = next_states.to(self.device)
        
        # Compute current Q-values
        current_q_values = self._compute_q_values(states, *args, **kwargs).gather(2, actions)
        
        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self._compute_target_q_values(next_states, *args, **kwargs)
            target_q_values = rewards + self.gamma * next_q_values
        
        # Compute loss and optimize
        loss = self.criterion(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Soft update target network
        self._soft_update_target_network()
    
    def _soft_update_target_network(self) -> None:
        """Soft update target network parameters"""
        for target_param, param in zip(self.target_q_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
    
    def save_model(self, path: str) -> None:
        """Save model checkpoint"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'q_net_state_dict': self.q_net.state_dict(),
            'target_q_net_state_dict': self.target_q_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
        }, path)
    
    def load_model(self, path: str) -> None:
        """Load model checkpoint"""
        if os.path.exists(path):
            checkpoint = torch.load(path)
            self.q_net.load_state_dict(checkpoint['q_net_state_dict'])
            self.target_q_net.load_state_dict(checkpoint['target_q_net_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
    
    def record_stats(self, episode_reward: float, episode_queue_length: float) -> None:
        """Record episode statistics"""
        self.reward_history.append(episode_reward)
        self.queue_length_history.append(episode_queue_length)
    
    def get_stats(self) -> tuple:
        """Get training statistics"""
        return self.reward_history, self.queue_length_history
