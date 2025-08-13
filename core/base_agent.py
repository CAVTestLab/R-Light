from abc import ABC, abstractmethod
import numpy as np
from typing import Any, Tuple, Optional

class BaseAgent(ABC):
    """Abstract base class for agents"""
    
    def __init__(self):
        self.stats = {
            'rewards': [],
            'queue_lengths': []
        }
    
    @abstractmethod
    def select_action(self, state: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Select action based on current state"""
        pass
    
    @abstractmethod
    def remember(self, state: np.ndarray, action: np.ndarray, 
                reward: Any, next_state: np.ndarray) -> None:
        """Store experience in replay buffer"""
        pass
    
    @abstractmethod
    def train(self, *args, **kwargs) -> None:
        """Train the neural network"""
        pass
    
    @abstractmethod
    def update_epsilon(self, episode: int, total_episodes: int) -> float:
        """Update exploration rate"""
        pass
    
    @abstractmethod
    def save_model(self, path: str) -> None:
        """Save model to file"""
        pass
    
    def record_stats(self, reward: float, queue_length: float) -> None:
        """Record training statistics"""
        self.stats['rewards'].append(reward)
        self.stats['queue_lengths'].append(queue_length)
