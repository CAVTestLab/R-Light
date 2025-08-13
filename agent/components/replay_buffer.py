import torch
import numpy as np
import random
from collections import deque
from typing import Tuple, List, Any

class ReplayBuffer:
    """Experience replay buffer for reinforcement learning"""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state: np.ndarray, action: np.ndarray, reward: Any, next_state: np.ndarray) -> None:
        """Add experience to buffer"""
        self.buffer.append((state, action, reward, next_state))
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample batch of experiences"""
        samples = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states = zip(*samples)
        
        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(np.array(actions), dtype=torch.long),
            torch.tensor(np.array(rewards), dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
        )
    
    def __len__(self) -> int:
        return len(self.buffer)
