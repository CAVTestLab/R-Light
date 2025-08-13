import torch
import numpy as np
from typing import Optional

from agent.base.base_dqn_agent import BaseDQNAgent

class DQNAgent(BaseDQNAgent):
    """DQN agent with Graph Convolutional Network"""
    
    def __init__(self):
        super().__init__(network_type='gcn', hidden_dims=(128, 128))
    
    def select_action(self, state: np.ndarray, adj: np.ndarray, num_nodes: int, 
                     training: bool = True, **kwargs) -> np.ndarray:
        """Select action using GCN-based Q-network"""
        if training and np.random.random() < self.epsilon:
            # Random exploration
            return np.random.randint(0, self.action_dim, size=num_nodes).reshape(1, num_nodes, 1)
        else:
            # Exploitation using Q-network
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
                adj_tensor = torch.tensor(adj, dtype=torch.float32).to(self.device)
                q_values = self.q_net(state_tensor, adj_tensor)
                actions = torch.argmax(q_values, dim=2)
                return actions.cpu().numpy().reshape(1, num_nodes, 1)
    
    def _compute_q_values(self, state: torch.Tensor, adj: np.ndarray, **kwargs) -> torch.Tensor:
        """Compute Q-values using GCN forward pass"""
        adj_tensor = torch.tensor(adj, dtype=torch.float32).to(self.device)
        return self.q_net(state, adj_tensor)
    
    def _compute_target_q_values(self, next_states: torch.Tensor, adj: np.ndarray, **kwargs) -> torch.Tensor:
        """Compute target Q-values for GCN"""
        adj_tensor = torch.tensor(adj, dtype=torch.float32).to(self.device)
        next_q_values = self.target_q_net(next_states, adj_tensor).max(2).values
        return torch.unsqueeze(next_q_values, dim=-1)