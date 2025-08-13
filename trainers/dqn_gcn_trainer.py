import numpy as np
from typing import Dict, Any
from core.base_trainer import BaseTrainer
from core.base_agent import BaseAgent
from agent.dqn_gcn_agent import DQNAgent

class DQNGCNTrainer(BaseTrainer):
    """DQN_GCN algorithm trainer"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.adj_matrix = None
    
    def create_agent(self) -> BaseAgent:
        """Create DQN_GCN agent instance"""
        return DQNAgent()
    
    def select_action_for_agent(self, state: np.ndarray, step: int) -> np.ndarray:
        """Select action using DQN_GCN specific parameters"""
        if self.adj_matrix is None:
            self.adj_matrix = self.env.get_adj_matrix()
        
        action = self.agent.select_action(
            state[np.newaxis],
            self.adj_matrix,
            self.env.num_nodes,
            training=True
        )
        return action.squeeze(0)
    
    def train_agent(self, step: int) -> None:
        """Train DQN_GCN agent with adjacency matrix"""
        self.agent.train(self.adj_matrix)
