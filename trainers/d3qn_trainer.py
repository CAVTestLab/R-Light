import numpy as np
from typing import Dict, Any
from core.base_trainer import BaseTrainer
from core.base_agent import BaseAgent
from agent.d3qn_agent import D3QNAgent

class D3QNTrainer(BaseTrainer):
    """D3QN algorithm trainer"""
    
    def create_agent(self) -> BaseAgent:
        """Create D3QN agent instance"""
        return D3QNAgent()
    
    def select_action_for_agent(self, state: np.ndarray, step: int) -> np.ndarray:
        """Select action using D3QN specific parameters"""
        action = self.agent.select_action(
            state[np.newaxis],
            self.env.num_nodes,
            training=True
        )
        return action.squeeze(0)
    
    def train_agent(self, step: int) -> None:
        """Train D3QN agent"""
        self.agent.train()
