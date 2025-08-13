import random
import torch
import torch.nn as nn
import numpy as np

from agent.base.base_dqn_agent import BaseDQNAgent

class D3QNAgent(BaseDQNAgent):
    """Double Deep Dueling Q-Network agent"""
    
    def __init__(self):
        super().__init__(network_type='dueling_mlp', hidden_dims=(512, 256))
    
    def _get_loss_function(self) -> nn.Module:
        """Use MSE loss for D3QN"""
        return nn.MSELoss()
    
    def select_action(self, state: np.ndarray, num_nodes: int, training: bool = True, **kwargs) -> np.ndarray:
        """Select action using dueling network"""
        if training and np.random.random() < self.epsilon:
            # Random exploration
            return np.random.randint(0, self.action_dim, size=num_nodes).reshape(1, num_nodes, 1)
        else:
            # Exploitation using Q-network
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
                q_values = self.q_net(state_tensor)
                actions = torch.argmax(q_values, dim=-1)
                return actions.cpu().numpy().reshape(1, num_nodes, 1)
    
    def _compute_q_values(self, state: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute Q-values using dueling network"""
        return self.q_net(state)
    
    def _compute_target_q_values(self, next_states: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute target Q-values using Double DQN with dueling architecture"""
        # Double DQN: use main network to select actions, target network to evaluate
        next_actions = torch.argmax(self.q_net(next_states), dim=-1, keepdim=True)
        next_q_values = self.target_q_net(next_states).gather(2, next_actions)
        return next_q_values
    def sample(self, batch_size):
        samples = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states = zip(*samples)
        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(np.array(actions), dtype=torch.long),
            torch.tensor(np.array(rewards), dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)
