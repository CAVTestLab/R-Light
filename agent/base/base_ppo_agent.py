import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional

class PPOMemoryBuffer:
    """Memory buffer for storing trajectories"""
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.phase_probs = []
        self.adjust_probs = []
        self.vals = []
        self.rewards = []
        self.dones = []

    def store_transition(self, state, action, log_probs, val, reward, done):
        phase_log_prob, adjust_log_prob = log_probs
        self.states.append(state)
        self.actions.append(action)
        self.phase_probs.append(phase_log_prob)
        self.adjust_probs.append(adjust_log_prob)
        self.vals.append(val)       
        self.rewards.append(reward)  
        self.dones.append(done)
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.phase_probs.clear()
        self.adjust_probs.clear()
        self.vals.clear()
        self.rewards.clear()
        self.dones.clear()
    
    def get_size(self):
        return len(self.states)
    
    def get_batch_data(self):
        return {
            'states': np.array(self.states),
            'actions': np.array(self.actions),
            'phase_probs': np.array(self.phase_probs),
            'adjust_probs': np.array(self.adjust_probs),
            'vals': self.vals,
            'rewards': self.rewards,
            'dones': self.dones
        }

class BasePPOAgent(ABC):
    """Abstract base class for PPO agents"""
    
    def __init__(self, state_dim: int, phase_action_dim: int, adjust_action_dim: int, 
                 num_nodes: int, config: dict):
        self.gamma = config['gamma']
        self.gae_lambda = config['gae_lambda']
        self.clip_epsilon = config['clip_epsilon']
        self.epochs = config['epochs']
        self.batch_size = config['batch_size']
        self.device = config['device']
        self.num_nodes = num_nodes
        
        self._build_networks(state_dim, phase_action_dim, adjust_action_dim, num_nodes)
        self._build_optimizer(config['lr'])
        
        self.memory = PPOMemoryBuffer()
        self.mse_loss = nn.MSELoss()
    
    @abstractmethod
    def _build_networks(self, state_dim: int, phase_action_dim: int, 
                       adjust_action_dim: int, num_nodes: int):
        """Build policy and critic networks"""
        pass
    
    def _build_optimizer(self, lr: float):
        """Build optimizer for networks"""
        self.optimizer = optim.Adam(
            list(self.policy.parameters()) + list(self.global_critic.parameters()),
            lr=lr
        )
    
    @abstractmethod
    def _compute_policy_output(self, state: torch.Tensor, edge_index: torch.Tensor) -> Tuple:
        """Compute policy network output"""
        pass
    
    @abstractmethod
    def _compute_critic_output(self, state: torch.Tensor) -> Tuple:
        """Compute critic network output"""
        pass
    
    def _sample_actions(self, phase_probs: torch.Tensor, adjust_probs: torch.Tensor) -> Tuple:
        """Sample actions from probability distributions"""
        phase_dist = torch.distributions.Categorical(phase_probs)
        adjust_dist = torch.distributions.Categorical(adjust_probs)
        
        phase_action = phase_dist.sample()
        adjust_action = adjust_dist.sample()
        
        combined_action = torch.stack((phase_action, adjust_action), dim=1)
        log_probs = (phase_dist.log_prob(phase_action), adjust_dist.log_prob(adjust_action))
        
        return combined_action, log_probs
    
    def select_action(self, state, edge_index):
        """Select action using current policy"""
        state = torch.FloatTensor(state).to(self.device)
        edge_index = torch.LongTensor(edge_index).to(self.device)

        with torch.no_grad():
            phase_probs, adjust_probs = self._compute_policy_output(state, edge_index)
            value, _ = self._compute_critic_output(state)

        value = value.squeeze(-1)
        combined_action, log_probs = self._sample_actions(phase_probs, adjust_probs)

        return (combined_action.cpu().numpy(), log_probs, 
                value.cpu().numpy(), state.cpu().numpy())
    
    def store_transition(self, state, action, log_probs, vals, reward, done):
        """Store transition in memory buffer"""
        self.memory.store_transition(state, action, log_probs, vals, reward, done)
    
    def _compute_gae_advantages(self, rewards: torch.Tensor, values: torch.Tensor, 
                               dones: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages"""
        steps = rewards.shape[0]
        global_rewards = rewards.sum(dim=1)
        global_dones = dones.prod(dim=1)

        advantages = torch.zeros_like(global_rewards).to(self.device)
        gae = torch.zeros(1).to(self.device)
        next_value = torch.zeros(1).to(self.device)

        for t in reversed(range(steps)):
            mask = 1.0 - global_dones[t]
            delta = global_rewards[t] + self.gamma * next_value * mask - values[t]
            gae = delta + self.gamma * self.gae_lambda * gae * mask
            advantages[t] = gae
            next_value = values[t]

        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def _compute_nll_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor, 
                         log_var: torch.Tensor) -> torch.Tensor:
        """Compute negative log-likelihood loss"""
        return 0.5 * torch.mean(log_var + (y_true - y_pred) ** 2 / torch.exp(log_var))
    
    def _compute_policy_loss(self, new_log_prob: torch.Tensor, old_log_prob: torch.Tensor,
                            advantage: torch.Tensor) -> torch.Tensor:
        """Compute clipped policy loss"""
        ratio = torch.exp(new_log_prob - old_log_prob)
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantage
        return -torch.min(surr1, surr2).mean()
    
    def _update_policy_step(self, state: torch.Tensor, action: torch.Tensor,
                           old_phase_prob: torch.Tensor, old_adjust_prob: torch.Tensor,
                           advantage: torch.Tensor, returns: torch.Tensor,
                           edge_index: torch.Tensor) -> torch.Tensor:
        """Single policy update step"""
        phase_probs, adjust_probs = self._compute_policy_output(state, edge_index)

        phase_dist = torch.distributions.Categorical(phase_probs)
        adjust_dist = torch.distributions.Categorical(adjust_probs)

        new_phase_log_prob = phase_dist.log_prob(action[:, 0])
        new_adjust_log_prob = adjust_dist.log_prob(action[:, 1])

        entropy = (phase_dist.entropy() + adjust_dist.entropy()).mean()

        actor_loss_phase = self._compute_policy_loss(new_phase_log_prob, old_phase_prob, advantage)
        actor_loss_adjust = self._compute_policy_loss(new_adjust_log_prob, old_adjust_prob, advantage)

        value_mean, value_logvar = self._compute_critic_output(state)
        critic_loss = self._compute_nll_loss(value_mean, returns, value_logvar)

        total_loss = (actor_loss_phase + actor_loss_adjust) + 1.0 * critic_loss - 0.01 * entropy
        
        return total_loss
    
    def learn(self, edge_index):
        """Train policy using collected experiences"""
        if self.memory.get_size() == 0:
            return

        batch_data = self.memory.get_batch_data()
        
        states = torch.FloatTensor(batch_data['states']).to(self.device)
        actions = torch.LongTensor(batch_data['actions']).to(self.device)
        old_phase_log_probs = torch.FloatTensor(batch_data['phase_probs']).to(self.device)
        old_adjust_log_probs = torch.FloatTensor(batch_data['adjust_probs']).to(self.device)
        old_vals = torch.stack([torch.FloatTensor(v) for v in batch_data['vals']]).squeeze(-1).to(self.device)
        rewards = torch.stack([torch.FloatTensor(r) for r in batch_data['rewards']]).to(self.device)
        dones = torch.stack([torch.FloatTensor(d) for d in batch_data['dones']]).to(self.device)

        steps = states.shape[0]

        advantages, returns = self._compute_gae_advantages(rewards, old_vals, dones)

        for _ in range(self.epochs):
            idxs = torch.randperm(steps).to(self.device)
            for start in range(0, steps, self.batch_size):
                end = start + self.batch_size
                batch_idx = idxs[start:end]

                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_phase_probs = old_phase_log_probs[batch_idx]
                batch_old_adjust_probs = old_adjust_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]

                total_loss = 0

                for i in range(batch_states.shape[0]):
                    loss = self._update_policy_step(
                        batch_states[i], batch_actions[i],
                        batch_old_phase_probs[i], batch_old_adjust_probs[i],
                        batch_advantages[i], batch_returns[i],
                        edge_index
                    )
                    total_loss += loss

                total_loss = total_loss / batch_states.shape[0]

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

        self._sync_target_network()
        self.memory.clear()

        return total_loss.item()
    
    @abstractmethod
    def _sync_target_network(self):
        """Synchronize target network with current policy"""
        pass
    
    def save_model(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'policy': self.policy.state_dict(),
            'critic': self.global_critic.state_dict()
        }, path)

    def load_model(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy'])
        self.global_critic.load_state_dict(checkpoint['critic'])
        self._sync_target_network()
