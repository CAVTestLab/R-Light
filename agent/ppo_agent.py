import torch
from agent.base.base_ppo_agent import BasePPOAgent
from agent.components.neural_networks import HybridGraphNet, GlobalCritic

class PPOAgent(BasePPOAgent):
    """PPO agent with hybrid graph network"""
    
    def _build_networks(self, state_dim: int, phase_action_dim: int, 
                       adjust_action_dim: int, num_nodes: int):
        """Build policy and critic networks"""
        self.policy = HybridGraphNet(state_dim, phase_action_dim, adjust_action_dim).to(self.device)
        self.policy_old = HybridGraphNet(state_dim, phase_action_dim, adjust_action_dim).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.global_critic = GlobalCritic(state_dim, num_nodes).to(self.device)
    
    def _compute_policy_output(self, state: torch.Tensor, edge_index: torch.Tensor):
        """Compute policy network output"""
        return self.policy_old(state, edge_index)
    
    def _compute_critic_output(self, state: torch.Tensor):
        """Compute critic network output"""
        return self.global_critic(state)
    
    def _sync_target_network(self):
        """Synchronize old policy with current policy"""
        self.policy_old.load_state_dict(self.policy.state_dict())
