import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dense_to_sparse
from abc import ABC, abstractmethod
from typing import Optional, Tuple

class BaseNetwork(nn.Module, ABC):
    """Abstract base class for neural networks"""
    
    def __init__(self, state_dim: int, action_dim: int):
        super(BaseNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
    
    @abstractmethod
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Forward pass through the network"""
        pass

class MLPNetwork(BaseNetwork):
    """Multi-layer perceptron network"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: Tuple[int, ...] = (512, 256)):
        super(MLPNetwork, self).__init__(state_dim, action_dim)
        
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, action_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.network(x)

class DuelingMLPNetwork(BaseNetwork):
    """Dueling architecture MLP network"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: Tuple[int, ...] = (512, 256)):
        super(DuelingMLPNetwork, self).__init__(state_dim, action_dim)
        
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU()
        )
        
        self.value_stream = nn.Linear(hidden_dims[1], 1)
        self.advantage_stream = nn.Linear(hidden_dims[1], action_dim)
    
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        features = self.feature_layer(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Dueling aggregation
        q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q_values

class GCNNetwork(BaseNetwork):
    """Graph Convolutional Network"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: Tuple[int, ...] = (128, 128)):
        super(GCNNetwork, self).__init__(state_dim, action_dim)
        
        self.conv_layers = nn.ModuleList()
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            self.conv_layers.append(GCNConv(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        
        self.output_layer = GCNConv(prev_dim, action_dim)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        edge_index, _ = dense_to_sparse(adj)
        
        for conv_layer in self.conv_layers:
            x = torch.relu(conv_layer(x, edge_index))
        
        return torch.relu(self.output_layer(x, edge_index))

class NetworkFactory:
    """Factory for creating neural networks"""
    
    _networks = {
        'mlp': MLPNetwork,
        'dueling_mlp': DuelingMLPNetwork,
        'gcn': GCNNetwork
    }
    
    @classmethod
    def create_network(cls, network_type: str, state_dim: int, action_dim: int, **kwargs) -> BaseNetwork:
        """Create network instance"""
        if network_type not in cls._networks:
            raise ValueError(f"Unsupported network type: {network_type}")
        
        return cls._networks[network_type](state_dim, action_dim, **kwargs)
    
    @classmethod
    def get_supported_networks(cls) -> list:
        """Get list of supported network types"""
        return list(cls._networks.keys())
