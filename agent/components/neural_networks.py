import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv
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

class AbstractGraphComponent(nn.Module, ABC):
    """Abstract base class for graph components"""
    
    @abstractmethod
    def _build_architecture(self):
        """Build network architecture"""
        pass
    
    @abstractmethod
    def _process_graph_data(self, x, edge_index):
        """Process graph-structured data"""
        pass

class LayerWrapper(nn.Module):
    """Universal layer wrapper with optional activation and normalization"""
    
    def __init__(self, layer, activation=None, normalize=False):
        super(LayerWrapper, self).__init__()
        self.layer = layer
        self.activation = activation
        self.normalize = normalize
        if normalize:
            self.norm = nn.LayerNorm(layer.out_channels if hasattr(layer, 'out_channels') else layer.out_features)
    
    def forward(self, *args, **kwargs):
        x = self.layer(*args, **kwargs)
        if self.normalize:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class GraphConvolutionFactory:
    """Factory for creating graph convolution layers"""
    
    @staticmethod
    def create_gat_layer(input_dim, output_dim, **kwargs):
        return GATConv(input_dim, output_dim, **kwargs)
    
    @staticmethod
    def create_gcn_layer(input_dim, output_dim, **kwargs):
        return GCNConv(input_dim, output_dim, **kwargs)

class DualHeadOutputModule(nn.Module):
    """Dual-head output module with configurable activation"""
    
    def __init__(self, input_dim, head1_dim, head2_dim, activation_fn=None):
        super(DualHeadOutputModule, self).__init__()
        self._head1_dim = head1_dim
        self._head2_dim = head2_dim
        self._activation_fn = activation_fn or F.softmax
        
        self._head1_projection = nn.Linear(input_dim, head1_dim)
        self._head2_projection = nn.Linear(input_dim, head2_dim)
    
    def _apply_activation(self, logits, dim=-1):
        return self._activation_fn(logits, dim=dim)
    
    def forward(self, features):
        head1_logits = self._head1_projection(features)
        head2_logits = self._head2_projection(features)
        
        head1_output = self._apply_activation(head1_logits)
        head2_output = self._apply_activation(head2_logits)
        
        return head1_output, head2_output

class HybridGraphNet(AbstractGraphComponent):
    """Hybrid graph network with GAT and GCN layers"""
    
    def __init__(self, input_dim, phase_output_dim, adjust_output_dim):
        super(HybridGraphNet, self).__init__()
        self._input_dim = input_dim
        self._phase_output_dim = phase_output_dim
        self._adjust_output_dim = adjust_output_dim
        self._hidden_dim = 64
        
        self._build_architecture()
    
    def _build_architecture(self):
        """Construct network layers"""
        factory = GraphConvolutionFactory()
        
        # Build graph conv layers
        gat_layer = factory.create_gat_layer(self._input_dim, self._hidden_dim)
        gcn_layer = factory.create_gcn_layer(self._hidden_dim, self._hidden_dim)
        
        # Wrap layers with activation
        self._conv_layer_1 = LayerWrapper(gat_layer, activation=F.relu, normalize=False)
        self._conv_layer_2 = LayerWrapper(gcn_layer, activation=F.relu, normalize=False)
        
        # Initialize output module
        self._output_module = DualHeadOutputModule(
            self._hidden_dim, 
            self._phase_output_dim, 
            self._adjust_output_dim,
            activation_fn=F.softmax
        )
    
    def _process_graph_data(self, x, edge_index):
        """Apply graph convolutions"""
        x = self._conv_layer_1(x, edge_index)
        x = self._conv_layer_2(x, edge_index)
        return x
    
    def forward(self, x, edge_index):
        """
        x: [num_nodes, input_dim]
        edge_index: [2, num_edges]
        """
        features = self._process_graph_data(x, edge_index)
        phase_probs, adjust_probs = self._output_module(features)
        return phase_probs, adjust_probs

class TensorReshaper:
    """Utility for tensor reshaping operations"""
    
    @staticmethod
    def flatten_to_1d(tensor):
        return tensor.view(-1)
    
    @staticmethod
    def calculate_flattened_dim(latent_dim, num_nodes):
        return latent_dim * num_nodes

class NormalizationLayer(nn.Module):
    """Layer normalization wrapper"""
    
    def __init__(self, input_dim):
        super(NormalizationLayer, self).__init__()
        self._norm = nn.LayerNorm(input_dim)
    
    def forward(self, x):
        return self._norm(x)

class SharedFeatureExtractor(nn.Module):
    """Shared feature extraction module"""
    
    def __init__(self, input_dim, output_dim, dropout_rate=0.0):
        super(SharedFeatureExtractor, self).__init__()
        self._layers = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
        )
        if dropout_rate > 0:
            self._layers.add_module('dropout', nn.Dropout(p=dropout_rate))
    
    def forward(self, x):
        return self._layers(x)

class DistributionHeadModule(nn.Module):
    """Distribution head for mean and log-variance prediction"""
    
    def __init__(self, input_dim, output_dim=1):
        super(DistributionHeadModule, self).__init__()
        self._mean_head = nn.Linear(input_dim, output_dim)
        self._logvar_head = nn.Linear(input_dim, output_dim)
    
    def forward(self, features):
        mean = self._mean_head(features)
        logvar = self._logvar_head(features)
        return mean, logvar

class GlobalCritic(nn.Module):
    """Global critic network with distribution output"""
    
    def __init__(self, latent_dim, num_nodes):
        super(GlobalCritic, self).__init__()
        self._latent_dim = latent_dim
        self._num_nodes = num_nodes
        
        # Calculate flattened input dimension
        self._input_dim = TensorReshaper.calculate_flattened_dim(latent_dim, num_nodes)
        
        # Build network components
        self._normalization = NormalizationLayer(self._input_dim)
        self._feature_extractor = SharedFeatureExtractor(self._input_dim, 256, dropout_rate=0.0)
        self._distribution_head = DistributionHeadModule(256, output_dim=1)
    
    def _preprocess_input(self, z):
        """Flatten and normalize input"""
        z = TensorReshaper.flatten_to_1d(z)
        z = self._normalization(z)
        return z
    
    def _extract_features(self, z):
        """Extract high-level features"""
        return self._feature_extractor(z)
    
    def _compute_distribution(self, features):
        """Compute distribution parameters"""
        return self._distribution_head(features)
    
    def forward(self, z):
        z_processed = self._preprocess_input(z)
        features = self._extract_features(z_processed)
        value_mean, value_logvar = self._compute_distribution(features)
        return value_mean, value_logvar
