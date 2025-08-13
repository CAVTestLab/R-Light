import torch

TRAINING_ALGORITHM = "DQN_GCN"  # Options: "DQN_GCN", "D3QN"

ENV_CONFIG = {
    'sumocfg_file': './env/intersection/hangzhou.sumocfg',
    'green_duration': 10,
    'yellow_duration': 3,
    'times': 13,
    'max_steps': 900,
    'gui': False,
    'state_dim': 8,
    'action_dim': 4
}

AGENT_CONFIG = {
    'state_dim': 8,
    'action_dim': 8,
    'learning_rate': 1e-4,
    'gamma': 0.99,
    'epsilon_start': 1.0,
    'epsilon_end': 0.1,
    'buffer_size': 50000,
    'batch_size': 64,
    'target_update': 20,
    'tau': 0.005,
    'device': "cuda" if torch.cuda.is_available() else "cpu"
}

TRAIN_CONFIG = {
    'total_episodes': 300,
    'log_interval': 10,
    'model_save_path': './models/traffic_control',
    'algorithm_configs': {
        'DQN_GCN': {
            'early_stop_ratio': 0.66,
            'use_adjacency_matrix': True
        },
        'D3QN': {
            'early_stop_ratio': 1.0,
            'use_adjacency_matrix': False
        }
    }
}

# Training configuration
TRAIN_CONFIG = {
    # Total number of training episodes
    'total_episodes': 300,
    # Logging interval
    'log_interval': 10,
    # Model save path
    'model_save_path': './models/traffic_control',
    # Algorithm specific configurations
    'algorithm_configs': {
        'DQN_GCN': {
            # Early stopping ratio
            'early_stop_ratio': 0.66,  # Stop at 66% of the episodes
            # Use adjacency matrix
            'use_adjacency_matrix': True
        },
        'D3QN': {
            # Complete training
            'early_stop_ratio': 1.0,
            # Do not use adjacency matrix
            'use_adjacency_matrix': False
        }
    }
}
