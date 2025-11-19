import torch

ENV_CONFIG = {
    'sumocfg_file': './env/intersection/hangzhou.sumocfg',
    'max_steps': 3600,
    'gui': False,
    'state_dim': 13,
    'action_dim': 8,
    'yellow_duration': 3,
    'default_green_time': 10,
    'min_green_time': 5,
    'adjust_step': 5,
    'adjust_dim': 3,
    'action_interval': 13  
}

PPO_CONFIG = {
    'lr':2e-5, 
    'gamma': 0.99,
    'gae_lambda': 0.97,
    'clip_epsilon': 0.2,
    'epochs': 10,
    'batch_size': 32,
    'exploration_noise': 0.1,
    'device': "cuda" if torch.cuda.is_available() else "cpu",
    'latent_dim': 32,
    'nll_weight': 1.0,
    'noise_std': 0.8
}

TRAIN_CONFIG = {
    'total_episodes': 300,
    'log_interval': 10,
    'save_interval': 20
}

