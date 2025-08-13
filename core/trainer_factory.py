from typing import Dict, Any
from core.base_trainer import BaseTrainer
from trainers.dqn_gcn_trainer import DQNGCNTrainer
from trainers.d3qn_trainer import D3QNTrainer

class TrainerFactory:
    """Factory class for creating trainers"""
    
    _trainers = {
        'DQN_GCN': DQNGCNTrainer,
        'D3QN': D3QNTrainer
    }
    
    @classmethod
    def create_trainer(cls, algorithm: str, config: Dict[str, Any]) -> BaseTrainer:
        """Create trainer instance"""
        if algorithm not in cls._trainers:
            raise ValueError(f"Unsupported algorithm: {algorithm}. Supported algorithms: {list(cls._trainers.keys())}")
        
        trainer_class = cls._trainers[algorithm]
        trainer_config = config.copy()
        trainer_config['algorithm'] = algorithm
        
        return trainer_class(trainer_config)
    
    @classmethod
    def get_supported_algorithms(cls) -> list:
        """Get list of supported algorithms"""
        return list(cls._trainers.keys())
