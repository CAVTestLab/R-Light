import os
import sys
from typing import Dict, Any

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import (
    TRAINING_ALGORITHM, 
    ENV_CONFIG, 
    AGENT_CONFIG, 
    TRAIN_CONFIG
)
from core.trainer_factory import TrainerFactory
from core.base_agent import BaseAgent

class TrainingManager:
    """Training manager using facade pattern"""
    
    def __init__(self, algorithm: str = TRAINING_ALGORITHM):
        self.algorithm = algorithm
        self.config = self._merge_configs()
        self.trainer = None
    
    def _merge_configs(self) -> Dict[str, Any]:
        """Merge configuration dictionaries"""
        merged_config = {}
        merged_config.update(ENV_CONFIG)
        merged_config.update(AGENT_CONFIG)
        merged_config.update(TRAIN_CONFIG)
        return merged_config
    
    def _validate_algorithm(self) -> None:
        """Validate algorithm support"""
        supported = TrainerFactory.get_supported_algorithms()
        if self.algorithm not in supported:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}. Supported: {supported}")
    
    def setup_trainer(self) -> None:
        """Setup trainer instance"""
        self._validate_algorithm()
        print(f"Using algorithm: {self.algorithm}")
        self.trainer = TrainerFactory.create_trainer(self.algorithm, self.config)
    
    def start_training(self) -> BaseAgent:
        """Start training process"""
        if self.trainer is None:
            self.setup_trainer()
        
        return self.trainer.train(
            total_episodes=self.config['total_episodes'],
            max_steps=self.config['max_steps'],
            model_save_path=self.config['model_save_path'],
            log_interval=self.config['log_interval']
        )

def main():
    """Main training function"""
    try:
        manager = TrainingManager()
        agent = manager.start_training()
        print("Training completed successfully!")
        return agent
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise

if __name__ == "__main__":
    agent = main()
