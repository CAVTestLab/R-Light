from abc import ABC, abstractmethod
import os
import time
import numpy as np
from typing import Dict, Any, Optional, Tuple
from env.traffic_env import TrafficEnv
from core.base_agent import BaseAgent

class BaseTrainer(ABC):
    """Abstract base class for trainers"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.env = None
        self.agent = None
    
    def setup_environment(self) -> TrafficEnv:
        """Setup training environment"""
        print("Initializing environment...")
        self.env = TrafficEnv()
        return self.env
    
    @abstractmethod
    def create_agent(self) -> BaseAgent:
        """Create agent (factory method)"""
        pass
    
    @abstractmethod
    def select_action_for_agent(self, state: np.ndarray, step: int) -> np.ndarray:
        """Select action using agent-specific parameters"""
        pass
    
    @abstractmethod
    def train_agent(self, step: int) -> None:
        """Train agent using algorithm-specific parameters"""
        pass
    
    def should_early_stop(self, episode: int, total_episodes: int) -> bool:
        """Check if early stopping condition is met"""
        algorithm_config = self.config['algorithm_configs'][self.config['algorithm']]
        early_stop_ratio = algorithm_config.get('early_stop_ratio', 1.0)
        return episode >= total_episodes * early_stop_ratio
    
    def train_episode(self, episode: int, total_episodes: int, max_steps: int) -> Dict[str, float]:
        """Train single episode"""
        # Update exploration rate
        epsilon = self.agent.update_epsilon(episode, total_episodes)
        
        # Reset environment
        state = self.env.reset()
        total_reward = 0
        queue_lengths = []
        step = 0
        
        start_time = time.time()
        
        while step < max_steps:
            # Select action using agent-specific method
            action = self.select_action_for_agent(state, step)
            
            # Execute action
            next_state, reward, info = self.env.step(action)
            
            # Record reward
            episode_reward = sum(reward)
            total_reward += episode_reward
            queue_lengths.append(info['queue_length'])
            
            # Store experience
            self.agent.remember(
                state[np.newaxis], 
                action, 
                reward, 
                next_state[np.newaxis]
            )
            
            # Train agent using algorithm-specific parameters
            self.train_agent(step)
            
            # Update state
            state = next_state
            step = info["step"]
        
        end_time = time.time()
        episode_time = end_time - start_time
        avg_queue_length = sum(queue_lengths) / len(queue_lengths)
        
        # Record statistics
        self.agent.record_stats(total_reward, avg_queue_length)
        
        return {
            'total_reward': total_reward,
            'avg_queue_length': avg_queue_length,
            'epsilon': epsilon,
            'episode_time': episode_time
        }
    
    def train(self, total_episodes: int, max_steps: int, 
              model_save_path: str, log_interval: int) -> BaseAgent:
        """Template method: main training loop"""
        # Setup environment
        self.setup_environment()
        
        # Create agent
        self.agent = self.create_agent()
        
        # Get environment info
        num_nodes = self.env.num_nodes
        state_dim = self.config['state_dim']
        action_dim = self.config['action_dim']
        
        print(f"Initializing {self.config['algorithm']} agent " + 
              f"(state_dim: {state_dim}, action_dim: {action_dim}, num_nodes: {num_nodes})...")
        
        print(f"Starting training for {total_episodes} episodes...")
        
        # Training loop
        for episode in range(total_episodes):
            # Train single episode
            episode_stats = self.train_episode(episode, total_episodes, max_steps)
            
            # Log progress
            if (episode + 1) % log_interval == 0:
                print(f"Episode {episode+1}/{total_episodes} - " +
                      f"Total reward: {episode_stats['total_reward']:.2f} - " + 
                      f"Queue length: {episode_stats['avg_queue_length']:.2f} - " + 
                      f"Epsilon: {episode_stats['epsilon']:.2f} - " +
                      f"Time: {episode_stats['episode_time']:.2f}s")
            
            # Check early stopping
            if self.should_early_stop(episode, total_episodes):
                break
        
        # Save model
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        self.agent.save_model(f"{model_save_path}_final.pth")
        
        # Close environment
        self.env.close()
        
        return self.agent
        # 关闭环境
        self.env.close()
        
        return self.agent
