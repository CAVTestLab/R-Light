import os
import numpy as np
import time

from env.traffic_env import TrafficEnv
from agent.d3qn_agent import D3QNAgent
from config import ENV_CONFIG, AGENT_CONFIG, TRAIN_CONFIG

def train(
    total_episodes=TRAIN_CONFIG['total_episodes'],
    max_steps=ENV_CONFIG['max_steps'],
    model_save_path=TRAIN_CONFIG['model_save_path'],
    log_interval=TRAIN_CONFIG['log_interval'],
):
    """Train D3QN agent for traffic light control"""
    print("Initializing environment...")
    env = TrafficEnv()

    num_nodes = env.num_nodes
    state_dim = AGENT_CONFIG['state_dim']
    action_dim = AGENT_CONFIG['action_dim']

    print(f"Initializing D3QN agent (state_dim: {state_dim}, action_dim: {action_dim}, nodes: {num_nodes})...")
    agent = D3QNAgent()

    print(f"Starting training for {total_episodes} episodes...")
    for episode in range(total_episodes):
        epsilon = agent.update_epsilon(episode, total_episodes)
        state = env.reset()
        total_reward = 0
        queue_lengths = []
        step = 0
        start_time = time.time()

        while step < max_steps:
            action = agent.select_action(
                state[np.newaxis],
                num_nodes,
                training=True
            ).squeeze(0)

            next_state, reward, info = env.step(action)

            episode_reward = sum(reward)
            total_reward += episode_reward
            queue_lengths.append(info['queue_length'])

            agent.remember(state[np.newaxis], action, reward, next_state[np.newaxis])
            agent.train()

            state = next_state
            step = info["step"]

        end_time = time.time()
        episode_time = end_time - start_time
        avg_queue_length = sum(queue_lengths) / len(queue_lengths)
        agent.record_stats(total_reward, avg_queue_length)

        if (episode + 1) % log_interval == 0:
            print(f"Episode {episode+1}/{total_episodes} - " +
                  f"Total reward: {total_reward:.2f} - " +
                  f"Queue length: {avg_queue_length:.2f} - " +
                  f"Epsilon: {epsilon:.2f} - " +
                  f"Time: {episode_time:.2f}s")

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    agent.save_model(f"{model_save_path}_final.pth")
    env.close()
    return agent


if __name__ == "__main__":
    agent = train()
