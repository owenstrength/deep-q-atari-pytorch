import os
import gymnasium as gym
import numpy as np
import torch
import cv2
from agent import DQNAgent  # Import the DQNAgent class
import ale_py

def evaluate_agent(env_name, model_path, num_episodes=10):
    """Evaluate the DQN agent in the specified environment."""
    # Create the environment
    gym.register_envs(ale_py)
    env = gym.make(env_name, render_mode="human")  # Render in human mode
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    agent = DQNAgent(env.action_space, device)  # Initialize the agent
    agent.load_model(model_path)  # Load the trained model

    total_rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset()  # Reset the environment
        done = False
        total_reward = 0

        while not done:
            # Preprocess the state
            state = preprocess_frame(state)  # Assuming preprocess_frame is defined in this file
            state = np.stack([state] * 4, axis=0)  # Stack the frames (4, 84, 84)
            state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension

            # Select action using the agent's act method
            action = agent.act(state)

            # Step the environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            state = next_state  # Update state

            if terminated or truncated:
                break

        total_rewards.append(total_reward)
        print(f"Episode {episode + 1}: Total Reward: {total_reward}")

    env.close()
    return total_rewards

def preprocess_frame(frame):
    """Convert RGB frame to grayscale, resize to 84x84, and normalize."""
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    normalized = resized / 255.0  # Normalize to [0,1]
    return normalized

if __name__ == "__main__":
    model_path = "logs/dqn_model_episode_3200.pth"  # Update this path to your model
    evaluate_agent("ALE/Pong-v5", model_path, num_episodes=10)  # Evaluate the agent 