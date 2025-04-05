import os
import gymnasium as gym
import numpy as np
import torch
import cv2
from agent import DQNAgent
import ale_py

def preprocess_frame(frame):
    """Convert RGB frame to grayscale, resize to 84x84, and normalize."""
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    normalized = resized / 255.0  # Normalize to [0,1]
    return normalized

def evaluate_agent(env_name, model_path, num_episodes=10):
    """Evaluate the DQN agent in the specified environment."""
    # Create the environment
    gym.register_envs(ale_py)
    env = gym.make(env_name, render_mode="human")  # Render in human mode
    
    # Use the same device as training
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                         "cuda" if torch.cuda.is_available() else 
                         "cpu")
    print(f"Using device: {device}")
    
    agent = DQNAgent(env.action_space, device)
    agent.load_model(model_path)
    
    total_rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        
        # Initial frame processing
        processed_frame = preprocess_frame(state)
        processed_frame_tensor = torch.FloatTensor(processed_frame).to(device)
        stacked_frames = torch.stack([processed_frame_tensor] * 4, dim=0)
        
        total_reward = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            # Get action using the same logic as in training
            action = agent.act(stacked_frames)
            
            # Step the environment
            next_state, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            
            # Process next frame
            processed_next_frame = preprocess_frame(next_state)
            processed_next_frame_tensor = torch.FloatTensor(processed_next_frame).to(device)
            
            # Update stacked frames (slide the window)
            next_stacked_frames = torch.roll(stacked_frames, shifts=-1, dims=0)
            next_stacked_frames[-1] = processed_next_frame_tensor
            stacked_frames = next_stacked_frames
            
        print(f"Episode {episode + 1}: Total Reward: {total_reward}")
        total_rewards.append(total_reward)
    
    env.close()
    
    # Calculate and print average reward
    average_reward = np.mean(total_rewards)
    print(f"Average Reward over {num_episodes} episodes: {average_reward:.2f}")
    
    return total_rewards

if __name__ == "__main__":
    env = "BeamRider-v5"
    model_path = f"logs_{env}/{env}_frame_2840000.pth"  # Update this path to your model
    evaluate_agent(f"ALE/{env}", model_path, num_episodes=25)  # Evaluate the agent