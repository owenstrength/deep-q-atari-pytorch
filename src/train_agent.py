import os
import ale_py
import gymnasium as gym
import numpy as np
import cv2
import torch
import pandas as pd
import psutil
import gc
from collections import deque
from agent import DQNAgent
from tqdm import tqdm

# Check available devices
device = torch.device("mps" if torch.backends.mps.is_available() else 
                     "cuda" if torch.cuda.is_available() else 
                     "cpu")
print(f"Using device: {device}")
#device = torch.device("cpu")  # Force CPU for stability

def preprocess_frame(frame):
    """Convert RGB frame to grayscale, resize to 84x84, and normalize."""
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    normalized = resized / 255.0  # Normalize to [0,1]
    return normalized

def train_agent(env_name, num_frames=50000000, checkpoint=None):
    """Train DQN agent following the original paper's methodology."""
    # Create environment
    gym.register_envs(ale_py)
    env = gym.make(env_name, render_mode="rgb_array")
    agent = DQNAgent(env.action_space, device)
    
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    log_file = "logs/reward_log.csv"
    
    # Initialize training variables
    log_data = []
    episode = 0
    frame_idx = 0
    evaluation_interval = 250000  # Evaluate every 250k frames
    save_model_interval = 10000  # Save model every 10k frames
    memory_check_interval = 5000  # Check memory usage every 5k frames
    gc_interval = 1000  # Run garbage collection every 1k frames
    
    # Load from checkpoint if provided
    if checkpoint and os.path.exists(checkpoint):
        agent.model.load_state_dict(torch.load(checkpoint, map_location=device))
        agent.target_model.load_state_dict(torch.load(checkpoint, map_location=device))
        # Extract frame number from filename
        try:
            start_frame = int(checkpoint.split('_')[-1].split('.')[0])
            frame_idx = start_frame
            print(f"Continuing from checkpoint at frame {start_frame}")
            
            # Try to load existing log data if available
            if os.path.exists(log_file):
                existing_log = pd.read_csv(log_file)
                if not existing_log.empty:
                    log_data = existing_log.to_dict('records')
                    episode = max(existing_log['episode']) + 1
                    print(f"Loaded existing log data, continuing from episode {episode}")
        except:
            print("Couldn't parse frame number from checkpoint filename, starting from frame 0")
    
    # Initialize the first state
    state, _ = env.reset()
    processed_frame = preprocess_frame(state)
    stacked_frames = np.stack([processed_frame] * 4, axis=0)
    
    # Pre-populate replay memory with random actions (if not loading from checkpoint)
    if not checkpoint:
        print("Initializing replay memory with random experiences...")
        for _ in tqdm(range(min(10000, agent.replay_buffer.maxlen))):  # Reduced from 50k to 10k
            action = env.action_space.sample()  # Random action
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # Process next frame
            processed_next_frame = preprocess_frame(next_state)
            next_stacked_frames = np.roll(stacked_frames, shift=-1, axis=0)
            next_stacked_frames[-1] = processed_next_frame
            
            # Store experience
            agent.store_experience((
                stacked_frames,
                action, 
                reward,
                next_stacked_frames,
                terminated
            ))
            
            stacked_frames = next_stacked_frames
            
            if terminated or truncated:
                state, _ = env.reset()
                processed_frame = preprocess_frame(state)
                stacked_frames = np.stack([processed_frame] * 4, axis=0)
    
    # Main training loop (frame-based, not episode-based as in the paper)
    print("Starting main training loop...")
    pbar = tqdm(total=num_frames, initial=frame_idx)
    
    episode_reward = 0
    lives = 0  # Track lives for handling terminal states in games with lives
    last_100_rewards = deque(maxlen=100)
    
    # Reset for training
    state, info = env.reset()
    processed_frame = preprocess_frame(state)
    stacked_frames = np.stack([processed_frame] * 4, axis=0)
    
    if 'lives' in info:
        lives = info['lives']
    
    while frame_idx < num_frames:
        # Select and perform action
        action = agent.select_action(stacked_frames)
        next_state, reward, terminated, truncated, info = env.step(action)
        
        episode_reward += reward
        frame_idx += 1
        pbar.update(1)
        
        # Check for loss of life for certain games
        life_terminal = False
        if 'lives' in info:
            if info['lives'] < lives:
                life_terminal = True
            lives = info['lives']
        
        # Process next frame
        processed_next_frame = preprocess_frame(next_state)
        
        # Create next stacked state
        next_stacked_frames = np.roll(stacked_frames, shift=-1, axis=0)
        next_stacked_frames[-1] = processed_next_frame
        
        # Store the transition
        agent.store_experience((
            stacked_frames,
            action,
            reward,
            next_stacked_frames,
            terminated or life_terminal
        ))
        
        # Update the current state
        stacked_frames = next_stacked_frames
        
        # Train the agent
        agent.train()
        
        # Run garbage collection periodically
        if frame_idx % gc_interval == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Check memory usage periodically
        if frame_idx % memory_check_interval == 0:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            memory_usage_gb = memory_info.rss / 1024 / 1024 / 1024
            tqdm.write(f"Memory usage: {memory_usage_gb:.2f} GB")
        
        # Handle episode termination
        if terminated or truncated:
            last_100_rewards.append(episode_reward)
            avg_reward = np.mean(last_100_rewards) if last_100_rewards else episode_reward
            
            # Log episode stats
            log_data.append({
                "episode": episode,
                "frames": frame_idx,
                "reward": episode_reward,
                "avg_reward_last_100": avg_reward,
                "epsilon": agent.epsilon
            })
            
            # Print progress
            tqdm.write(f"Episode {episode}: Frames: {frame_idx}, Reward: {episode_reward}, " 
                       f"Avg Reward (100): {avg_reward:.2f}, Epsilon: {agent.epsilon:.4f}")
            
            # Reset the environment
            state, info = env.reset()
            processed_frame = preprocess_frame(state)
            stacked_frames = np.stack([processed_frame] * 4, axis=0)
            episode_reward = 0
            episode += 1
            
            if 'lives' in info:
                lives = info['lives']
            
            # Force garbage collection on episode end
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Save the model periodically
        if frame_idx % save_model_interval == 0:
            model_path = f"logs/{env_name.split('/')[1]}_frame_{frame_idx}.pth"
            torch.save(agent.model.state_dict(), model_path)
            print(f"Model saved to {model_path}")
            
            # Save log data to CSV
            if log_data:
                pd.DataFrame(log_data).to_csv(log_file, index=False)
    
    # Final save
    model_path = f"logs/{env_name.split('/')[1]}_model_final.pth"
    torch.save(agent.model.state_dict(), model_path)
    print(f"Final model saved to {model_path}")
    
    # Save final log data
    if log_data:
        pd.DataFrame(log_data).to_csv(log_file, index=False)
    
    # Close environment
    env.close()
    pbar.close()
    print("Training complete!")

if __name__ == "__main__":
    # Uncomment line below to continue from your last checkpoint
    #train_agent("ALE/Pong-v5", num_frames=5000000, checkpoint="logs/Pong-v5_frame_190000.pth")
    
    # Or start fresh
    train_agent("ALE/Pong-v5", num_frames=5000000)