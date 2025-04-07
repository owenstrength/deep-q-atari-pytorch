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
from config import config
import argparse

CONFIGS = config

parser = argparse.ArgumentParser(description='Train a reinforcement learning agent.')
parser.add_argument('--env', type=str, default='ALE/Pong-v5', help='Name of the environment')
parser.add_argument('--checkpoint', type=str, default=None, help='Path to the checkpoint to resume training from')
parser.add_argument('--num_frames', type=int, default=CONFIGS.NUM_FRAMES, help='Number of frames to train the agent')


# Check available devices
device = torch.device("cuda" if torch.cuda.is_available() else 
                     "mps" if torch.backends.mps.is_available() else 
                     "cpu")
print(f"Using device: {device}")

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
    
    if checkpoint:
        agent.epsilon = 0.05  # Set epsilon when resuming training, we assume we got this far
    
    # Create logs directory
    os.makedirs(f"logs_{env_name.split('/')[1]}", exist_ok=True)
    log_file = f"logs_{env_name.split('/')[1]}/reward_log.csv"
    
    # Initialize training variables
    log_data = []
    episode = 0
    frame_idx = 0
    save_model_interval = 10000  # Save model every 10k frames
    memory_check_interval = 5000  # Check memory usage every 5k frames
    gc_interval = 1000  # Run garbage collection every 1k frames
    
    # Load from checkpoint if provided
    if checkpoint and os.path.exists(checkpoint):
        checkpoint_dict = torch.load(checkpoint, map_location=device)
        agent.model.load_state_dict(checkpoint_dict['model_state_dict'])
        agent.target_model.load_state_dict(checkpoint_dict['model_state_dict'])
        agent.optimizer.load_state_dict(checkpoint_dict['optimizer_state_dict'])
        agent.frame_count = checkpoint_dict.get('frame_count', 0)
        agent.train_step = checkpoint_dict.get('train_step', 0)
        frame_idx = checkpoint_dict.get('frame_idx', 0)
        episode = checkpoint_dict.get('episode', 0)
        print(f"Continuing from checkpoint at frame {frame_idx}, episode {episode}")
        
        # Load existing log data if available
        if os.path.exists(log_file):
            try:
                existing_log = pd.read_csv(log_file)
                if not existing_log.empty:
                    log_data = existing_log.to_dict('records')
                    print(f"Loaded existing log data with {len(log_data)} entries")
            except Exception as e:
                print(f"Error loading log data: {e}")
    
    # Initialize the first state
    state, _ = env.reset()
    processed_frame = preprocess_frame(state)
    # Convert to tensor immediately
    processed_frame_tensor = torch.FloatTensor(processed_frame).to(device)
    stacked_frames = torch.stack([processed_frame_tensor] * 4, dim=0)
    
    # Pre-populate replay memory with random actions (if not loading from checkpoint)
    if not checkpoint:
        print("Initializing replay memory with random experiences...")
        init_experiences = min(10000, agent.buffer_size)  # Reduced from 50k to 10k
        for _ in tqdm(range(init_experiences)):
            action = env.action_space.sample()  # Random action
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # Process next frame
            processed_next_frame = preprocess_frame(next_state)
            processed_next_frame_tensor = torch.FloatTensor(processed_next_frame).to(device)
            
            # Create next stacked frames using tensor operations
            next_stacked_frames = torch.roll(stacked_frames, shifts=-1, dims=0)
            next_stacked_frames[-1] = processed_next_frame_tensor
            
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
                processed_frame_tensor = torch.FloatTensor(processed_frame).to(device)
                stacked_frames = torch.stack([processed_frame_tensor] * 4, dim=0)
    
    # Main training loop
    print("Starting main training loop...")
    pbar = tqdm(total=num_frames, initial=frame_idx)
    
    episode_reward = 0
    lives = 0  # Track lives for handling terminal states in games with lives
    last_100_rewards = deque(maxlen=100)
    
    # Reset for training
    state, info = env.reset()
    processed_frame = preprocess_frame(state)
    processed_frame_tensor = torch.FloatTensor(processed_frame).to(device)
    stacked_frames = torch.stack([processed_frame_tensor] * 4, dim=0)
    
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
        processed_next_frame_tensor = torch.FloatTensor(processed_next_frame).to(device)
        
        # Create next stacked state using tensor operations
        next_stacked_frames = torch.roll(stacked_frames, shifts=-1, dims=0)
        next_stacked_frames[-1] = processed_next_frame_tensor
        
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
            gpu_memory_allocated = 0
            if torch.cuda.is_available():
                gpu_memory_allocated = torch.cuda.memory_allocated() / 1024 / 1024 / 1024
                gpu_memory_reserved = torch.cuda.memory_reserved() / 1024 / 1024 / 1024
                tqdm.write(f"Memory usage: RAM {memory_usage_gb:.2f} GB, GPU allocated: {gpu_memory_allocated:.2f} GB, GPU reserved: {gpu_memory_reserved:.2f} GB")
            else:
                tqdm.write(f"Memory usage: RAM {memory_usage_gb:.2f} GB")
        
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
                "epsilon": agent.epsilon,
                "q_value": agent.q_values
            })
            
            # Print progress
            tqdm.write(f"Episode {episode}: Frames: {frame_idx}, Reward: {episode_reward}, " 
                       f"Avg Reward (100): {avg_reward:.2f}, Epsilon: {agent.epsilon:.4f}")
            
            # Reset the environment
            state, info = env.reset()
            processed_frame = preprocess_frame(state)
            processed_frame_tensor = torch.FloatTensor(processed_frame).to(device)
            stacked_frames = torch.stack([processed_frame_tensor] * 4, dim=0)
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
            # Create a dictionary with all the data to save
            checkpoint_dict = {
                'model_state_dict': agent.model.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'frame_idx': frame_idx,
                'episode': episode,
                'frame_count': agent.frame_count,
                'train_step': agent.train_step
            }
            
            model_path = f"logs_{env_name.split('/')[1]}/{env_name.split('/')[1]}_frame_{frame_idx}.pth"
            torch.save(checkpoint_dict, model_path)
            print(f"Model saved to {model_path}")
            
            # Only keep the most recent checkpoint to save space
            for old_checkpoint in os.listdir(f"logs_{env_name.split('/')[1]}"):
                if old_checkpoint.endswith(".pth") and old_checkpoint != model_path.split('/')[-1]:
                    try:
                        os.remove(os.path.join(f"logs_{env_name.split('/')[1]}", old_checkpoint))
                    except Exception as e:
                        print(f"Could not remove old checkpoint: {e}")
            
            # Save log data to CSV
            if log_data:
                pd.DataFrame(log_data).to_csv(log_file, index=False)
    
    # Final save
    checkpoint_dict = {
        'model_state_dict': agent.model.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'frame_idx': frame_idx,
        'episode': episode,
        'frame_count': agent.frame_count,
        'train_step': agent.train_step
    }
    
    model_path = f"logs_{env_name.split('/')[1]}/{env_name.split('/')[1]}_model_final.pth"
    torch.save(checkpoint_dict, model_path)
    print(f"Final model saved to {model_path}")
    
    # Save final log data
    if log_data:
        pd.DataFrame(log_data).to_csv(log_file, index=False)
    
    # Close environment
    env.close()
    pbar.close()
    print("Training complete!")

if __name__ == "__main__":
    args = parser.parse_args()
    env = args.env
    checkpoint = args.checkpoint
    num_frames = args.num_frames
    if checkpoint:
        train_agent(env, num_frames, checkpoint)
    else:
        train_agent(env, num_frames)
    