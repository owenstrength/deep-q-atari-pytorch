import os
import ale_py
import gymnasium as gym
import numpy as np
import random
import cv2
import torch
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from model import DQN  # Ensure model is implemented correctly
from tqdm import tqdm  # Import tqdm for the loading bar
import pandas as pd  # Import pandas for CSV logging

# Check if Metal, then CUDA, then CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

class DQNAgent:
    def __init__(self, action_space):
        self.action_space = action_space
        self.model = DQN(action_space.n).to(device)
        self.target_model = DQN(action_space.n).to(device)  # Target network for stability
        self.target_model.load_state_dict(self.model.state_dict())  # Initialize weights
        self.target_model.eval()  # Target model is not trained directly

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        self.replay_buffer = deque(maxlen=10000)  # Experience replay buffer
        self.batch_size = 32
        self.gamma = 0.99  # Discount factor
        self.epsilon = 0.05 
        self.target_update_freq = 1000  # Update target network every N steps

        self.train_step = 0  # Training step counter

    def select_action(self, state):
        """Select an action using epsilon-greedy strategy."""
        if random.random() < self.epsilon:
            return random.choice(range(self.action_space.n))  # Explore
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)  # Add batch dimension
                q_values = self.model(state_tensor)
                return q_values.argmax().item()  # Exploit

    def store_experience(self, experience):
        """Store experience in the replay buffer."""
        self.replay_buffer.append(experience)

    def train(self):
        """Train the agent using experience replay."""
        if len(self.replay_buffer) < self.batch_size:
            return  # Not enough samples

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        states_tensor = torch.FloatTensor(np.array(states)).to(device)
        actions_tensor = torch.LongTensor(actions).to(device)
        rewards_tensor = torch.FloatTensor(rewards).to(device)
        next_states_tensor = torch.FloatTensor(np.array(next_states)).to(device)
        dones_tensor = torch.FloatTensor(dones).to(device)

        # Compute Q values for selected actions
        q_values = self.model(states_tensor).gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

        # Compute Double DQN target Q values
        next_q_values = self.model(next_states_tensor).detach()
        max_next_actions = next_q_values.argmax(1)  # Greedy action selection
        next_q_target_values = self.target_model(next_states_tensor).gather(1, max_next_actions.unsqueeze(1)).squeeze(1)

        target_q_values = rewards_tensor + (self.gamma * next_q_target_values * (1 - dones_tensor))

        # Compute loss using Huber loss for stability
        loss = F.smooth_l1_loss(q_values, target_q_values.detach())

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())  # Update target network

def preprocess_frame(frame):
    """Convert RGB frame to grayscale, resize to 84x84, and normalize."""
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    normalized = resized / 255.0  # Normalize to [0,1]
    return normalized

def train_agent(env_name, num_episodes=50000):
    """Train DQN agent on Pong."""
    gym.register_envs(ale_py)
    env = gym.make(env_name, render_mode="rgb_array")  # Ensure correct Gymnasium API usage
    agent = DQNAgent(env.action_space)

    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    log_file = "logs/reward_log.csv"
    
    # Initialize CSV logging
    log_data = []

    # Use tqdm to create a progress bar for the training loop
    for episode in tqdm(range(num_episodes), desc="Training Episodes"):
        state, _ = env.reset()
        
        # Process first frame and stack it 4 times
        processed_frame = preprocess_frame(state)
        stacked_frames = np.stack([processed_frame] * 4, axis=0)  # (4, 84, 84)

        done = False
        total_reward = 0

        while not done:
            action = agent.select_action(stacked_frames)
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Preprocess new frame
            processed_next_frame = preprocess_frame(next_state)

            # Create next stacked state using np.roll() for efficiency
            next_stacked_frames = np.roll(stacked_frames, shift=-1, axis=0)
            next_stacked_frames[-1] = processed_next_frame  # Replace last frame

            total_reward += reward

            # Store experience
            agent.store_experience((
                stacked_frames,
                action,
                reward, 
                next_stacked_frames,
                terminated
            ))
            
            stacked_frames = next_stacked_frames  # Update current state
            agent.train()  # Train the agent

            if terminated or truncated:
                break
        
        # Log the episode number and total reward
        log_data.append({"episode": episode + 1, "reward": total_reward})

        # Update the progress bar with the current reward
        tqdm.write(f"Episode {episode + 1}: Total Reward: {total_reward}", end="\r")

        # Save the model every 100 episodes
        if (episode + 1) % 100 == 0:
            torch.save(agent.model.state_dict(), f"logs/dqn_model_episode_{episode + 1}.pth")
            # Save log data to CSV every 100 episodes
            pd.DataFrame(log_data).to_csv(log_file, index=False)

    # Final save of log data to CSV
    pd.DataFrame(log_data).to_csv(log_file, index=False)

    env.close()

if __name__ == "__main__":
    train_agent("ALE/Pong-v5")  # Train the agent