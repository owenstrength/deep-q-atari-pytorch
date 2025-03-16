import gymnasium as gym
import ale_py
import numpy as np
import random
import torch
import torch.optim as optim
import torch.nn.functional as F
from model import DQN  # Import the DQN model

# Check if Metal is available and set the device accordingly
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

class SimpleAgent:
    def __init__(self, action_space):
        self.action_space = action_space
        self.model = DQN(action_space.n).to(device)  # Move model to the selected device
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.replay_buffer = []  # Experience replay buffer
        self.batch_size = 32
        self.gamma = 0.99  # Discount factor

    def select_action(self, state, epsilon=0.1):
        if random.random() < epsilon:
            return random.choice(range(self.action_space.n))  # Explore
        else:
            with torch.no_grad():
                # Ensure proper dimensions: [batch, channels, height, width]
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)  # Add batch dimension and move to device
                q_values = self.model(state_tensor)
                return q_values.argmax().item()  # Exploit

    def store_experience(self, experience):
        self.replay_buffer.append(experience)
        if len(self.replay_buffer) > 10000:  # Limit the size of the replay buffer
            self.replay_buffer.pop(0)

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return  # Not enough samples to train

        # Sample a batch of experiences
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors with proper shapes and move to device
        states_tensor = torch.FloatTensor(np.array(states)).to(device)
        actions_tensor = torch.LongTensor(actions).to(device)
        rewards_tensor = torch.FloatTensor(rewards).to(device)
        next_states_tensor = torch.FloatTensor(np.array(next_states)).to(device)
        dones_tensor = torch.FloatTensor(dones).to(device)

        # Compute Q values
        q_values = self.model(states_tensor).gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
        next_q_values = self.model(next_states_tensor).max(1)[0]
        target_q_values = rewards_tensor + (self.gamma * next_q_values * (1 - dones_tensor))

        # Compute loss and update the model
        loss = F.mse_loss(q_values, target_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

def preprocess_frame(frame):
    """Convert RGB frame to grayscale and resize."""
    # Convert to grayscale by taking mean across RGB channels
    gray = np.mean(frame, axis=2).astype(np.float32)
    # Normalize to range [0, 1]
    normalized = gray / 255.0
    return normalized

def train_agent(env_name, num_episodes=1000):
    # Register Atari environments
    gym.register_envs(ale_py)
    env = gym.make(env_name)
    agent = SimpleAgent(env.action_space)

    for episode in range(num_episodes):
        state, _ = env.reset()
        
        # Initialize frame stack with the first frame
        processed_frame = preprocess_frame(state)
        # Stack the same frame 4 times for initial state
        stacked_frames = np.stack([processed_frame] * 4, axis=0)  # Shape: [4, height, width]
        
        done = False
        total_reward = 0

        while not done:
            action = agent.select_action(stacked_frames)
            next_state, reward, done, truncated, _ = env.step(action)
            
            # Preprocess new frame
            processed_next_frame = preprocess_frame(next_state)
            
            # Create next stacked frames by removing oldest frame and adding new frame
            next_stacked_frames = np.concatenate([
                stacked_frames[1:],  # Remove oldest frame
                np.expand_dims(processed_next_frame, axis=0)  # Add new frame as last
            ], axis=0)

            total_reward += reward

            # Store experience
            agent.store_experience((
                stacked_frames,
                action,
                reward, 
                next_stacked_frames,
                done
            ))
            
            # Update current state
            stacked_frames = next_stacked_frames
            
            # Train the agent
            agent.train()

            if done or truncated:
                break

        print(f"Episode {episode + 1}: Total Reward: {total_reward}")

    env.close()

if __name__ == "__main__":
    train_agent("ALE/Pong-v5")  # change for different env
