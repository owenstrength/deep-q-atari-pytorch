from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from model import DQN

class DQNAgent:
    def __init__(self, action_space, device):
        self.action_space = action_space
        self.device = device
        self.model = DQN(action_space.n).to(self.device)
        self.target_model = DQN(action_space.n).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        
        # RMSProp parameters from the paper
        self.optimizer = optim.RMSprop(
            self.model.parameters(),
            lr=0.00025,
            alpha=0.95,
            eps=0.01
        )
        
        # Experience replay buffer (capacity of 1,000,000 from the paper)
        self.replay_buffer = deque(maxlen=1000000) # changed to 100k
        self.batch_size = 32
        self.gamma = 0.99  # Discount factor
        
        # Epsilon annealing
        self.epsilon_start = 1.0
        self.epsilon_end = 0.05 #paper ends at point 0.05
        self.epsilon_decay_steps = 1000000  # 1M frames for linear decay
        self.epsilon = self.epsilon_start
        
        # Target network update frequency (every 10,000 steps from the paper)
        self.target_update_freq = 1000 # changed to 100
        
        # Counters
        self.train_step = 0
        self.frame_count = 0

    def select_action(self, state):
        """Select an action using epsilon-greedy strategy with annealing epsilon."""
        # Update epsilon based on linear schedule
        if self.frame_count < self.epsilon_decay_steps:
            self.epsilon = self.epsilon_start - (self.frame_count / self.epsilon_decay_steps) * (self.epsilon_start - self.epsilon_end)
        else:
            self.epsilon = self.epsilon_end
        
        self.frame_count += 1
        
        if random.random() < self.epsilon:
            return random.choice(range(self.action_space.n))  # Explore
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.model(state_tensor)
                return q_values.argmax().item()  # Exploit

    def store_experience(self, experience):
        """Store experience in the replay buffer with reward clipping."""
        state, action, reward, next_state, done = experience
        
        # Clip rewards to [-1, 1] as per the paper
        clipped_reward = np.clip(reward, -1, 1)
        
        self.replay_buffer.append((state, action, clipped_reward, next_state, done))

    def train(self):
        """Train the agent using experience replay."""
        if len(self.replay_buffer) < self.batch_size:
            return  # Not enough samples
            
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        next_states_tensor = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones_tensor = torch.FloatTensor(dones).to(self.device)
        
        # Compute Q values for current states and actions
        q_values = self.model(states_tensor).gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
        
        # Compute target Q values (using target network as per the paper)
        with torch.no_grad():
            next_q_values = self.target_model(next_states_tensor).max(1)[0]
            target_q_values = rewards_tensor + (self.gamma * next_q_values * (1 - dones_tensor))
        
        # Compute loss using Huber loss (smooth_l1_loss) as per the paper
        loss = F.smooth_l1_loss(q_values, target_q_values)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping (not explicitly mentioned in the paper but often used)
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
            
        self.optimizer.step()
        
        self.train_step += 1
        
        # Update target network periodically
        if self.train_step % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def load_model(self, model_path):
        """Load the trained DQN model from the specified path."""
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def act(self, state):
        """Select best action (used for evaluation)."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.model(state_tensor)
            return q_values.argmax().item()