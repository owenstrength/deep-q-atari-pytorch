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
        self.q_values = []
        
        # RMSProp parameters from the paper
        self.optimizer = optim.RMSprop(
            self.model.parameters(),
            lr=0.00025,
            alpha=0.95,
            eps=0.01
        )
        
        # GPU-optimized replay buffer
        self.buffer_size = 30000  # Reduced from 1M to 100K to 30k
        
        # Preallocate tensors for state and next_state directly on GPU
        self.state_buffer = torch.zeros((self.buffer_size, 4, 84, 84), 
                                      dtype=torch.float32, device=self.device)
        self.next_state_buffer = torch.zeros((self.buffer_size, 4, 84, 84), 
                                           dtype=torch.float32, device=self.device)
        
        # These are smaller, so we keep them on CPU until batch creation
        self.action_buffer = np.zeros(self.buffer_size, dtype=np.int64)
        self.reward_buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.done_buffer = np.zeros(self.buffer_size, dtype=np.bool_)
        
        # Buffer tracking
        self.buffer_idx = 0
        self.buffer_full = False
        
        self.batch_size = 32
        self.gamma = 0.99  # Discount factor
        
        # Epsilon annealing
        self.epsilon_start = 1.0
        self.epsilon_end = 0.05
        self.epsilon_decay_steps = 1000000  # 1M frames for linear decay
        self.epsilon = self.epsilon_start
        
        # Target network update frequency
        self.target_update_freq = 10000  # Changed from 10K to 1K
        
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
                # No need to convert to tensor if already a tensor
                if not isinstance(state, torch.Tensor):
                    state = torch.FloatTensor(state).to(self.device)
                
                # Add batch dimension if needed
                if state.dim() == 3:
                    state = state.unsqueeze(0)
                
                q_values = self.model(state)
                return q_values.argmax().item()  # Exploit

    def store_experience(self, experience):
        """Store experience directly in GPU tensors to avoid later transfers."""
        state, action, reward, next_state, done = experience
        
        # Clip rewards to [-1, 1] as per the paper
        clipped_reward = np.clip(reward, -1, 1)
        
        # Store experience in buffers
        if isinstance(state, np.ndarray):
            self.state_buffer[self.buffer_idx] = torch.FloatTensor(state).to(self.device)
        else:
            self.state_buffer[self.buffer_idx] = state
            
        self.action_buffer[self.buffer_idx] = action
        self.reward_buffer[self.buffer_idx] = clipped_reward
        
        if isinstance(next_state, np.ndarray):
            self.next_state_buffer[self.buffer_idx] = torch.FloatTensor(next_state).to(self.device)
        else:
            self.next_state_buffer[self.buffer_idx] = next_state
            
        self.done_buffer[self.buffer_idx] = done
        
        # Update buffer position
        self.buffer_idx = (self.buffer_idx + 1) % self.buffer_size
        if self.buffer_idx == 0:
            self.buffer_full = True

    def train(self):
        """Train the agent using experience replay with GPU optimizations."""
        # Check if we have enough samples
        if not self.buffer_full and self.buffer_idx < self.batch_size:
            return
            
        # Sample batch of experiences
        buffer_range = self.buffer_size if self.buffer_full else self.buffer_idx
        batch_indices = np.random.choice(buffer_range, self.batch_size, replace=False)
        
        # Gather batch data - states/next_states already on GPU
        states = self.state_buffer[batch_indices]
        next_states = self.next_state_buffer[batch_indices]
        
        # Transfer smaller tensors to GPU only when needed
        actions = torch.LongTensor(self.action_buffer[batch_indices]).to(self.device)
        rewards = torch.FloatTensor(self.reward_buffer[batch_indices]).to(self.device)
        dones = torch.FloatTensor(self.done_buffer[batch_indices]).to(self.device)
        
        # Compute current Q values
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        self.q_values = current_q_values
        
        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute Huber loss
        loss = F.smooth_l1_loss(current_q_values, target_q_values)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
            
        self.optimizer.step()
        
        self.train_step += 1
        
        # Update target network periodically
        if self.train_step % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        
        # Explicitly delete tensors to help with memory management
        del actions, rewards, dones, current_q_values, next_q_values, target_q_values, loss

    def load_model(self, model_path):
        """Load the trained DQN model from the specified path."""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.target_model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'frame_count' in checkpoint:
                self.frame_count = checkpoint['frame_count']
            if 'train_step' in checkpoint:
                self.train_step = checkpoint['train_step']
        else:
            # Legacy loading for old checkpoints
            self.model.load_state_dict(checkpoint)
            self.target_model.load_state_dict(checkpoint)
            
        self.model.eval()

    def act(self, state):
        """Select best action (used for evaluation) without exploration."""
        with torch.no_grad():
            if not isinstance(state, torch.Tensor):
                state = torch.FloatTensor(state).to(self.device)
            
            if state.dim() == 3:
                state = state.unsqueeze(0)
                
            q_values = self.model(state)
            return q_values.argmax().item()