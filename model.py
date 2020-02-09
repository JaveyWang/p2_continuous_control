import random
from collections import namedtuple, deque

import torch
from torch.distributions import Normal
from torch import nn, optim
import numpy as np

UPDATE_EVERY = 1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, state_size=33, action_size=4, seed=2020):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        "*** YOUR CODE HERE ***"

        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_size)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.tanh = nn.Tanh()
    
    def forward(self, states):
        """Build a network that maps state -> action values."""
        x = self.fc3(self.relu(self.fc2(self.relu(self.fc1(states)))))
        return self.tanh(x)

    
class Critic(nn.Module):
    def __init__(self, state_size=33, action_size=4, seed=2020):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256+action_size, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, states, actions):
        """Build a network that maps states -> values."""
        x_cat = torch.cat([self.fc1(states), actions], dim=1)
        x = self.relu(self.fc2(x_cat))
        x = self.relu(self.fc3(x))
        return self.fc4(x)
    
class Agent(nn.Module):
    def __init__(self, state_size=33, action_size=4, buffer_size=20000, batch_size=128, lr=1e-4, gamma=0.99, pre_trained=False, seed=2020):
        super(Agent, self).__init__()
        self.batch_size = batch_size
        self.gamma = gamma
        self.t_step = 0
        self.pre_trained = pre_trained
        
        self.actor_local = Actor(state_size=state_size, action_size=action_size, seed=seed).to(device)
        self.actor_target = Actor(state_size=state_size, action_size=action_size, seed=seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=lr)

        self.critic_local = Critic(state_size=state_size, action_size=action_size, seed=seed).to(device)
        self.critic_target = Critic(state_size=state_size, action_size=action_size, seed=seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=3 * lr)
        
        if pre_trained:
            ckpt = torch.load('DDPG.ckpt')
            self.actor_target.load_state_dict(ckpt['actor_target'])
            self.critic_target.load_state_dict(ckpt['critic_target'])
            print('Load model successful')

        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, seed)
        
    def act(self, states):
        states = torch.from_numpy(states).float().to(device)
        with torch.no_grad():
            if self.pre_trained:
                mu = self.actor_target(states)
            else:
                mu = self.actor_local(states)
            normal = Normal(mu, 0.1)
            actions = torch.clamp(normal.sample(), -1, 1)
        return actions.cpu().data.numpy()
    
    def step(self, states, actions, rewards, next_states, dones):
        for s, a, r, s_prime, done in zip(states, actions, rewards, next_states, dones):
            self.memory.add(s, a, r, s_prime, done)
            
        self.t_step += 1
        if self.t_step % UPDATE_EVERY == 0:
            if len(self.memory) > self.batch_size:
                self.learn()
        
    def learn(self):
        # DDPG implementation
        experiences = self.memory.sample()
        states, actions, rewards, next_states, dones = experiences
        
        # update critic
        q_value = self.critic_local(states, actions)
        with torch.no_grad():
            actions_next = self.actor_target(next_states)
            q_value_prime = self.critic_target(next_states, actions_next)
        td_error = rewards + self.gamma * q_value_prime * (1 - dones) - q_value  # one-step estimate
        
        self.critic_optimizer.zero_grad()
        (td_error ** 2).mean().backward()
        self.critic_optimizer.step()
        self.soft_update(self.critic_local, self.critic_target)
        
        # update actor
        actions_pred = self.actor_local(states)
        actor_loss = - self.critic_target(states, actions_pred).mean()  # Use target network can stablize the learning process.
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.soft_update(self.actor_local, self.actor_target)                
        
    def soft_update(self, local_model, target_model, tau=0.001):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
            
            
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)