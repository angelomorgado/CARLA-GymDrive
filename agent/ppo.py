import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=64):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        return self.actor(x), self.critic(x)

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, eps_clip=0.2, clip_value=0.5, batch_size=64,
                 entropy_coef=0.01, max_grad_norm=0.5):
        self.actor_critic = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.clip_value = clip_value
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.last_loss = None

    def save_model(self, filepath):
        torch.save(self.actor_critic.state_dict(), filepath)

    def load_model(self, filepath):
        self.actor_critic.load_state_dict(torch.load(filepath))

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_probs, _ = self.actor_critic(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action.item()

    def train(self, state, action, reward, next_state, terminated, truncated):
        dataset = TensorDataset(state, action, reward, next_state, terminated, truncated)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for state, action, reward, next_state, done, truncated in dataloader:
            action_probs, value = self.actor_critic(state)
            dist = Categorical(action_probs)
            entropy = dist.entropy().mean()

            new_log_prob = dist.log_prob(action.squeeze(-1))
            old_dist = Categorical(torch.exp(new_log_prob.detach()))
            old_log_prob = old_dist.log_prob(action.squeeze(-1))
            ratio = torch.exp(new_log_prob - old_log_prob)
            clipped_ratio = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip)

            advantage = self.__calculate_advantage(state, reward, next_state, done)
            actor_loss = -torch.min(ratio * advantage, clipped_ratio * advantage).mean()
            critic_loss = F.mse_loss(value.squeeze(-1), self.__calculate_td_target(reward, next_state, done))
            loss = actor_loss + critic_loss - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

        self.last_loss = loss.item()

    def __calculate_td_target(self, reward, next_state, terminated):
        with torch.no_grad():
            next_value = self.actor_critic(next_state)[1]
            mask = torch.tensor(1 - terminated, dtype=torch.float32)
            td_target = reward + self.gamma * next_value * mask
        return td_target

    def __calculate_advantage(self, state, reward, next_state, terminated):
        with torch.no_grad():
            next_value = self.actor_critic(next_state)[1]
            mask = torch.tensor(1 - terminated, dtype=torch.float32)
            td_target = reward + self.gamma * next_value * mask
            value = self.actor_critic(state)[1]
            advantage = td_target - value.squeeze(-1)
        return advantage

    def get_loss(self):
        return self.last_loss
