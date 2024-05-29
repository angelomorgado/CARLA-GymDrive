import numpy as np
import random
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal
from collections import namedtuple
import cv2
import os

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'log_prob', 'value'))

class RolloutBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.transitions = []

    def add_transition(self, transition):
        self.transitions.append(transition)

    def sample(self):
        return self.transitions

class ActorCritic(nn.Module):
    def __init__(self, output_n, action_dim, action_std_init):
        super(ActorCritic, self).__init__()

        self.efficientnet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)
        self.efficientnet = nn.Sequential(*list(self.efficientnet.children())[:-1])
        self.efficientnet.eval()
        for param in self.efficientnet.parameters():
            param.requires_grad = False

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        self.rest_model = nn.Sequential(
            nn.Linear(6, 256),
        )

        self.actor = nn.Sequential(
            nn.Linear(1280 + 256, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
            nn.Tanh()
        )

        self.critic = nn.Sequential(
            nn.Linear(1280 + 256, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self.action_dim = action_dim
        self.action_var = torch.full((action_dim,), action_std_init * action_std_init)

    def forward(self, state):
        rgb_input, rest_input = state

        if len(rgb_input.shape) == 3:
            rgb_input = rgb_input.unsqueeze(0)
        if len(rest_input.shape) == 1:
            rest_input = rest_input.unsqueeze(0)

        image_features = self.efficientnet(rgb_input)
        image_features = self.global_avg_pool(image_features)
        image_features = torch.flatten(image_features, 1)
        rest_output = self.rest_model(rest_input)

        combined_features = torch.cat((image_features, rest_output), dim=-1)

        action_mean = self.actor(combined_features)
        state_value = self.critic(combined_features)

        return action_mean, state_value

class PPOAgent:
    def __init__(self, environment_name=None, lr_actor=3e-4, lr_critic=1e-3, gamma=0.99, K_epochs=80, eps_clip=0.2, action_std_init=0.6, env=None):
        if env is None:
            self.env = gym.make(environment_name)
        else:
            self.env = env

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_dim = self.env.action_space.shape[0]
        self.buffer = RolloutBuffer()
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = ActorCritic(self.env.action_space.shape[0], self.action_dim, action_std_init).to(self.device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])
        self.policy_old = ActorCritic(self.env.action_space.shape[0], self.action_dim, action_std_init).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def select_action(self, state):
        rgb_state, rest_state = self.process_state(state)
        with torch.no_grad():
            action_mean, state_value = self.policy_old((rgb_state, rest_state))
            cov_mat = torch.diag(self.policy.action_var).unsqueeze(dim=0).to(self.device)
            dist = MultivariateNormal(action_mean, cov_mat)
            action = dist.sample()
            action_logprob = dist.log_prob(action)

        self.buffer.add_transition(Transition((rgb_state, rest_state), action, None, None, action_logprob, state_value))

        return action.cpu().numpy().flatten()

    def process_state(self, state):
        rgb_data = cv2.resize(state['rgb_data'], (224, 224))
        rgb_data = np.transpose(rgb_data, (2, 0, 1))
        rgb_data = torch.from_numpy(rgb_data).float().to(self.device)
        position = torch.FloatTensor(state['position']).to(self.device)
        target_position = torch.FloatTensor(state['target_position']).to(self.device)
        rest = torch.cat((position, target_position)).to(self.device)
        return (rgb_data, rest)

    def update(self):
        transitions = self.buffer.sample()
        batch = Transition(*zip(*transitions))

        rewards = []
        discounted_reward = 0
        for reward in reversed(batch.reward):
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        old_states = (
            torch.stack([s[0] for s in batch.state]).detach().to(self.device),
            torch.stack([s[1] for s in batch.state]).detach().to(self.device)
        )
        old_actions = torch.stack(batch.action).detach().to(self.device)
        old_logprobs = torch.stack(batch.log_prob).detach().to(self.device)
        old_state_values = torch.stack(batch.value).detach().to(self.device)

        advantages = rewards.detach() - old_state_values.detach()

        for _ in range(self.K_epochs):
            action_mean, state_values = self.policy(old_states)
            cov_mat = torch.diag(self.policy.action_var).unsqueeze(dim=0).to(self.device)
            dist = MultivariateNormal(action_mean, cov_mat)

            logprobs = dist.log_prob(old_actions)
            dist_entropy = dist.entropy()
            state_values = state_values.squeeze()

            ratios = torch.exp(logprobs - old_logprobs.detach())

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()

    def train(self, max_episodes=1000):
        for episode in range(max_episodes):
            state, _ = self.env.reset()
            while True:
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                reward = torch.tensor([reward], device=self.device)

                self.buffer.transitions[-1] = self.buffer.transitions[-1]._replace(next_state=self.process_state(next_state), reward=reward)

                state = next_state

                if terminated or truncated:
                    break

            self.update()
            print(f"Episode {episode} completed")
    
    def test(self):
        # Test the agent for 1 episode and return the total reward
        state, _ = self.env.reset()
        rewards = []
        
        while True:
            action = self.select_action(state)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            rewards.append(reward)
            state = next_state

            if terminated or truncated:
                break
        
        return np.sum(rewards)
        
    def save_model(self, filename):
        torch.save(self.policy_old.state_dict(), filename)

    def load_model(self, filename):
        self.policy_old.load_state_dict(torch.load(filename))
        self.policy.load_state_dict(torch.load(filename))
