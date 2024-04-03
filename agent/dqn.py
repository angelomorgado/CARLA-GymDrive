import math
import random
from collections import namedtuple, deque
from itertools import count

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from env.aux.point_net import PointNetfeat

class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, plot_graph=True):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.batch_size = 32
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Neural Network for Q-value approximation
        self.model = self.__build_model()
        self.target_model = self.__build_model()
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

        self.last_loss = None
            

    def __build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_size)
        )
        return model.to(self.device)

    def remember(self, state, action, reward, next_state, terminated, truncated):
        self.memory.append((state, action, reward, next_state, terminated, truncated))

    def act(self, state):
        # Epsilon-greedy policy
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        try:
            state = np.array(state)
        except ValueError:
            state = np.array(state[0])
        state = torch.FloatTensor(state).to(self.device)
        q_values = self.model(state)
        return torch.argmax(q_values).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
            
        minibatch = random.sample(self.memory, self.batch_size)
        states, targets = [], []
        for state, action, reward, next_state, terminated, truncated in minibatch:
            target = reward
            if not terminated and not truncated:
                next_state = torch.FloatTensor(next_state).to(self.device)
                target = reward + self.gamma * torch.max(self.target_model(next_state)).item()
            
            # Compute Q-values for the current state
            try:
                state = np.array(state)
            except ValueError:
                state = np.array(state[0])
            state = torch.FloatTensor(state).to(self.device)
            q_values = self.model(state)
            
            # Update the Q-value for the chosen action towards the target
            q_values[action] = target
            
            states.append(state)
            targets.append(q_values)

        states = torch.stack(states)
        targets = torch.stack(targets)

        self.optimizer.zero_grad()
        loss = self.loss_fn(self.model(states), targets)
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.last_loss = loss.item()
    
    # def replay(self):
    #     if len(self.memory) < self.batch_size:
    #         return
        
    #     minibatch = random.sample(self.memory, self.batch_size)
    #     states, targets = [], []
    #     for state, action, reward, next_state, terminated, truncated in minibatch:
    #         target = reward
    #         if not terminated and not truncated:
    #             next_state = torch.FloatTensor(next_state).to(self.device)
    #             target = reward + self.gamma * torch.max(self.target_model(next_state)).item()

    #         try:
    #             state = np.array(state)
    #         except ValueError:
    #             state = np.array(state[0])
    #         state = torch.FloatTensor(state).to(self.device)
    #         q_values = self.model(state)
    #         q_values[action] = target
    #         states.append(state)
    #         targets.append(q_values)

    #     states = torch.stack(states)
    #     targets = torch.stack(targets)

    #     self.optimizer.zero_grad()
    #     loss = self.loss_fn(self.model(states), targets)
    #     loss.backward()
    #     self.optimizer.step()

    #     if self.epsilon > self.epsilon_min:
    #         self.epsilon *= self.epsilon_decay

    #     self.last_loss = loss.item()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
        
    def get_loss(self):
        return self.last_loss  

    def save(self, filename):
        torch.save(self.model.state_dict(), filename)

    def load(self, filename):
        self.model.load_state_dict(torch.load(filename))
        self.target_model.load_state_dict(self.model.state_dict())
