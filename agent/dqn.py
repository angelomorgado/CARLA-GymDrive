import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from env.aux.point_net import PointNetfeat

class DQNAgent:
    def __init__(self, observation_space, action_space, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, batch_size=32):
        self.observation_space = observation_space
        self.action_space = action_space
        self.memory = []
        self.batch_size = batch_size
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
        
        # Feature extractors
        self.lidar_pointfeat = PointNetfeat(global_feat=True)
        self.lidar_pointfeat.eval()

        self.last_loss = None

    def __build_model(self):
        # Define the input sizes for each observation component
        lidar_size = (1024,)
        position_size = (3,)    
        rgb_size = (360, 640, 3)  # Assuming rgb_data is a 360x640x3 image
        situation_size = 1      # Assuming situation is a single scalar value
        target_position_size = (3,)  # Assuming target_position is a 3-dimensional vector

        # Calculate the total size of the concatenated input
        input_size = np.prod(lidar_size) + np.prod(position_size) + np.prod(rgb_size) + situation_size + np.prod(target_position_size)

        # Define the neural network architecture
        model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_space.n)
        )
        return model.to(self.device)

    def remember(self, state, action, reward, next_state, terminated, truncated):
        self.memory.append((state, action, reward, next_state, terminated, truncated))

    def act(self, state):
        # Epsilon-greedy policy
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space.n)
        
        state = self.__process_state(state)
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
                next_state = self.__process_state(next_state)
                target = reward + self.gamma * torch.max(self.target_model(next_state)).item()

            state = self.__process_state(state)
            q_values = self.model(state)

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

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def train(self, state, action, reward, next_state, terminated, truncated):
        self.remember(state, action, reward, next_state, terminated, truncated)
        self.replay()
        self.update_target_model()

    def get_loss(self):
        return self.last_loss

    def save(self, filename):
        torch.save(self.model.state_dict(), filename)

    def load(self, filename):
        self.model.load_state_dict(torch.load(filename))
        self.target_model.load_state_dict(self.model.state_dict())

    def __process_state(self, state):
        if isinstance(state, dict):
            lidar_data = torch.from_numpy(state['lidar_data']).float()
            lidar_data = lidar_data.unsqueeze(0)
            lidar_data, _, _ = self.lidar_pointfeat(lidar_data)
            lidar_data = lidar_data.to(self.device).squeeze(0)

            position = torch.FloatTensor(state['position']).to(self.device)
            rgb_data = torch.FloatTensor(state['rgb_data'].flatten()).to(self.device)
            situation = torch.FloatTensor([state['situation']]).to(self.device)
            target_position = torch.FloatTensor(state['target_position']).to(self.device)
        elif isinstance(state, tuple) and len(state) == 2:
            # Assuming state is a tuple with the format (state_data, environment_info)
            state_data, environment_info = state
            lidar_data = torch.from_numpy(state_data['lidar_data']).float()
            lidar_data = lidar_data.unsqueeze(0)
            lidar_data, _, _ = self.lidar_pointfeat(lidar_data)
            lidar_data = lidar_data.to(self.device).squeeze(0)

            position = torch.FloatTensor(state_data['position']).to(self.device)
            rgb_data = torch.FloatTensor(state_data['rgb_data'].flatten()).to(self.device)
            situation = torch.FloatTensor([state_data['situation']]).to(self.device)
            target_position = torch.FloatTensor(state_data['target_position']).to(self.device)
        else:
            raise ValueError("Invalid state format")

        concatenated_state = torch.cat((lidar_data, position, rgb_data, situation, target_position))

        return concatenated_state
