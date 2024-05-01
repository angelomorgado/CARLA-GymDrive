import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from env.env_aux.point_net import PointNetfeat

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
        self.model1, self.model2, self.model3, self.final_model = self.__build_model()
        self.target_model1, self.target_model2, self.target_model3, self.target_final_model = self.__build_model()
        self.update_target_model()
        self.all_models = nn.ModuleList([self.model1, self.model2, self.model3, self.final_model])
        self.optimizer = Adam(self.all_models.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()
        
        # Feature extractors
        #   Lidar PointNet
        self.lidar_pointfeat = PointNetfeat(global_feat=True)
        self.lidar_pointfeat.eval().to(self.device)
        #   RGB ResNet50
        self.resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
        self.resnet50.eval().to(self.device)
        self.resnet50 = nn.Sequential(*list(self.resnet50.children())[:-1])

        self.last_loss = None

    def __build_model(self):
        # Define the input sizes for each observation component
        lidar_size = (1024,)
        position_size = (3,)    
        rgb_size = (2048,) #(360, 640, 3)  # Assuming rgb_data is a 360x640x3 image
        situation_size = 1      # Assuming situation is a single scalar value
        target_position_size = (3,)  # Assuming target_position is a 3-dimensional vector
        
        # 3 mlp: 1 for lidar, 1 for rgb, and 1 for the rest, then join them in a fourth mlp
        # Calculate the total size of the concatenated input
        input_size = np.prod(lidar_size) + np.prod(position_size) + np.prod(rgb_size) + situation_size + np.prod(target_position_size)
        print(f"Input size: {input_size}")

        # Define the neural network architecture
        # RGB: 2048
        model1 = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512)
        )
        
        # Lidar: 1024
        model2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        
        # Rest: 7
        model3 = nn.Sequential(
            nn.Linear(7, 256),
        )
        
        # Join the three models: 512 + 256 + 256 = 1024
        final_model = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.action_space.n)
        )
        
        return model1.to(self.device), model2.to(self.device), model3.to(self.device), final_model.to(self.device)

    def update_target_model(self):
        self.target_model1.load_state_dict(self.model1.state_dict())
        self.target_model2.load_state_dict(self.model2.state_dict())
        self.target_model3.load_state_dict(self.model3.state_dict())
        self.target_final_model.load_state_dict(self.final_model.state_dict())

    def remember(self, state, action, reward, next_state, terminated, truncated):
        self.memory.append((state, action, reward, next_state, terminated, truncated))

    def act(self, state):
        # Epsilon-greedy policy
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space.n)
        
        return self.forward(state)

    def forward(self, state):
        states = self.__process_state(state)
        
        # Model1, model2, model3, and final_model blah blah
        rgb_value = self.model1(states[0])
        lidar_value = self.model2(states[1])
        rest_value = self.model3(states[2])
        
        q_values = self.final_model(torch.cat((rgb_value, lidar_value, rest_value)))
        
        return torch.argmax(q_values).item()
    
    def target_forward(self, state):
        states = self.__process_state(state)
        
        rgb_value = self.target_model1(states[0])
        lidar_value = self.target_model2(states[1])
        rest_value = self.target_model3(states[2])
        
        q_values = self.target_final_model(torch.cat((rgb_value, lidar_value, rest_value)))
        
        return q_values

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states, targets = [], []
        for state, action, reward, next_state, terminated, truncated in minibatch:
            target = reward
            if not terminated and not truncated:
                next_states = self.__process_state(next_state)  # Assuming states in memory are already processed
                print(next_states)
                exit()
                target = reward + self.gamma * torch.max(self.target_forward(next_states)).item()

            processed_states = state  # Assuming states in memory are already processed

            rgb_value = self.model1(processed_states[0])
            lidar_value = self.model2(processed_states[1])
            rest_value = self.model3(processed_states[2])
            
            q_values = self.final_model(torch.cat((rgb_value, lidar_value, rest_value)))

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
        if isinstance(state, tuple):
            # If state is a tuple, assume it contains the necessary information
            states, environment_info = state
            rgb_data, lidar_data, position, situation, target_position = states

            # Convert lidar_data and rgb_data to tensors
            lidar_data = torch.from_numpy(lidar_data).float().to(self.device)
            rgb_data = torch.from_numpy(rgb_data).float().to(self.device)

        elif isinstance(state, dict):
            # If state is a dictionary, extract the relevant information
            rgb_data = torch.from_numpy(state['rgb_data']).float().to(self.device)
            lidar_data = torch.from_numpy(state['lidar_data']).float().to(self.device)
            position = torch.FloatTensor(state['position']).to(self.device)
            situation = torch.FloatTensor([state['situation']]).to(self.device)
            target_position = torch.FloatTensor(state['target_position']).to(self.device)

        else:
            raise ValueError("Invalid state format")

        lidar_data = lidar_data.unsqueeze(0)
        lidar_data, _, _ = self.lidar_pointfeat(lidar_data)
        lidar_data = lidar_data.squeeze(0)
        
        # RGB (2048,)
        # Assuming rgb_data is a numpy array with shape (360, 640, 3)
        rgb_data = rgb_data.transpose(0, 2)  # Transpose to (3, 360, 640)
        rgb_features = self.resnet50(rgb_data.unsqueeze(0)).squeeze()  # Reshape to (1, 2048) and squeeze to (2048,)

        rest = torch.cat((position, situation, target_position)).to(self.device)

        return (rgb_features, lidar_data, rest)