'''
This script is used to make a custom feature extractor for the stable-baselines3 model.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.preprocessing import is_image_space, get_flattened_obs_dim
import gymnasium as gym
from gymnasium import spaces
import cv2
import numpy as np

from env.env_aux.point_net import PointNetfeat
from env.observation_action_space import continuous_action_space as action_space

# ================== DQN ==================
class CustomExtractor_DQN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        # Compute the combined feature dimension
        image_dim = 512  # Dimensionality of the image features
        rest_dim = 256   # Dimensionality of the rest features
        features_dim = image_dim + rest_dim
        
        super().__init__(observation_space, features_dim=features_dim)
        
        # Define the neural network architecture
        # RGB: (224, 224) -> 512
        self.model1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)  # Global average pooling to get a fixed-size feature vector
        )

        # Rest: 7
        self.model3 = nn.Sequential(
            nn.Linear(7, 256),
        )

        self.final_model = nn.Sequential(
            nn.Linear(features_dim, 512),  # Combine image and rest features, output 512-dimensional vector
            nn.ReLU(),
            nn.Linear(512, 4)
        )

    def forward(self, observations):
        rgb_input, rest_input = self.process_observations(observations)
        
        image_features = self.model1(rgb_input)
        image_features = torch.squeeze(image_features).unsqueeze(0)  # Remove dummy dimensions
        rest_output = self.model3(rest_input).unsqueeze(0)
        combined_features = torch.cat((image_features, rest_output), dim=1)
        return self.final_model(combined_features)

    def process_observations(self, state):
        # Resize the RGB image to 224x224
        rgb_data = cv2.resize(state['rgb_data'], (224, 224))
        rgb_data = cv2.normalize(rgb_data, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        rgb_data = torch.from_numpy(rgb_data).float().to(self.device)
        position = torch.FloatTensor(state['position']).to(self.device)
        situation = torch.FloatTensor([state['situation']]).to(self.device)
        target_position = torch.FloatTensor(state['target_position']).to(self.device)

        rest = torch.cat((position, situation, target_position)).to(self.device)

        return (rgb_data, rest)
    
# ================== PPO ==================
class CustomExtractor_PPO(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict):
        # Compute the combined feature dimension
        image_dim = 512  # Dimensionality of the image features
        rest_dim = 256   # Dimensionality of the rest features
        features_dim = image_dim + rest_dim
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        super().__init__(observation_space, features_dim=features_dim)
        self.action_dim = action_space.shape[0]  # Dimensionality of the action space

        # Define the neural network architecture for processing the image
        self.model1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)  # Global average pooling to get a fixed-size feature vector
        )

        # Define the neural network architecture for processing the rest of the input
        # 3 + 3 + 4 = 7
        self.model3 = nn.Sequential(
            nn.Linear(10, 256),
        )

        # Define the final fully connected layers for the policy head
        self.policy_head_mean = nn.Linear(features_dim, self.action_dim)
        self.policy_head_std = nn.Linear(features_dim, self.action_dim)

    def forward(self, observations):
        rgb_input, rest_input = self.process_observations(observations)
        
        image_features = self.model1(rgb_input)
        image_features = torch.squeeze(image_features)  # Remove dummy dimensions
        rest_output = self.model3(rest_input)
        rest_output = torch.squeeze(rest_output)  # Remove dummy dimensions
        print("image_features:", image_features.shape)
        print("rest_output:", rest_output.shape)
        combined_features = torch.cat((image_features, rest_output), dim=0).unsqueeze(0)
        print("combined_features:", combined_features.shape)

        # Policy head for continuous action space
        mean = torch.tanh(self.policy_head_mean(combined_features))  # Ensure mean is within [-1, 1]
        print("mean:", mean.shape)
        std = F.softplus(self.policy_head_std(combined_features)) + 1e-5  # Standard deviation must be positive
        print("std:", std.shape)
        # Concatenate mean and std along a new dimension
        policy_output = torch.cat((mean.unsqueeze(0), std.unsqueeze(0)), dim=1)
        
        return policy_output

    def process_observations(self, observations):
        # Check if the first element of the dictionary is a tensor
        first_key = next(iter(observations.keys()))
        if isinstance(observations[first_key], torch.Tensor):
            # Convert all elements into NumPy arrays
            observations = {key: value.squeeze().cpu().numpy() for key, value in observations.items()}
            # Transpose the first value to the end to match the expected input shape
            # observations['rgb_data'] = observations['rgb_data'].squeeze().cpu().numpy()
            observations['rgb_data'] = np.transpose(observations['rgb_data'], (1, 2, 0))
        
        # print(observations['situation'], "AAAAAAAA")
        # print(observations['position'], "AAAAAAAA")
        # print(observations['rgb_data'])
        # print(observations['rgb_data'].shape)
                
        # Resize the RGB image to 224x224
        rgb_data = cv2.resize(observations['rgb_data'], (224, 224))
        rgb_data = np.transpose(rgb_data, (2, 0, 1))
        # Normalize the pixel values to be in the range [0, 1]
        rgb_data = cv2.normalize(rgb_data, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        rgb_data = torch.from_numpy(rgb_data).float().unsqueeze(0).to(self.device)  # Add batch dimension

        position = torch.FloatTensor(observations['position'])
        situation = torch.FloatTensor(observations['situation'])
        target_position = torch.FloatTensor(observations['target_position'])

        # print name and shapes
        # print('position:', position.shape)
        # print('situation:', situation.shape)
        # print('target_position:', target_position.shape)
        
        rest = torch.cat((position, situation, target_position)).unsqueeze(0).to(self.device)
        # print('rest:', rest.shape)
        

        return (rgb_data, rest)
