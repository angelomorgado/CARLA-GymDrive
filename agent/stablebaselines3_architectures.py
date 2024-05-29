import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.preprocessing import is_image_space, get_flattened_obs_dim
import gymnasium as gym
from gymnasium import spaces
import cv2
import numpy as np

# ================== DQN ==================
class CustomExtractor_DQN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, end_to_end: bool = False):
        # Compute the combined feature dimension
        image_dim = 1280  # Dimensionality of the EfficientNet features
        rest_dim = 256    # Dimensionality of the rest features
        features_dim = image_dim + rest_dim
        
        super().__init__(observation_space, features_dim=features_dim)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load EfficientNet model from NVIDIA Torch Hub
        self.efficientnet = None
        if not end_to_end:
            self.efficientnet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)
            self.efficientnet = nn.Sequential(*list(self.efficientnet.children())[:-1])
            self.efficientnet.eval()
            for param in self.efficientnet.parameters():
                param.requires_grad = False  # Freeze the EfficientNet parameters
        else:
            self.efficientnet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=False)
            self.efficientnet = nn.Sequential(*list(self.efficientnet.children())[:-1])
            self.efficientnet.train()

        # Add adaptive average pooling to reduce the spatial dimensions to 1x1
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # Rest: 6 -> 256 (position and target_position)
        self.model3 = nn.Sequential(
            nn.Linear(6, 256),
        )

        self.final_model = nn.Sequential(
            nn.Linear(features_dim, 512),  # Combine image and rest features, output 512-dimensional vector
            nn.ReLU(),
            nn.Linear(512, 4)  # Discrete actions
        )

    def forward(self, observations):
        rgb_input, rest_input = self.process_observations(observations)
        
        image_features = self.efficientnet(rgb_input)
        image_features = self.global_avg_pool(image_features)
        image_features = torch.flatten(image_features, 1)
        
        rest_output = self.model3(rest_input)
        
        if len(rest_output.shape) == 1:
            rest_output = rest_output.unsqueeze(0)
        
        combined_features = torch.cat((image_features, rest_output), dim=1)
        
        return combined_features
        # return self.final_model(combined_features)
    
    def process_observations(self, observations):
        rgb_data = F.interpolate(observations['rgb_data'], size=(224, 224), mode='bilinear', align_corners=False)
        # Normalize the pixel values to be in the range [0, 1]
        rgb_data = cv2.normalize(rgb_data.cpu().numpy(), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        rgb_data = torch.from_numpy(rgb_data).float().to(self.device)

        position = observations['position'].squeeze()
        target_position = observations['target_position'].squeeze()
        if len(position.shape) == 1:
            position = position.unsqueeze(0)
            target_position = target_position.unsqueeze(0)
        rest = torch.cat((position,target_position),dim=1).to(self.device)
      
        return (rgb_data, rest)

# ================== PPO ==================
class CustomExtractor_PPO(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, end_to_end: bool = False):
        # Compute the combined feature dimension
        image_dim = 1280  # Dimensionality of the EfficientNet features
        rest_dim = 256   # Dimensionality of the rest features
        features_dim = image_dim + rest_dim
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        super().__init__(observation_space, features_dim=features_dim)
        self.action_dim = continuous_action_space.shape[0]  # Dimensionality of the action space

        # Load EfficientNet model from NVIDIA Torch Hub
        self.efficientnet = None
        if not end_to_end:
            self.efficientnet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)
            self.efficientnet = nn.Sequential(*list(self.efficientnet.children())[:-1])
            self.efficientnet.eval()
            for param in self.efficientnet.parameters():
                param.requires_grad = False  # Freeze the EfficientNet parameters
        else:
            self.efficientnet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=False)
            self.efficientnet = nn.Sequential(*list(self.efficientnet.children())[:-1])
            self.efficientnet.train()

        # Add adaptive average pooling to reduce the spatial dimensions to 1x1
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # Define the neural network architecture for processing the rest of the input
        # 3 + 3 = 6
        self.model3 = nn.Sequential(
            nn.Linear(6, 256),
        )

        # Define the final fully connected layers for the policy head
        self.policy_head_mean = nn.Linear(features_dim, self.action_dim)
        self.policy_head_std = nn.Linear(features_dim, self.action_dim)

    def forward(self, observations):
        rgb_input, rest_input = self.process_observations(observations)
        
        # image_features = self.model1(rgb_input)
        # image_features = torch.squeeze(image_features)  # Remove dummy dimensions
        image_features = self.efficientnet(rgb_input)
        image_features = self.global_avg_pool(image_features)
        image_features = torch.flatten(image_features, 1).squeeze(0)
        
        rest_output = self.model3(rest_input)
        rest_output = torch.squeeze(rest_output)  # Remove dummy dimensions
        # print("image_features:", image_features.shape)
        # print("rest_output:", rest_output.shape)
        if len(rest_output.shape) == 1:
            combined_features = torch.cat((image_features, rest_output), dim=0).unsqueeze(0)
        else:
            combined_features = torch.cat((image_features, rest_output), dim=1)
        # print("combined_features:", combined_features.shape)
        # print("====================================")
        return combined_features

        # Policy head for continuous action space
        # mean = torch.tanh(self.policy_head_mean(combined_features))  # Ensure mean is within [-1, 1]
        # print("mean:", mean.shape)
        # std = F.softplus(self.policy_head_std(combined_features)) + 1e-5  # Standard deviation must be positive
        # print("std:", std.shape)
        # # Concatenate mean and std along a new dimension
        # policy_output = torch.cat((mean.unsqueeze(0), std.unsqueeze(0)), dim=1)
        
        # return policy_output

    def process_observations(self, observations):
        rgb_data = F.interpolate(observations['rgb_data'], size=(224, 224), mode='bilinear', align_corners=False)
        # Normalize the pixel values to be in the range [0, 1]
        rgb_data = cv2.normalize(rgb_data.cpu().numpy(), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        rgb_data = torch.from_numpy(rgb_data).float().to(self.device)

        position = observations['position'].squeeze()
        target_position = observations['target_position'].squeeze()
        if len(position.shape) == 1:
            position = position.unsqueeze(0)
            target_position = target_position.unsqueeze(0)
        rest = torch.cat((position,target_position),dim=1).to(self.device)
      
        return (rgb_data, rest)


# Define the continuous action space for PPO
continuous_action_space = spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)

# Define the discrete action space for DQN
discrete_action_space = spaces.Discrete(4)
