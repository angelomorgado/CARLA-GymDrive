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
class CustomExtractor_DQN_End2end(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        # Compute the combined feature dimension
        image_dim = 512  # Dimensionality of the EfficientNet features
        rest_dim = 128   # Dimensionality of the rest features
        features_dim = image_dim + rest_dim
        
        super().__init__(observation_space, features_dim=features_dim)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Custom CNN for processing the RGB data
        self.image_model = nn.Sequential(
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
        self.rest_model = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )

    def forward(self, observations):
        rgb_input, rest_input = self.process_observations(observations)
        
        image_features = self.image_model(rgb_input)
        image_features = torch.flatten(image_features, 1)

        rest_output = self.rest_model(rest_input)
                
        if len(rest_output.shape) == 1:
            rest_output = rest_output.unsqueeze(0)
        
        combined_features = torch.cat((image_features, rest_output), dim=1)
        
        return combined_features
    
    def process_observations(self, observations):
        rgb_data = F.interpolate(observations['rgb_data'], size=(224, 224), mode='bilinear', align_corners=False)
        rgb_data = rgb_data / 255.0  # Normalize the pixel values to be in the range [0, 1]

        return (rgb_data.float().to(self.device), observations['rest'])

class CustomExtractor_DQN_Modular(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, end_to_end: bool = False):
        # Compute the combined feature dimension
        image_dim = 1280  # Dimensionality of the EfficientNet features
        rest_dim = 256    # Dimensionality of the rest features
        features_dim = image_dim + rest_dim
        
        super().__init__(observation_space, features_dim=features_dim)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load EfficientNet model from NVIDIA Torch Hub
        self.efficientnet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)
        self.efficientnet = nn.Sequential(*list(self.efficientnet.children())[:-1])
        self.efficientnet.eval()
        # Freeze the EfficientNet parameters
        for param in self.efficientnet.parameters():
            param.requires_grad = False  

        # Add adaptive average pooling to reduce the spatial dimensions to 1x1
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # Define the neural network architecture for processing the rest of the input
        # 3 -> 256
        self.rest_model = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU()
        )

    def forward(self, observations):
        rgb_input, rest_input = self.process_observations(observations)
        
        rest_output = self.rest_model(rest_input)
                
        if len(rest_output.shape) == 1:
            rest_output = rest_output.unsqueeze(0)

        combined_features = torch.cat((rgb_input, rest_output), dim=1)
        return combined_features
    
    def process_observations(self, observations):
        rgb_data = F.interpolate(observations['rgb_data'], size=(224, 224), mode='bilinear', align_corners=False)
        rgb_data = rgb_data / 255.0  # Normalize the pixel values to be in the range [0, 1]

        rgb_data = torch.from_numpy(rgb_data).float().to(self.device)
        rgb_data = self.efficientnet(rgb_data)
        rgb_data = self.global_avg_pool(rgb_data)
        rgb_data = torch.flatten(rgb_data, 1).squeeze(0)

        return (rgb_data, observations['rest'])

# ================== PPO ==================
class CustomExtractor_PPO_End2end(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict):
        image_dim = 512  # Dimensionality of the CNN features
        rest_dim = 128   # Dimensionality of the rest features
        features_dim = image_dim + rest_dim
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        super().__init__(observation_space, features_dim=features_dim)
        self.action_dim = continuous_action_space.shape[0]  # Dimensionality of the action space

        # Custom CNN for processing the RGB data
        self.image_model = nn.Sequential(
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
        self.rest_model = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )

    def forward(self, observations):
        rgb_input, rest_input = self.process_observations(observations)
        
        image_features = self.image_model(rgb_input)
        image_features = torch.flatten(image_features, 1)

        rest_output = self.rest_model(rest_input)
                
        if len(rest_output.shape) == 1:
            rest_output = rest_output.unsqueeze(0)

        combined_features = torch.cat((image_features, rest_output), dim=1)
        return combined_features

    def process_observations(self, observations):
        rgb_data = F.interpolate(observations['rgb_data'], size=(224, 224), mode='bilinear', align_corners=False)
        rgb_data = rgb_data / 255.0  # Normalize the pixel values to be in the range [0, 1]

        return (rgb_data.float().to(self.device), observations['rest'])


class CustomExtractor_PPO_Modular(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict):
        # Compute the combined feature dimension
        image_dim = 1280  # Dimensionality of the EfficientNet features
        rest_dim = 256   # Dimensionality of the rest features
        features_dim = image_dim + rest_dim
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        super().__init__(observation_space, features_dim=features_dim)
        self.action_dim = continuous_action_space.shape[0]  # Dimensionality of the action space

        # Load EfficientNet model from NVIDIA Torch Hub
        self.efficientnet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)
        self.efficientnet = nn.Sequential(*list(self.efficientnet.children())[:-1])
        self.efficientnet.eval()
        # Freeze the EfficientNet parameters
        for param in self.efficientnet.parameters():
            param.requires_grad = False  

        # Add adaptive average pooling to reduce the spatial dimensions to 1x1
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # Define the neural network architecture for processing the rest of the input
        # 3 -> 256
        self.rest_model = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU()
        )

    def forward(self, observations):
        rgb_input, rest_input = self.process_observations(observations)
        
        rest_output = self.rest_model(rest_input)
                
        if len(rest_output.shape) == 1:
            rest_output = rest_output.unsqueeze(0)

        combined_features = torch.cat((rgb_input, rest_output), dim=1)
        return combined_features

    def process_observations(self, observations):
        rgb_data = F.interpolate(observations['rgb_data'], size=(224, 224), mode='bilinear', align_corners=False)
        rgb_data = rgb_data / 255.0  # Normalize the pixel values to be in the range [0, 1]

        rgb_data = torch.from_numpy(rgb_data).float().to(self.device)
        rgb_data = self.efficientnet(rgb_data)
        rgb_data = self.global_avg_pool(rgb_data)
        rgb_data = torch.flatten(rgb_data, 1).squeeze(0)

        return (rgb_data, observations['rest'])

# Define the continuous action space for PPO
continuous_action_space = spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)

# Define the discrete action space for DQN
discrete_action_space = spaces.Discrete(4)
