'''
This script is used to make a custom feature extractor for the stable-baselines3 model.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.preprocessing import is_image_space, get_flattened_obs_dim
import gymnasium as gym

from env.aux.point_net import PointNetfeat

class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        # Initialize the base class with a dummy feature dimension
        super().__init__(observation_space, features_dim=1)

        extractors = {}
        total_concat_size = 0
        
        # Initialize the PointNet features extractor for lidar data
        if "lidar_data" in observation_space.spaces:
            self.lidar_pointfeat = PointNetfeat()
            self.lidar_pointfeat.train()
            total_concat_size += 1024  # Assuming PointNetfeat outputs a 1024-dimensional feature vector
            extractors['lidar_data'] = self.lidar_pointfeat

        # Add more extractors for other observation types if needed

        self.extractors = nn.ModuleDict(extractors)
        self._features_dim = total_concat_size

    def forward(self, observations):
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():
            if key == "lidar_data":
                encoded_tensor_list.append(self.__process_lidar(observations[key]))
            else:
                encoded_tensor_list.append(extractor(observations[key]))

        return torch.cat(encoded_tensor_list, dim=1)

    def __process_lidar(self, lidar_data):        
        # Extract features using PointNet
        out, _, _ = self.lidar_pointfeat(lidar_data)
        
        return out