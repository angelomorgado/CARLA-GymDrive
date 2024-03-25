'''
Pre-processing Module:
    - This module is used to preprocess the observation data before feeding it to the policy network
'''
import numpy as np
from env.aux.farthest_sampler import FarthestSampler
from env.aux.point_net import PointNetfeat
import torch

class PreProcessing:
    def __init__(self) -> None:
        self.sampler = FarthestSampler()
        self.pointfeat = PointNetfeat(global_feat=True)
        self.pointfeat = self.pointfeat.eval()
    
    def preprocess_data(self, observation_data):
        observation_data['lidar_data'] = self.__process_lidar(observation_data['lidar_data'])
        return observation_data

    # This method extracts the features from the lidar data before feeding it to the policy network
    def __process_lidar(self, lidar_data):
        lidar_data = lidar_data[:, :-1]
        lidar_data = lidar_data.transpose([1, 0])
        
        # Sample the lidar data so the number of points remains constant without affecting the quality of the data
        sampler = FarthestSampler()
        lidar_data, _ = sampler.sample(lidar_data, 500)
        
        return np.float32(lidar_data)
        