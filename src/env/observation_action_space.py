from gymnasium import spaces
import numpy as np

# Change this according to your needs.
observation_shapes = {
    'rgb_data': (360, 640, 3),
    'lidar_data': (3, 500), 
    'position': (3,),
    'target_position': (3,),
    'next_waypoint_position': (3,),
    'speed': (1,),
    'num_of_stuations': 4
}

situations_map = {
    "Road": 0,
    "Roundabout": 1,
    "Junction": 2,
    "Tunnel": 3
}

observation_space = spaces.Dict({
    'rgb_data': spaces.Box(low=0, high=255, shape=observation_shapes['rgb_data'], dtype=np.uint8),
    'lidar_data': spaces.Box(low=-np.inf, high=np.inf, shape=observation_shapes['lidar_data'], dtype=np.float32),
    'position': spaces.Box(low=-np.inf, high=np.inf, shape=observation_shapes['position'], dtype=np.float32),
    'target_position': spaces.Box(low=-np.inf, high=np.inf, shape=observation_shapes['target_position'], dtype=np.float32),
    'next_waypoint_position': spaces.Box(low=-np.inf, high=np.inf, shape=observation_shapes['next_waypoint_position'], dtype=np.float32),
    'speed': spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
    'situation': spaces.Discrete(observation_shapes['num_of_stuations'])
})

# For continuous actions (steering [-1.0, 1.0], throttle/brake [-1.0, 1.0])
continuous_action_space = spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)

# For discrete actions (accelerate, deaccelerate, turn left, turn right)
discrete_action_space = spaces.Discrete(4)
