import gymnasium as gym
from env.environment import CarlaEnv

env = gym.make('carla-rl-gym-v0', time_limit=55, initialize_server=False, random_weather=True, synchronous_mode=False, continuous=False, show_sensor_data=False, has_traffic=False, verbose=False)

print("Observation Space: ", env.observation_space)

print("\nAction Space: ", env.action_space.n)
