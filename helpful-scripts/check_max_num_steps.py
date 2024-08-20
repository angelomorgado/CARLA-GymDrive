'''
This script is used to check the maximum number of steps per episode so the reward function can be adjusted accordingly.
'''

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import gymnasium as gym
import time
import numpy as np
from src.env.environment import CarlaEnv


env = gym.make('carla-rl-gym-v0', time_limit=30, verbose=False, initialize_server=True, random_weather=True, synchronous_mode=True, continuous=True, show_sensor_data=False, has_traffic=False)

episode_durations = []
steps_per_second  = []
total_steps       = []

for i in range(10):
    step_count = 0

    observation, info = env.reset()
    start_time = time.time()
    terminated = False
    truncated = False

    while not terminated and not truncated:
        a = [0.0, 0.0]
        action = env.action_space.sample()  # Replace with your desired action selection logic
        observation, reward, terminated, truncated, info = env.step(a)
        step_count += 1

    end_time = time.time()
    duration = end_time - start_time
    steps_per_second = step_count / duration

    print(f"Episode duration: {duration} seconds")
    print(f"Steps per second: {steps_per_second}")
    print(f"Total steps: {step_count}")
    print("========================================")
    
print("\n\n_________________________________________________")
print("Average steps per second: ", np.mean(steps_per_second))
print("Average total steps: ", np.mean(step_count))
print("Average episode duration: ", np.mean(duration))

env.close()