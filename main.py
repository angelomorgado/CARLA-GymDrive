'''
Main window:
    It acts as the center of the entire program controlling the entire process.
    This depends on the project at hand.
    Various examples of mains that act as tutorials can be found in the examples folder.

    It is possible to call specific scenarios by having a list of them e.g, l = ["Town01-ClearNoon-Road-0", "Town01-WetNight-Road-0"] and then calling them in the reset method as such: env.reset(l[i])
'''

import time
from env.environment import CarlaEnv
import gymnasium as gym

def steps_main():
    # env = CarlaEnv('carla-rl-gym', time_limit=60, initialize_server=False, random_weather=True, synchronous_mode=True, continuous=False, show_sensor_data=True)
    env = gym.make('carla-rl-gym-v0', time_limit=300, initialize_server=False, random_weather=True, synchronous_mode=True, continuous=False, show_sensor_data=True, random_traffic=True)
    obs, info = env.reset()
    
    # Number of steps
    for i in range(300):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print("Reward: ", reward)

        if terminated or truncated:
            observation, info = env.reset()

    env.close()

def episodes_main():
    # env = CarlaEnv(time_limit=50, initialize_server=False, random_weather=True, synchronous_mode=True, continuous=True, show_sensor_data=True)
    env = gym.make('carla-rl-gym-v0', time_limit=10, initialize_server=False, random_weather=True, synchronous_mode=True, continuous=False, show_sensor_data=True, random_traffic=True)

    # Number of episodes
    for i in range(2):
        print("================================ Episode", i, " ================================")
        obs, info = env.reset()
        while True:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            # print("Reward:", reward, "Terminated:", terminated, "Truncated:", truncated)
            
            if terminated or truncated:
                print('Episode terminated cleaning environment')
                break
    env.close()

if __name__ == '__main__':
    episodes_main()
