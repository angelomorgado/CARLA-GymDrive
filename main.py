'''
This script is used to show how to use the environment by running a few episodes or steps.
'''
from src.env.environment import CarlaEnv # It is mandatory to import the environment even if it is not used in this script
import gymnasium as gym

def steps_main():
    # env = CarlaEnv('carla-rl-gym-v0', time_limit=300, initialize_server=False, random_weather=True, synchronous_mode=True, continuous=False, show_sensor_data=True, random_traffic=True)  # <-- Alternative way to create the environment
    env = gym.make('carla-rl-gym-v0', time_limit=15, initialize_server=True, random_weather=True, synchronous_mode=True, continuous=False, show_sensor_data=True, random_traffic=True)
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
    env = gym.make('carla-rl-gym-v0', time_limit=15, initialize_server=True, random_weather=True, synchronous_mode=True, continuous=False, show_sensor_data=True, random_traffic=True)

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
