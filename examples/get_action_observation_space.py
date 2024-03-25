'''
get_action_observation_space.py
 - This script is used to get the action and observation space of the environment.
'''
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from env.environment import CarlaEnv

def main():
    env = CarlaEnv(initialize_server=True, has_traffic=False, verbose=False)
    obs, info = env.reset("Town01-ClearNoon-Road-0")

    obs_shape = []
    for o in obs:
        try:
            obs_shape.append(obs[o].shape)
        except AttributeError:
            obs_shape.append(obs[o])

    print("==============================================================")
    print("Action Space:", env.action_space)
    print("Defined observation Space:", env.observation_space)
    print("Actual observation Space:", obs_shape)
    print("==============================================================")

    env.close()

if __name__ == '__main__':
    main()