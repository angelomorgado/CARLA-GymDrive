'''
This script checks if the environment is compatible with the OpenAI Gym API.
'''

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from env.environment import CarlaEnv
from stable_baselines3.common.env_checker import check_env

def main():
    env = CarlaEnv(initialize_server=True, has_traffic=False, verbose=False)
    check_env(env)
    env.close()

if __name__ == '__main__':
    main()