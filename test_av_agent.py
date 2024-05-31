import gymnasium as gym
import numpy as np
from env.environment import CarlaEnv

from agent.stablebaselines3_architectures import CustomExtractor_PPO, CustomExtractor_DQN
from agent.dqn import DQN_Agent
from agent.ppo import PPO_Agent

def test_DQN_agent(num_episodes= 10, end_to_end=True, filename=None):    
    env = gym.make('carla-rl-gym-v0', time_limit=55, verbose=False, initialize_server=True, random_weather=False, synchronous_mode=True, continuous=False, show_sensor_data=True, has_traffic=False)
    
    agent = DQN_Agent(env=env, end_to_end=end_to_end)
    agent.load_model_weights(filename)
        
    rewards_list = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        r = []
        while True:
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            r.append(reward)
            state = next_state
            
            if terminated or truncated:
                rewards_list.append(np.sum(r))
                break
    
    return rewards_list

def test_PPO_agent(num_episodes= 10, end_to_end=True, filename=None):    
    env = gym.make('carla-rl-gym-v0', time_limit=55, verbose=False, initialize_server=True, random_weather=False, synchronous_mode=True, continuous=False, show_sensor_data=True, has_traffic=False)
    
    agent = PPO_Agent(env=env, end_to_end=end_to_end)
    agent.load_model(filename)
    
    rewards_list = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        r = []
        while True:
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            r.append(reward)
            state = next_state
            
            if terminated or truncated:
                rewards_list.append(np.sum(r))
                break
    
    return rewards_list
            
def main():
    dqn_rewards = test_DQN_agent(num_episodes=10, end_to_end=True, filename="checkpoints/dqn/dqn_final_agent.pth")
    # ppo_rewards = test_PPO_agent(num_episodes=10, end_to_end=True, filename="checkpoints/ppo/ppo_final_agent.pth")
    
    print("DQN Rewards: ", dqn_rewards)
    # print("PPO Rewards: ", ppo_rewards)

if __name__ == "__main__":
    main()
    