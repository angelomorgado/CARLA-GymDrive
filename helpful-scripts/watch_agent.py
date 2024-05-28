import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from env.environment import CarlaEnv
from stable_baselines3 import PPO
import gymnasium as gym

from agent.stablebaselines3_architectures import CustomCombinedExtractor

from stable_baselines3.common.evaluation import evaluate_policy

def main():
    env = gym.make('carla-rl-gym-v0', time_limit=55, initialize_server=True, random_weather=True, synchronous_mode=True, continuous=True, show_sensor_data=True, has_traffic=False)
    
    model = PPO.load("checkpoints/sb3_ad_ppo_final")  # Load the trained model
    
    # Evaluate the trained model
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)
    print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")
    
    
    # Episodes
    for i in range(2):
        obs, info = env.reset()
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, terminated, truncated, info = env.step(action)
            # env.render()  # Render the environment
            if terminated or truncated:  # Break if episode is done
                break
    
    env.close()


if __name__ == '__main__':
    main()