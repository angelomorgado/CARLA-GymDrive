import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from env.environment import CarlaEnv
from stable_baselines3 import PPO
import gymnasium as gym

from agent.custom_feature_extractor import CustomCombinedExtractor

def main():
    env = gym.make('carla-rl-gym-v0', time_limit=50, initialize_server=True, random_weather=False, synchronous_mode=True, continuous=True, show_sensor_data=True)
    
    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
    )
    
    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        n_steps=1024,
        batch_size=64,
        n_epochs=4,
        gamma=0.999,
        gae_lambda=0.98,
        ent_coef=0.01,
        verbose=1,
    )

    model.learn(total_timesteps=int(1000))

    model.save("ppo_test-agent")
    
    env.close()


if __name__ == '__main__':
    main()