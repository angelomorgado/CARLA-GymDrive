import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from env.environment import CarlaEnv
from stable_baselines3 import PPO
import gymnasium as gym
from matplotlib import pyplot as plt

from agent.custom_feature_extractor import CustomCombinedExtractor

def plot_graph(episodes, rewards, losses):
    # Make a plot with two lines on the same figure and the y axis has two sets of values
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Rewards', color='tab:blue')
    ax1.plot(episodes, rewards, color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Loss', color='tab:red')
    ax2.plot(episodes, losses, color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    
    fig.tight_layout()
    plt.savefig(f'agent/plots/dqn_{len(episodes)}_loss_reward.png')

def main():
    env = gym.make('carla-rl-gym-v0', time_limit=50, initialize_server=False, random_weather=False, synchronous_mode=True, continuous=True, show_sensor_data=True)
    
    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
    )
    
    model = PPO(
        policy="MultiInputPolicy",
        policy_kwargs=policy_kwargs,
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