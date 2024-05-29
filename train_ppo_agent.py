import gymnasium as gym
from agent.ppo import PPOAgent
from matplotlib import pyplot as plt
import numpy as np
import wandb
from env.environment import CarlaEnv
import traceback

def plot_reward(reward_means, episodes):
    plt.plot(episodes, reward_means)
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title('Average Reward over Training Episodes')
    plt.savefig(f'agent/plots/ppo_average_reward.png')
    plt.close()  # Close the figure to free up memory

# Set up wandb
LOG_IN_WANDB = False
if LOG_IN_WANDB:
    wandb.init(project='CarlaGym-PPO')
    wandb.define_metric("episode")
    wandb.define_metric("reward_mean", step_metric="episode")
    wandb.define_metric("reward_std", step_metric="episode")

# Set environment and training parameters
num_episodes_train = 5000
num_episodes_test = 10
lr_actor = 3e-4
lr_critic = 1e-3
evaluate_every = 100
action_std_init = 0.6
gamma = 0.99
K_epochs = 80
eps_clip = 0.2

env = gym.make('carla-rl-gym-v0', time_limit=55, verbose=False, initialize_server=True, random_weather=True, synchronous_mode=True, continuous=True, show_sensor_data=False, has_traffic=False)

reward_means = []
episodes = []

agent = PPOAgent(env=env, lr_actor=lr_actor, lr_critic=lr_critic, gamma=gamma, K_epochs=K_epochs, eps_clip=eps_clip, action_std_init=action_std_init)

try:
    for m in range(num_episodes_train):
        print(f"============================= Train Episode {m+1} =======================================")
        agent.train()

        if m % evaluate_every == 0:
            print("=-=-=-=-=-=-=-=-=-=-=")
            print("Evaluation Phase for training episode: {}".format(m+1))

            G = np.zeros(num_episodes_test)
            for k in range(num_episodes_test):
                print("             - Evaluation episode number ", k+1)
                g = agent.test()
                G[k] = g

            reward_mean = G.mean()
            reward_sd = G.std()
            print(f"The test reward for training episode {m+1} is {reward_mean} with a standard deviation of {reward_sd}.")
            reward_means.append(reward_mean)
            episodes.append(m+1)

            plot_reward(reward_means, episodes)

            if LOG_IN_WANDB:
                wandb.log({"reward_mean": reward_mean, "reward_std": reward_sd, "episode": m+1})

            agent.save_model(f"checkpoints/ppo/ppo_{m+1}_checkpoint.pth")
            print(f"Saved checkpoint ppo_{m+1}_checkpoint.pth!")

    agent.save_model(f"checkpoints/ppo/ppo_final_agent.pth")

    with open("logs/last_execution.txt", "w") as f:
        f.write(f"reward_means: {reward_means}\n")
        f.write(f"episodes: {episodes}\n")
        f.write(f"gamma: {gamma}\n")
        f.write(f"learning_rate_actor: {lr_actor}\n")
        f.write(f"learning_rate_critic: {lr_critic}\n")
        f.write(f"num_episodes_train: {num_episodes_train}\n")
        f.write(f"num_episodes_test: {num_episodes_test}\n")
        f.write(f"evaluate_every: {evaluate_every}\n")

    print(f"Agent finalized training! Saved at checkpoints/ppo/ppo_final_agent.pth")
except KeyboardInterrupt:
    print("Training interrupted by user. Saving model weights...")
    agent.save_model(f"checkpoints/ppo/ppo_interrupted_agent.pth")
    with open("logs/last_execution.txt", "w") as f:
        f.write(f"reward_means: {reward_means}\n")
        f.write(f"episodes: {episodes}\n")
        f.write(f"gamma: {gamma}\n")
        f.write(f"learning_rate_actor: {lr_actor}\n")
        f.write(f"learning_rate_critic: {lr_critic}\n")
        f.write(f"num_episodes_train: {num_episodes_train}\n")
        f.write(f"num_episodes_test: {num_episodes_test}\n")
        f.write(f"evaluate_every: {evaluate_every}\n")
except Exception as e:
    print(f"Error!!!: {e}")
    traceback.print_exc()
finally:
    if LOG_IN_WANDB:
        wandb.finish()
    env.close()
