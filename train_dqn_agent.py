import gymnasium as gym
from agent.dqn import DQN_Agent
from matplotlib import pyplot as plt
import numpy as np
import wandb
from env.environment import CarlaEnv
# 
def plot_reward(reward_means, episodes):
    # Plot the average performance of the agent over the training episodes
    plt.plot(reward_means, episodes)
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title('Average Reward over Training Episodes')
    plt.savefig(f'agent/plots/dqn_average_reward.png')

# Set up wandb
LOG_IN_WANDB = False
if LOG_IN_WANDB:
    wandb.init(project='CarlaGym-DQN-tests')
    # Define custom x axis metric
    wandb.define_metric("episode")
    # Define which metrics will be plotted against 'episode'
    wandb.define_metric("reward_mean", step_metric="episode")
    wandb.define_metric("reward_std", step_metric="episode")

# Set environment and training parameters
num_episodes_train = 10000
num_episodes_test = 10
learning_rate = 5e-4
evaluate_every = 1000

# Create the environment
env = gym.make('carla-rl-gym-v0', time_limit=55, verbose=False, initialize_server=True, random_weather=True, synchronous_mode=True, continuous=False, show_sensor_data=False, has_traffic=False)
action_space_size = env.action_space.n

# Plot average performance of 5 trials
num_seeds = 5
l = num_episodes_train // 10
res = np.zeros((num_seeds, l))
gamma = 0.99

reward_means = []
episodes = []
# Create an instance of the DQN_Agent class
agent = DQN_Agent(env=env, lr=learning_rate)
# agent.load_model_weights("checkpoints/dqn/dqn_final_agent_10000.pth")

# Training loop
for m in range(num_episodes_train):
    print(f"============================= Train Episode {m+1} =======================================")
    agent.train()

    # Evaluate the agent every 10 episodes during training
    if m % evaluate_every == 0:
        print("=-=-=-=-=-=-=-=-=-=-=")
        print("Evaluation Phase for training episode: {}".format(m+1))

        # Evaluate the agent's performance over 20 test episodes
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

        # Plot graph
        plot_reward(reward_means, episodes)

        # Log metrics to wandb
        if LOG_IN_WANDB:
            wandb.log({"reward_mean": reward_mean, "reward_std": reward_sd, "episode": m+1})

        # Save model
        agent.save_model_weights(f"checkpoints/dqn/dqn_{m+1}_checkpoint.pth")
        print(f"Saved checkpoint dqn_{m+1}_checkpoint.pth!")

# Save final model
agent.save_model_weights(f"checkpoints/dqn/dqn_final_agent.pth")
# Save episode_means and episodes in logs file in the following format:
# "reward_means: [0.0, 0.0, 0.0, 0.0, 0.0]"
# "episodes: [0, 1, 2, 3, 4]"
with open("logs/last_execution.txt", "w") as f:
    f.write(f"reward_means: {reward_means}\n")
    f.write(f"episodes: {episodes}\n")
    f.write(f"gamma: {gamma}\n")
    f.write(f"learning_rate: {learning_rate}\n")
    f.write(f"num_episodes_train: {num_episodes_train}\n")
    f.write(f"num_episodes_test: {num_episodes_test}\n")
    f.write(f"evaluate_every: {evaluate_every}\n")
    f.write(f"action_space_size: {action_space_size}\n")
    f.write(f"num_seeds: {num_seeds}\n")


print(f"Agent finalized training! Saved at checkpoints/dqn/dqn_final_agent.pth")
if LOG_IN_WANDB:
    wandb.finish()
env.close()