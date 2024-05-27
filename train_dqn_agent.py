import gymnasium as gym
from agent.dqn import DQN_Agent
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
    plt.savefig(f'agent/plots/dqn_average_reward.png')
    plt.close()  # Close the figure to free up memory

# Set up wandb
LOG_IN_WANDB = False
if LOG_IN_WANDB:
    wandb.init(project='CarlaGym-DQN-tests')
    wandb.define_metric("episode")
    wandb.define_metric("reward_mean", step_metric="episode")
    wandb.define_metric("reward_std", step_metric="episode")

# Set environment and training parameters
num_episodes_train = 10000
num_episodes_test = 10
learning_rate = 5e-4
evaluate_every = 1000

# Set Agent type
end_to_end_agent = False # If false the agent will be modular (i.e., the perception module will be separated from the decision-making module)

env = gym.make('carla-rl-gym-v0', time_limit=55, verbose=False, initialize_server=True, random_weather=True, synchronous_mode=True, continuous=False, show_sensor_data=False, has_traffic=False)
action_space_size = env.action_space.n

num_seeds = 5
l = num_episodes_train // 10
res = np.zeros((num_seeds, l))
gamma = 0.99

reward_means = []
episodes = []

agent = DQN_Agent(env=env, lr=learning_rate, end_to_end=end_to_end_agent)

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

            agent.save_model_weights(f"checkpoints/dqn/dqn_{m+1}_checkpoint.pth")
            print(f"Saved checkpoint dqn_{m+1}_checkpoint.pth!")

    agent.save_model_weights(f"checkpoints/dqn/dqn_final_agent.pth")

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
except Exception as e:
    print(f"Error!!!: {e}")
    traceback.print_exc()
finally:
    if LOG_IN_WANDB:
        wandb.finish()
    env.close()