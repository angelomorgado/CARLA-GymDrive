import gymnasium as gym
from agent.dqn import DQN_Agent
from matplotlib import pyplot as plt
import numpy as np
import wandb
from env.environment import CarlaEnv
# 
def plot_reward(reward_means):
    # Plot the average performance of the agent over the training episodes
    plt.plot(reward_means)
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title('Average Reward over Training Episodes')
    plt.savefig(f'agent/plots/dqn_average_reward.png')

# Set up wandb
wandb.init(project='CarlaGym-DQN')

# Set environment and training parameters
num_episodes_train = 5000
num_episodes_test = 5
learning_rate = 5e-4
evaluate_every = 100

# Create the environment
env = gym.make('carla-rl-gym-v0', time_limit=55, verbose=False, initialize_server=False, random_weather=True, synchronous_mode=True, continuous=False, show_sensor_data=False, has_traffic=False)
action_space_size = env.action_space.n

# Plot average performance of 5 trials
num_seeds = 5
l = num_episodes_train // 10
res = np.zeros((num_seeds, l))
gamma = 0.99

reward_means = []
# Create an instance of the DQN_Agent class
agent = DQN_Agent(env=env, lr=learning_rate)
# Optional: Load the model weights from a previous checkpoint (there is no checkpoint at the moment)
# print("Loading model weights from checkpoint...")
# agent.load_model_weights("checkpoints/dqn/dqn_last_checkpoint.pth")

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

        # Plot graph
        plot_reward(reward_means)

        # Log metrics to wandb
        wandb.log({"reward_mean": reward_mean, "reward_std": reward_sd, "episode": m+1})

        # Save model
        agent.save_model_weights(f"checkpoints/dqn/dqn_{m+1}_checkpoint.pth")
        print(f"Saved checkpoint dqn_{m+1}_checkpoint.pth!")

# Save final model
agent.save_model_weights(f"checkpoints/dqn/dqn_final_agent.pth")
print(f"Agent finalized training! Saved at checkpoints/dqn/dqn_final_agent.pth")
wandb.finish()