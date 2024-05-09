import gymnasium as gym
import tqdm
from agent.dqn import DQN_Agent
from matplotlib import pyplot as plt
import numpy as np
from env.environment import CarlaEnv

# Set environment and training parameters
num_episodes_train = 5
num_episodes_test = 2
learning_rate = 5e-4
evaluate_every = 3

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

# Training loop
for m in range(num_episodes_train):
    print(f"============================= Train Episode {m} =======================================")
    agent.train()

    # Evaluate the agent every 10 episodes during training
    if m % evaluate_every == 0:
        print("Evaluation Phase for training episode: {}".format(m))

        # Evaluate the agent's performance over 20 test episodes
        G = np.zeros(num_episodes_test)
        for k in range(num_episodes_test):
            print("Evaluation episode number ", k)
            g = agent.test()
            G[k] = g

        reward_mean = G.mean()
        reward_sd = G.std()
        print(f"The test reward for training episode {m} is {reward_mean} with a standard deviation of {reward_sd}.")
        reward_means.append(reward_mean)


# Plot the average performance of the agent over the training episodes
plt.plot(reward_means)
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.title('Average Reward over Training Episodes')
plt.savefig('agent/plots/dqn_average_reward.png')