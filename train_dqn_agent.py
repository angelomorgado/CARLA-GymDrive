import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from agent.dqn import DQNAgent
from env.environment import CarlaEnv

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

# Environment
env = gym.make('carla-rl-gym-v0', time_limit=55, initialize_server=False, random_weather=True, synchronous_mode=True, continuous=False, show_sensor_data=False, has_traffic=False)
state_space = env.observation_space
action_space = env.action_space

# Agent
agent = DQNAgent(state_space, action_space)

rewards = np.array([])
losses = np.array([])

# Training
num_episodes = 10
for episode in range(num_episodes):
    state = env.reset()
    truncated = False
    terminated = False
    total_reward = 0
    
    episode_losses = []
    
    while True:
        # Choose action
        action = agent.act(state)
        
        # Take action
        next_state, reward, terminated, truncated, info = env.step(action)
        
        # Train the agent
        agent.train(state, action, reward, next_state, terminated, truncated)
        
        # Update state
        state = next_state
    
        episode_losses.append(agent.get_loss())
        
        total_reward += reward
        
        if terminated or truncated:
            # Store reward
            rewards = np.append(rewards, total_reward)
            break
    
    try:
        losses = np.append(losses, np.mean(episode_losses))
    except TypeError:
        losses = np.append(losses, 0)
    
    # Print episode info
    print(f"Episode: {episode + 1}, Total Reward: {total_reward}, Loss: {losses[-1]}")
    
    # Save model every 100 episodes
    if (episode + 1) % 5 == 0:
        agent.save(f"checkpoints/dqn/dqn_model_{episode + 1}.pt")
        
        # print(f"rewards shape: {rewards.shape}, losses shape: {losses.shape}, episodes: {episode + 1}")
        
        episodes = np.arange(episode + 1)
        plot_graph(episodes, rewards, losses)

# Close environment
env.close()
