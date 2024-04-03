import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from agent.dqn import DQNAgent

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
env_name = 'CartPole-v1'
env = gym.make(env_name)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Agent
agent = DQNAgent(state_size, action_size)

rewards = np.array([])
losses = np.array([])

# Training
num_episodes = 1000
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
        
        # Store experience
        agent.remember(state, action, reward, next_state, terminated, truncated)
        
        # Update state
        state = next_state
        
        # Replay
        agent.replay()
        
        episode_losses.append(agent.get_loss())
        
        # Update target network
        agent.update_target_model()
        
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
    if (episode + 1) % 30 == 0:
        agent.save(f"agent/checkpoints/dqn_model_{episode + 1}.pt")
        
        # print(f"rewards shape: {rewards.shape}, losses shape: {losses.shape}, episodes: {episode + 1}")
        
        episodes = np.arange(episode + 1)
        plot_graph(episodes, rewards, losses)

# Close environment
env.close()