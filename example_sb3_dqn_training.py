'''
This is an example script to train a DQN agent in the Carla environment using the stable-baselines3 library.

This is just a simple example to show that it is possible to train an agent using the stable-baselines3 library.

If you want to train an agent for a more real-life problem, you should consider using more complex models and hyperparameters; or even using other RL libraries compatible with the gym interface.
'''
from src.env.environment import CarlaEnv # It is mandatory to import the environment even if it is not used in this script
from stable_baselines3 import DQN
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

def make_env():
    env = gym.make('carla-rl-gym-v0', time_limit=30, initialize_server=True, random_weather=False, synchronous_mode=True, continuous=False, show_sensor_data=True, has_traffic=True, verbose=False)
    env = DummyVecEnv([lambda: env])
    env = VecTransposeImage(env)
    return env

def main():
    # Create the environment
    env = make_env()
    
    # Create the agent
    model = DQN(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=1e-4,
        buffer_size=10000,
        learning_starts=1000,
        batch_size=32,
        tau=0.005,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=2000,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        verbose=1,
    )
    
    # Learn 10000 steps
    model.learn(total_timesteps=10000)
    
    # Save the agent
    model.save(f"dqn_example_agent")
    
    # Close the environment
    env.close()

if __name__ == '__main__':
    main()