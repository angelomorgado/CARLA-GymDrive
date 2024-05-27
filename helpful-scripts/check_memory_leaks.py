from memory_profiler import profile
import gymnasium as gym
from env.environment import CarlaEnv

@profile
def reset_and_profile(env):
    state, _ = env.reset()
    return state

# Example usage
if __name__ == "__main__":
    env = gym.make('carla-rl-gym-v0', time_limit=55, verbose=False, initialize_server=True, random_weather=True, synchronous_mode=True, continuous=False, show_sensor_data=False, has_traffic=False)
    for _ in range(5000):  # Run multiple times to observe memory growth
        state = reset_and_profile(env)