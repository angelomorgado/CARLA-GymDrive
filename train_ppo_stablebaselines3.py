from env.environment import CarlaEnv
from stable_baselines3 import PPO
import gymnasium as gym

from agent.stablebaselines3_architectures import CustomExtractor_PPO

from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList, StopTrainingOnMaxEpisodes
import numpy as np

class CustomEvalCallback(EvalCallback):
    def __init__(self, env, eval_freq, log_path, n_eval_episodes=5, deterministic=True, render=False):
        super().__init__(env, best_model_save_path=log_path, log_path=log_path, eval_freq=eval_freq, 
                         n_eval_episodes=n_eval_episodes, deterministic=deterministic, render=render)
        self.eval_results = []
        self.episode_numbers = []

    def _on_step(self) -> bool:
        result = super()._on_step()
        if self.n_calls % self.eval_freq == 0:
            self.eval_results.append(self.last_mean_reward)
            self.episode_numbers.append(self.n_calls // self.eval_freq)
        return result

    def save_results(self, path):
        with open(path, "w") as f:
            for episode, result in zip(self.episode_numbers, self.eval_results):
                f.write(f"Episode {episode*100}: Mean Reward = {result}\n")

def main():
    env = gym.make('carla-rl-gym-v0', time_limit=55, initialize_server=True, random_weather=True, synchronous_mode=True, continuous=True, show_sensor_data=False, has_traffic=False)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=1000 * env.spec.max_episode_steps,
        save_path="./checkpoints/",
        name_prefix="ppo_av_checkpoint",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    
    eval_callback = CustomEvalCallback(env, eval_freq=100 * env.spec.max_episode_steps, log_path="./logs/")
    
    callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=5000, verbose=1)
    
    callback = CallbackList([checkpoint_callback, eval_callback, callback_max_episodes])
    
    policy_kwargs = dict(
        features_extractor_class=CustomExtractor_PPO,
    )
    
    model = PPO(
        policy="MultiInputPolicy",
        policy_kwargs=policy_kwargs,
        env=env,
        n_steps=1024,
        batch_size=64,
        n_epochs=4,
        gamma=0.999,
        tensorboard_log="./ppo_av_tensorboard/",
        gae_lambda=0.98,
        ent_coef=0.01,
        verbose=1,
    )
    
    model.learn(total_timesteps=5000 * env.spec.max_episode_steps, callback=callback)
    
    model.save("checkpoints/sb3_ad_ppo_final")
    env.close()
    
    eval_callback.save_results("logs/ppo_modular_5000_last_execution.txt")

if __name__ == '__main__':
    main()
