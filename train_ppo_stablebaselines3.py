from env.environment import CarlaEnv
from stable_baselines3 import PPO
import gymnasium as gym

from agent.stablebaselines3_architectures import CustomExtractor_PPO_End2end, CustomExtractor_PPO_Modular

from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList, StopTrainingOnMaxEpisodes
import numpy as np
import wandb

# Set up wandb
LOG_IN_WANDB = True
if LOG_IN_WANDB:
    wandb.init(project='CarlaGym-DQN-v2')
    wandb.define_metric("episode")
    wandb.define_metric("reward_mean", step_metric="episode")

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
            if LOG_IN_WANDB:
                wandb.log({"reward_mean": self.last_mean_reward, "episode": self.n_calls // self.eval_freq})
        return result

    def get_results(self):
        return (self.eval_results, self.episode_numbers)

def main():
    env = gym.make('carla-rl-gym-v0', time_limit=55, initialize_server=True, random_weather=False, synchronous_mode=True, continuous=True, show_sensor_data=False, has_traffic=False, verbose=False)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./checkpoints/ppo/",
        name_prefix="ppo_av_checkpoint",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    
    # Adjust eval_freq to be appropriate for every 100 episodes
    eval_callback = CustomEvalCallback(env, eval_freq=env.spec.max_episode_steps * 100, log_path="./logs/")
    
    callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=5000, verbose=1)
    
    callback = CallbackList([checkpoint_callback, eval_callback, callback_max_episodes])
    
    policy_kwargs = dict(
        features_extractor_class=CustomExtractor_PPO_Modular,
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
    
    # Calculate total_timesteps based on 5000 episodes
    total_timesteps = 5000 * env.spec.max_episode_steps
    model.learn(total_timesteps=total_timesteps, callback=callback)
    
    model.save("checkpoints/ppo/ppo_sb3_modular_5000_final")
    env.close()

    if LOG_IN_WANDB:
        wandb.finish()
    
    eval_list, episodes_list = eval_callback.get_results()

    with open("ppo_modular_5000_last_execution_modular.txt", "w") as f:
            f.write(f"reward_means: {eval_list}\n")
            f.write(f"episodes: {episodes_list}\n")
            f.write(f"n_steps: {model.num_timesteps}\n")
            f.write(f"batch_size: {model.batch_size}\n")
            f.write(f"n_epochs: {model.n_epochs}\n")
            f.write(f"gamma: {model.gamma}\n")
            f.write(f"gae_lambda: {model.gae_lambda}\n")
            f.write(f"ent_coef: {model.ent_coef}\n")

if __name__ == '__main__':
    main()
