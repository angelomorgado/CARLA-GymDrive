from env.environment import CarlaEnv
from stable_baselines3 import PPO
import gymnasium as gym

from agent.stablebaselines3_architectures import CustomExtractor_PPO_End2end, CustomExtractor_PPO_Modular

from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList, StopTrainingOnMaxEpisodes
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
import numpy as np
import wandb

LOG_IN_WANDB = True
END2END = True
NUM_EPISODES = 15000
EVALUATE_EVERY = 1000

# Set up wandb
if LOG_IN_WANDB:
    wandb.init(project='CarlaGym-phase2')
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

def make_env():
    env = gym.make('carla-rl-gym-v0', time_limit=30, initialize_server=True, random_weather=False, synchronous_mode=True, continuous=True, show_sensor_data=False, has_traffic=False, verbose=False)
    env = DummyVecEnv([lambda: env])
    env = VecTransposeImage(env)
    return env

def main():
    env = make_env()
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./checkpoints/ppo/",
        name_prefix="ppo_av_checkpoint",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    
    # Adjust eval_freq to be appropriate for every 1000 episodes
    eval_callback = CustomEvalCallback(env, eval_freq=env.envs[0].spec.max_episode_steps * EVALUATE_EVERY, log_path="./logs/")
    
    callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=NUM_EPISODES, verbose=1)
    
    callback = CallbackList([checkpoint_callback, eval_callback, callback_max_episodes])
    
    policy_kwargs = None
    
    if not END2END:
        policy_kwargs = dict(
            features_extractor_class=CustomExtractor_PPO_Modular,
        )
    else:
        policy_kwargs = dict(
            features_extractor_class=CustomExtractor_PPO_End2end,
        )
    
    model = PPO(
        policy="MultiInputPolicy",
        policy_kwargs=policy_kwargs,
        env=env,
        n_steps=1024,
        batch_size=64,
        n_epochs=4,
        gamma=0.999,
        tensorboard_log="./tensorboard/",
        gae_lambda=0.98,
        ent_coef=0.01,
        verbose=1,
    )
    
    # Calculate total_timesteps based on NUM_EPISODES episodes
    total_timesteps = NUM_EPISODES * (env.envs[0].spec.max_episode_steps + 100) # The +100 is to ensure that the last episode is completed even if the number of steps is reached
    model.learn(total_timesteps=total_timesteps, callback=callback)
    
    n = "modular" if not END2END else "end2end"
    model.save(f"checkpoints/ppo/ppo_sb3_{n}_{NUM_EPISODES}_final")

    eval_list, episodes_list = eval_callback.get_results()

    with open(f"ppo_{n}_{NUM_EPISODES}_last_execution.txt", "w") as f:
            f.write(f"reward_means: {eval_list}\n")
            f.write(f"episodes: {episodes_list}\n")
            f.write(f"n_steps: {model.num_timesteps}\n")
            f.write(f"batch_size: {model.batch_size}\n")
            f.write(f"n_epochs: {model.n_epochs}\n")
            f.write(f"gamma: {model.gamma}\n")
            f.write(f"gae_lambda: {model.gae_lambda}\n")
            f.write(f"ent_coef: {model.ent_coef}\n")
    
    if LOG_IN_WANDB:
        wandb.finish()
        
    env.close()

if __name__ == '__main__':
    main()
