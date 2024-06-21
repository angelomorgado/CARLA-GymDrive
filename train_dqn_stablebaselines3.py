from env.environment import CarlaEnv
from stable_baselines3 import DQN
import gymnasium as gym

from agent.stablebaselines3_architectures import CustomExtractor_DQN_End2end, CustomExtractor_DQN_Modular

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
    env = gym.make('carla-rl-gym-v0', time_limit=30, initialize_server=True, random_weather=False, synchronous_mode=True, continuous=False, show_sensor_data=False, has_traffic=False, verbose=False)
    env = DummyVecEnv([lambda: env])
    env = VecTransposeImage(env)
    return env

def main():
    env = make_env()
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./checkpoints/dqn/",
        name_prefix="dqn_av_checkpoint",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    
    eval_callback = CustomEvalCallback(env, eval_freq=env.envs[0].spec.max_episode_steps * EVALUATE_EVERY, log_path="./logs/")
    
    callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=NUM_EPISODES, verbose=1)
    
    callback = CallbackList([checkpoint_callback, eval_callback, callback_max_episodes])
    
    policy_kwargs = None
    
    if not END2END:
        policy_kwargs = dict(
            features_extractor_class=CustomExtractor_DQN_Modular,
        )
    else:
        policy_kwargs = dict(
            features_extractor_class=CustomExtractor_DQN_End2end,
        )
    
    model = DQN(
        policy="MultiInputPolicy",
        policy_kwargs=policy_kwargs,
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
        tensorboard_log="./tensorboard/",
        verbose=1,
    )
    
    # Calculate total_timesteps based on NUM_EPISODES episodes
    total_timesteps = NUM_EPISODES * (env.envs[0].spec.max_episode_steps + 100) # The +100 is to ensure that the last episode is completed even if the number of steps is reached
    model.learn(total_timesteps=total_timesteps, callback=callback)
    
    n = "modular" if not END2END else "end2end"
    model.save(f"checkpoints/dqn/dqn_sb3_{n}_{NUM_EPISODES}_final")

    eval_list, episodes_list = eval_callback.get_results()

    with open(f"dqn_{n}_{NUM_EPISODES}_last_execution.txt", "w") as f:
            f.write(f"reward_means: {eval_list}\n")
            f.write(f"episodes: {episodes_list}\n")
            f.write(f"learning_rate: {model.learning_rate}\n")
            f.write(f"buffer_size: {model.buffer_size}\n")
            f.write(f"learning_starts: {model.learning_starts}\n")
            f.write(f"batch_size: {model.batch_size}\n")
            f.write(f"tau: {model.tau}\n")
            f.write(f"gamma: {model.gamma}\n")
            f.write(f"train_freq: {model.train_freq}\n")
            f.write(f"gradient_steps: {model.gradient_steps}\n")
            f.write(f"target_update_interval: {model.target_update_interval}\n")
            f.write(f"exploration_fraction: {model.exploration_fraction}\n")
            f.write(f"exploration_final_eps: {model.exploration_final_eps}\n")
    
    if LOG_IN_WANDB:
        wandb.finish()
        
    env.close()

if __name__ == '__main__':
    main()
