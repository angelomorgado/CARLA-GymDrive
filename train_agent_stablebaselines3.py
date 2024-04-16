from env.environment import CarlaEnv
from stable_baselines3 import PPO
import gymnasium as gym

from agent.custom_feature_extractor import CustomCombinedExtractor

from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList, StopTrainingOnMaxEpisodes

def main():
    env = gym.make('carla-rl-gym-v0', time_limit=55, initialize_server=True, random_weather=True, synchronous_mode=True, continuous=True, show_sensor_data=True, has_traffic=False)
    
    # Stable baselines callback
    # Save a checkpoint every 1000 steps
    checkpoint_callback = CheckpointCallback(
        save_freq=1000,
        save_path="./checkpoints/",
        name_prefix="ppo_av_checkpoint",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    
    # Use deterministic actions for evaluation
    eval_callback = EvalCallback(env, best_model_save_path="./logs/",
                                log_path="./logs/", eval_freq=500,
                                deterministic=True, render=False)
    
    callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=1000, verbose=1)
    
    callback = CallbackList([checkpoint_callback, eval_callback, callback_max_episodes])
    
    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
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
    
    for i in range(10):
        model.learn(total_timesteps=int(100), callback=callback)
    
    model.save("checkpoints/sb3_ad_ppo_final")
    env.close()


if __name__ == '__main__':
    main()