from env.environment import CarlaEnv
from stable_baselines3 import DQN
import gymnasium as gym

from agent.stablebaselines3_architectures import CustomExtractor_DQN

from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList, StopTrainingOnMaxEpisodes

def main():
    env = gym.make('carla-rl-gym-v0', time_limit=55, initialize_server=True, random_weather=True, synchronous_mode=True, continuous=False, show_sensor_data=False, has_traffic=False)
    
    # Stable baselines callback
    # Save a checkpoint every 100000 steps
    checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path="./checkpoints/",
        name_prefix="dqn_av_checkpoint",
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
        features_extractor_class=CustomExtractor_DQN,
    )
    
    model = DQN(
        policy="MultiInputPolicy",
        policy_kwargs=policy_kwargs,
        env=env,
        buffer_size=5000,
        learning_starts=1000,
        batch_size=32,
        gamma=0.99,
        train_freq=(4, 'step'),
        gradient_steps=1,
        target_update_interval=1000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        tensorboard_log="./dqn_av_tensorboard/",
        verbose=1,
    )
    
    model.learn(total_timesteps=int(1000000), callback=callback)
    
    model.save("checkpoints/sb3_ad_dqn_final")
    env.close()

if __name__ == '__main__':
    main()
