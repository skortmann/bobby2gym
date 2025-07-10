# battery storage optimisation with Reinforcement Learning
import sys
from tqdm.auto import tqdm
import argparse
import stable_baselines3 as sb3
from stable_baselines3.common.monitor import Monitor
import yaml

from wandb.integration.sb3 import WandbCallback
import os
import wandb

try:
    from battery_efficiency import BatteryEfficiency
    from battery_environment import Battery
    from common import load_training_settings_from_yaml as load_settings, update_settings
    print("Imports were successful.")
except ImportError:
    print("Imports were not successful.")
    sys.exit(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and test DQN with DA ID')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file')
    args = parser.parse_args()

    settings = load_settings(args.config, da_or_da_id='da')
    if settings['use_wandb']:
        wandb.login(key="95be751a02f3d42e6fdd0b54e8fbf97f0aa2bb93")

    for i in range(settings['number_runs']):

        # update settings
        settings = update_settings(settings, da_or_da_id='da')
        modulus = settings['modulus']
        time = settings['time']
        if settings['use_wandb']:
            run = wandb.init(
                project=f"Battery_optimisation_single_{modulus}",
                config=settings,
                sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
                monitor_gym=False,  # auto-upload the videos of agents playing the game
                save_code=True,  # optional
            )
        print("wandb loaded or skipped")
        # making the directories
        os.makedirs(settings['models_dir'], exist_ok=True)
        os.makedirs(settings['logdir'], exist_ok=True)
        os.makedirs(settings['eval_logdir'], exist_ok=True)
        os.makedirs(settings['config_dir'], exist_ok=True)
        print('config_dir: ', settings['config_dir'])
        with open(os.path.join(settings['config_dir'], 'config.yaml'), 'w') as file:
            yaml.dump(load_settings(args.config), file)
        # store args.config into folder
        print("directories made")
        # load training and evaluation callback environment
        env = Battery(settings["environment"], train=True)
        print("Environment loaded")
        eval_env = Battery(settings["environment"], train=False)
        print("Evaluation environment loaded")
        eval_env = sb3.common.monitor.Monitor(eval_env, settings['eval_logdir'])
        print("environments loaded")
        env.reset()
        eval_env.reset()
        print("Both environments reseted")
        model = sb3.DQN("MlpPolicy",
                        env=env,
                        verbose=settings['verbose'],
                        tensorboard_log=settings['logdir'],
                        **settings['dqn_kwargs'])
        print("Agent loaded")
        ###
        eval_callback = sb3.common.callbacks.EvalCallback(eval_env,
                                                          n_eval_episodes=settings['ep_per_eval_call'],
                                                          best_model_save_path=settings['models_dir'],
                                                          log_path=settings['logdir'],
                                                          eval_freq=(settings['delta_ep_eval_call'] * settings[
                                                              "episode_length"]),
                                                          deterministic=True,
                                                          render=False)
        callbacks = [eval_callback]
        if settings['use_wandb']:
            callbacks.append(WandbCallback(verbose=settings['verbose'],
                                           model_save_path=settings['models_dir'],
                                           model_save_freq=1000,))
        print("callback loaded")
        model.learn(total_timesteps=(settings['total_timesteps']),
                    reset_num_timesteps=False,
                    tb_log_name='DQN',
                    callback=callbacks,
                    progress_bar=True)
        print("Model have learned")
        print("Evaluating the policy...")
        mean_reward, std_reward = sb3.common.evaluation.evaluate_policy(model, eval_env, n_eval_episodes=settings['ep_eval_policy'],
                                                                        deterministic=True)
        print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
        if settings['use_wandb']:
            run.log({'eval_mean_reward': mean_reward, 'eval_std_reward': std_reward})
            run.finish()
