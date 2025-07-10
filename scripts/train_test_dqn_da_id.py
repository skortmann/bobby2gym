#!/usr/bin/env python3
"""
Battery Storage Optimization with Reinforcement Learning
"""

import sys
import os
import yaml
import argparse
import logging
from typing import Any, Dict, Tuple, Optional

import wandb
import stable_baselines3 as sb3
from stable_baselines3.common.monitor import Monitor
from wandb.integration.sb3 import WandbCallback

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# Import custom classes
try:
    from battery_efficiency import BatteryEfficiency
    from battery_environment import Battery
    from common import load_training_settings_from_yaml as load_settings, update_settings

    logging.info("Imports were successful.")
except ImportError as e:
    logging.error("Imports were not successful: %s", e)
    sys.exit(1)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Train and test DQN with DA ID')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file')
    return parser.parse_args()


def setup_wandb(settings: Dict[str, Any]) -> Optional[wandb.sdk.wandb_run.Run]:
    """Initialize wandb if enabled in the settings.

    Returns:
        The wandb run object if wandb is used, else None.
    """
    if settings.get('use_wandb'):
        api_key = os.environ.get("WANDB_API_KEY")
        if not api_key:
            logging.error("WANDB_API_KEY not found in environment variables.")
            sys.exit(1)
        wandb.login(key=api_key)
        modulus = settings.get('modulus', 'default')
        run = wandb.init(
            project=f"Battery_optimisation_multi_{modulus}",
            config=settings,
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            monitor_gym=False,  # auto-upload videos of agent gameplay
            save_code=True,  # optional
        )
        logging.info("wandb initialized.")
        return run
    else:
        logging.info("wandb skipped.")
        return None


def create_directories(settings: Dict[str, Any]) -> None:
    """Create necessary directories based on the settings."""
    os.makedirs(settings['models_dir'], exist_ok=True)
    os.makedirs(settings['logdir'], exist_ok=True)
    os.makedirs(settings['eval_logdir'], exist_ok=True)
    os.makedirs(settings['config_dir'], exist_ok=True)
    logging.info("Directories created. Config directory: %s", settings['config_dir'])


def save_config(settings: Dict[str, Any], config_path: str) -> None:
    """Save the current configuration into the config directory."""
    config_save_path = os.path.join(settings['config_dir'], 'config.yaml')
    try:
        with open(config_save_path, 'w') as file:
            # Reload the settings from the original config file for saving
            yaml.dump(load_settings(config_path, da_or_da_id='da_id'), file)
        logging.info("Configuration saved to %s", config_save_path)
    except Exception as e:
        logging.error("Failed to save configuration: %s", e)


def create_environments(settings: Dict[str, Any]) -> Tuple[Any, Any]:
    """Create and reset both the training and evaluation environments.

    Returns:
        A tuple containing the training environment and the evaluation environment.
    """
    train_env = Battery(settings["environment"], train=True)
    eval_env = Battery(settings["environment"], train=False)
    eval_env = Monitor(eval_env, settings['eval_logdir'])
    train_env.reset()
    eval_env.reset()
    sb3.common.env_checker.check_env(train_env)
    logging.info("Training and evaluation environments initialized and reset.")
    return train_env, eval_env


def create_agent(settings: Dict[str, Any], env: Any) -> sb3.DQN:
    """Instantiate the DQN agent with the given environment and settings.

    Returns:
        The instantiated DQN agent.
    """
    model = sb3.DQN(
        "MultiInputPolicy",
        env=env,
        verbose=settings['verbose'],
        tensorboard_log=settings['logdir'],
        device="auto",
        **settings['dqn_kwargs']
    )
    logging.info("Agent created.")
    return model


def setup_callbacks(settings: Dict[str, Any], eval_env: Any) -> list:
    """Set up evaluation and wandb callbacks.

    Returns:
        A list of callbacks.
    """
    eval_freq = settings['delta_ep_eval_call'] * settings["episode_length"]
    eval_callback = sb3.common.callbacks.EvalCallback(
        eval_env,
        n_eval_episodes=settings['ep_per_eval_call'],
        best_model_save_path=settings['models_dir'],
        log_path=settings['logdir'],
        eval_freq=eval_freq,
        deterministic=True,
        render=False
    )
    callbacks = [eval_callback]
    if settings.get('use_wandb'):
        callbacks.append(WandbCallback(
            verbose=settings['verbose'],
            model_save_path=settings['models_dir'],
            model_save_freq=1000,
        ))
    logging.info("Callbacks set up.")
    return callbacks


def train_model(model: sb3.DQN, settings: Dict[str, Any], callbacks: list) -> None:
    """Train the model using the provided settings and callbacks."""
    try:
        model.learn(
            total_timesteps=settings['total_timesteps'],
            reset_num_timesteps=False,
            tb_log_name='DQN',
            callback=callbacks,
            progress_bar=True
        )
        logging.info("Model training completed.")
    except Exception as e:
        logging.error("Error during model training: %s", e)
        sys.exit(1)


def evaluate_models(model: sb3.DQN, settings: Dict[str, Any], eval_env: Any) -> Dict[str, float]:
    """Evaluate both the current and best saved models.

    Returns:
        A dictionary containing evaluation results.
    """
    eval_mean_reward_model, eval_std_reward_model = sb3.common.evaluation.evaluate_policy(
        model, eval_env, n_eval_episodes=settings['ep_eval_policy'], deterministic=True
    )
    best_model_path = os.path.join(settings['models_dir'], 'best_model')
    best_model = sb3.DQN.load(best_model_path)
    eval_mean_reward_bestmodel, eval_std_reward_bestmodel = sb3.common.evaluation.evaluate_policy(
        best_model, eval_env, n_eval_episodes=settings['ep_eval_policy'], deterministic=True
    )
    logging.info("Evaluation - Current Model: Mean Reward = %.2f +/- %.2f",
                 eval_mean_reward_model, eval_std_reward_model)
    logging.info("Evaluation - Best Model: Mean Reward = %.2f +/- %.2f",
                 eval_mean_reward_bestmodel, eval_std_reward_bestmodel)

    return {
        'eval_mean_reward_model': eval_mean_reward_model,
        'eval_std_reward_model': eval_std_reward_model,
        'eval_mean_reward_bestmodel': eval_mean_reward_bestmodel,
        'eval_std_reward_bestmodel': eval_std_reward_bestmodel
    }


def main() -> None:
    args = parse_arguments()

    # Load and update settings from the YAML config file
    settings = load_settings(args.config, da_or_da_id='da_id')
    settings = update_settings(settings, da_or_da_id='da_id')

    # Setup wandb if enabled
    run = setup_wandb(settings)

    # Create required directories and save the configuration
    create_directories(settings)
    save_config(settings, args.config)

    # Initialize training and evaluation environments
    train_env, eval_env = create_environments(settings)

    # Create the agent and set up callbacks
    model = create_agent(settings, train_env)
    callbacks = setup_callbacks(settings, eval_env)

    # Train the agent
    train_model(model, settings, callbacks)

    # Evaluate the policy from both the current and best saved models
    eval_results = evaluate_models(model, settings, eval_env)

    # Log evaluation results to wandb if enabled and finish the run
    if run:
        try:
            run.log(eval_results)
        except Exception as e:
            logging.error("Failed to log results to wandb: %s", e)
        finally:
            run.finish()

    logging.info("Training and evaluation complete.")


if __name__ == '__main__':
    main()
