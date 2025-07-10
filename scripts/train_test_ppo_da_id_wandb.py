# battery storage optimisation with Reinforcement Learning
import sys
import numpy as np
import argparse
import stable_baselines3 as sb3
import yaml
from pickle import dump
from wandb.integration.sb3 import WandbCallback
import os
import wandb

# import custom classes:
try:
    from battery_environment import Battery
    from common import (
        update_settings,
        get_settings_from_wandb_config,
    )

    print("Imports were successful.")
except ImportError:
    print("Imports were not successful.")
    sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and test PPO with DA ID")
    run = wandb.init(
        project="Battery_optimisation_multi_best_milp",
        config=wandb.config,
        sync_tensorboard=True,
        monitor_gym=False,
        save_code=True,
    )

    # update wandb.config
    wandb.config.update(
        update_settings(
            wandb.config, da_or_da_id="da_id", set_time=True, run_id=run.name
        )
    )
    settings = get_settings_from_wandb_config()
    print("wandb loaded or skipped")

    # making the directories
    os.makedirs(wandb.config["models_dir"], exist_ok=True)
    os.makedirs(wandb.config["config_dir"], exist_ok=True)

    # writing settings as yaml file to folder
    with open(os.path.join(wandb.config["config_dir"], "config.yaml"), "w") as file:
        yaml.dump(settings, file)
        print("config file stored")

    # load training and evaluation callback environment
    env = Battery(env_settings=wandb.config["environment"], train=True)
    eval_env = Battery(wandb.config["environment"], train=False)
    eval_env = sb3.common.monitor.Monitor(eval_env)
    env.reset()
    eval_env.reset()

    print("Both environments reset")

    # Initialize PPO model
    model = sb3.PPO(
        "MultiInputPolicy",
        env=env,
        verbose=wandb.config["verbose"],
        tensorboard_log="tensorboard_logs",
        **wandb.config["ppo_kwargs"],
    )
    print("Agent loaded")

    # Evaluation callback
    eval_callback = sb3.common.callbacks.EvalCallback(
        eval_env,
        n_eval_episodes=wandb.config["ep_per_eval_call"],
        best_model_save_path=wandb.config["models_dir"],
        eval_freq=(wandb.config["delta_ep_eval_call"] * wandb.config["episode_length"]),
        deterministic=True,
        render=False,
    )
    callbacks = [eval_callback]
    callbacks.append(
        WandbCallback(
            verbose=wandb.config["verbose"],
            model_save_path=wandb.config["models_dir"],
            model_save_freq=1000,
        )
    )
    print("Callbacks loaded")

    # Train PPO model
    model.learn(
        total_timesteps=wandb.config["total_timesteps"],
        reset_num_timesteps=False,
        tb_log_name="PPO",
        callback=callbacks,
        progress_bar=True,
    )
    print("Model training completed")

    # Evaluate the policy
    eval_mean_reward_model, eval_std_reward_model = (
        sb3.common.evaluation.evaluate_policy(
            model,
            eval_env,
            n_eval_episodes=wandb.config["ep_eval_policy"],
            deterministic=True,
        )
    )

    # Load best model for evaluation
    best_model = sb3.PPO.load(os.path.join(wandb.config["models_dir"], "best_model"))
    eval_mean_reward_best_model, eval_std_reward_best_model = (
        sb3.common.evaluation.evaluate_policy(
            best_model,
            eval_env,
            n_eval_episodes=wandb.config["ep_eval_policy"],
            deterministic=True,
        )
    )

    # MILP evaluation logic
    alpha_d_average = eval_env.get_average_degradation_coefficients()
    run.log(
        {
            "eval_mean_reward_best_model": eval_mean_reward_best_model,
            "eval_std_reward_best_model": eval_std_reward_best_model,
            "eval_mean_reward_model": eval_mean_reward_model,
            "eval_std_reward_model": eval_std_reward_model,
            "alpha_d_average": alpha_d_average,
        }
    )

    settings_milp_eval = settings.copy()
    os.makedirs(wandb.config["milp_eval"], exist_ok=True)
    settings_milp_eval["environment"].update({"episode_length": (292 * 24 * 4)})
    milp_results = {}

    model_types = ["model", "best_model"]
    for model_type in model_types:
        milp_eval_env = Battery(
            settings_milp_eval["environment"],
            train=False,
            eval_const_deg=settings_milp_eval["environment"]["degradation"],
            alpha_d_constant=alpha_d_average,
        )
        milp_eval_env = sb3.common.monitor.Monitor(milp_eval_env)
        obs, info_ = milp_eval_env.reset()
        milp_model = sb3.PPO.load(os.path.join(wandb.config["models_dir"], model_type))

        info = {
            "ts": np.array([]),
            "action_sum": np.array([]),
            "action_sum_kwh": np.array([]),
            "reward": np.array([]),
            "action_id": np.array([]),
            "action_da": np.array([]),
            "price_id": np.array([]),
            "price_da": np.array([]),
            "soc": np.array([]),
            "efficiency": np.array([]),
            "cycle_num": np.array([]),
            "alpha_d": np.array([]),
        }

        while True:
            action, _states = milp_model.predict(obs, deterministic=True)
            obs, rewards, terminated, truncated, info_ = milp_eval_env.step(action)
            if info_["label"] == 1:
                for key in info.keys():
                    info[key] = np.append(info[key], info_[key])
            if terminated or truncated:
                info_path = os.path.join(
                    wandb.config["milp_eval"], f"info_{model_type}.pkl"
                )
                info["profit"] = (
                    info["action_da"] * info["price_da"]
                    + info["action_id"] * info["price_id"]
                )
                info["profit_cumsum"] = np.cumsum(info["profit"])
                info["reward_cumsum"] = np.cumsum(info["reward"])
                with open(info_path, "wb") as f:
                    dump(info, f)
                milp_results[model_type] = info["profit_cumsum"][-1]
                run.log(
                    {
                        f"milp_eval_total_profit_{model_type}": info["profit_cumsum"][
                            -1
                        ],
                        f"milp_eval_total_reward_{model_type}": info["reward_cumsum"][
                            -1
                        ],
                    }
                )
                del milp_model
                del milp_eval_env
                break

    print(
        f"mean_reward_bestmodel:{eval_mean_reward_best_model:.2f} +/- {eval_std_reward_best_model:.2f}"
    )
    for model_type in model_types:
        print(f"cumulative_profit_{model_type}: ", milp_results[model_type])
