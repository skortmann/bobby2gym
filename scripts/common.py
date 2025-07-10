# common.py
import yaml
import os
from datetime import datetime
import wandb
import time
import random
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
import numpy as np
import warnings
from torch.optim.optimizer import Optimizer

from typing import TYPE_CHECKING


try:
    from tqdm import TqdmExperimentalWarning

    # Remove experimental warning
    warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
    from tqdm.rich import tqdm
except ImportError:
    # Rich not installed, we only throw an error
    # if the progress bar is used
    tqdm = None


if TYPE_CHECKING:
    pass


def load_training_settings_from_yaml(file_name_in_config_folder, da_or_da_id="da"):
    if da_or_da_id not in ["da", "da_id"]:
        raise ValueError("Please specify if da or da_id.")
    current_dir = os.getcwd()
    # Check if 'scripts' is in the current directory
    if "scripts" in current_dir:
        file_path = os.path.join("config", da_or_da_id, file_name_in_config_folder)
    else:
        file_path = os.path.join(
            "scripts", "config", da_or_da_id, file_name_in_config_folder
        )
    with open(file_path, "r") as file:
        settings = yaml.safe_load(file)
    return settings


def load_config(config_path):
    """Loads a YAML configuration file and returns it as a dictionary."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_eval_settings_from_yaml(file_name_in_run_folder):
    with open(file_name_in_run_folder, "r") as file:
        settings = yaml.safe_load(file)
    return settings


def update_settings(settings, da_or_da_id="da", set_time="False", run_id=""):
    current_dir = os.getcwd()
    if da_or_da_id not in ["da", "da_id"]:
        raise ValueError("Please specify if da or da_id.")
    if da_or_da_id == "da":
        decision_per_hour = 1
    elif da_or_da_id == "da_id":
        decision_per_hour = 5  # 4 intraday and 1 day ahead
    hours_per_day = 24
    # correcting data paths.
    if "scripts" in current_dir:
        prefix_data = "../data/"
        # prefix_dicts = "./"
    else:
        prefix_data = "./data/"
        # prefix_dicts = "./scripts/"
    # updating further parameters.
    settings["episode_length"] = (
        settings["days_per_episode"] * decision_per_hour * hours_per_day
    )
    settings["total_timesteps"] = int(
        settings["total_episodes"] * settings["episode_length"] * (7 / 5)
    )
    settings["environment"].update(
        {
            "train_data_path": os.path.join(
                prefix_data,
                "processed_data",
                settings["environment"]["train_data_name"],
            ),
            "test_data_path": os.path.join(
                prefix_data, "processed_data", settings["environment"]["test_data_name"]
            ),
            "episode_length": settings["episode_length"],
            "hour_horizon": settings["hour_horizon"],
        }
    )
    if set_time:
        time.sleep(random.uniform(0, 10))
        time_idx = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        settings.update(
            {
                "time": time_idx,
            }
        )
    else:
        time_idx = settings["time"]

    if run_id != "":
        run_name = run_id  # wandb id
    else:
        run_name = time_idx

    directory = os.path.join(settings["directory"], settings["subdirectory"])

    # settings['models_dir'] = f"{prefix_dicts}/runs/{da_or_da_id}/{directory}/{run_name}/model"
    # settings['logdir'] = f"{prefix_dicts}/runs/{da_or_da_id}/{directory}/{run_name}/log"
    # settings['eval_logdir'] = f"{prefix_dicts}/runs/{da_or_da_id}/{directory}/{run_name}/eval_log"
    # settings['config_dir'] = f"{prefix_dicts}/runs/{da_or_da_id}/{directory}/{run_name}/"
    # settings['milp_eval'] = f"{prefix_dicts}/runs/{da_or_da_id}/{directory}/{run_name}/milp_eval"
    # settings['milp_eval_logdir'] = f"{prefix_dicts}/runs/{da_or_da_id}/{directory}/{run_name}/milp_eval_log"

    settings["models_dir"] = f"output/runs/{directory}/{run_name}/trained_model"
    settings["config_dir"] = f"output/runs/{directory}/{run_name}/"
    settings["milp_eval"] = f"output/runs/{directory}/{run_name}/results_eval"

    return settings


# Assuming wandb is already initialized and config is set
def get_settings_from_wandb_config():
    settings = {}
    for key in wandb.config.keys():
        if key not in ["environment", "dqn_kwargs"]:
            settings[key] = wandb.config[key]
        else:
            settings[key] = {}
            for sub_key in wandb.config[key].keys():
                settings[key][sub_key] = wandb.config[key][sub_key]
    return settings


class ReduceLROnPlateauCallback(BaseCallback):
    def __init__(self, check_every, reduce_factor, patience, min_lr=0, verbose=0):
        super(ReduceLROnPlateauCallback, self).__init__()
        self.check_every = check_every
        self.reduce_factor = reduce_factor
        self.patience = patience
        self.min_lr = min_lr
        self.verbose = verbose
        self.best_loss = np.inf
        self.patience_counter = 0
        self.lr_scheduler = None

    def _on_training_start(self) -> None:
        self.lr_scheduler = self.model.get_lr_schedule()

    def _on_step(self) -> bool:
        if self.n_calls % self.check_every == 0:
            logs = self.model.get_eval_log()
            if logs is not None and logs["eval_loss"] < self.best_loss:
                self.best_loss = logs["eval_loss"]
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    current_lr = self.lr_scheduler(self._current_progress_remaining)
                    new_lr = max(self.min_lr, self.reduce_factor * current_lr)
                    self.model.lr_schedule = lambda _: new_lr
                    if self.verbose:
                        print(f"Reduced learning rate to {new_lr}.")
                    self.patience_counter = 0
        return True


class UpdateLearningRateOnNoModelImprovement(BaseCallback):
    """
    Stop the training early if there is no new best model (new best mean reward) after more than N consecutive evaluations.

    It is possible to define a minimum number of evaluations before start to count evaluations without improvement.

    It must be used with the ``EvalCallback``.

    :param max_no_improvement_evals: Maximum number of consecutive evaluations without a new best model.
    :param min_evals: Number of evaluations before start to count evaluations without improvements.
    :param verbose: Verbosity level: 0 for no output, 1 for indicating when training ended because no new best model
    """

    parent: EvalCallback

    def __init__(
        self,
        optimizer,
        max_no_improvement_evals: int,
        min_evals: int = 0,
        verbose: int = 0,
        start_lr=1,
        min_lr=0,
        factor=0.1,
        threshold=1e-4,
        threshold_mode="rel",
        eps=1e-8,
    ):
        super().__init__(verbose=verbose)
        if not isinstance(optimizer, Optimizer):
            raise TypeError(f"{type(optimizer).__name__} is not an Optimizer")
        self.optimizer = optimizer
        self.max_no_improvement_evals = max_no_improvement_evals
        self.min_evals = min_evals
        self.last_best_mean_reward = -np.inf
        self.no_improvement_evals = 0
        self.lr = float(start_lr)
        self.min_lr = float(min_lr)
        self.eps = float(eps)
        if factor >= 1.0:
            raise ValueError("Factor should be < 1.0.")
        self.factor = factor
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.verbose = verbose
        print("-----------------------------------------------")
        print("Initialized UpdateLearningRateOnNoModelImprovement callback.")

    def _on_step(self) -> bool:
        assert (
            self.parent is not None
        ), "``UpdateLearningRateOnNoModelImprovement`` callback must be used with an ``EvalCallback``"

        update_learning_rate = False
        if self.n_calls > self.min_evals:
            if self.parent.best_mean_reward > self.last_best_mean_reward:
                self.no_improvement_evals = 0
            else:
                self.no_improvement_evals += 1
                if self.no_improvement_evals > self.max_no_improvement_evals:
                    update_learning_rate = True
                    self.no_improvement_evals = 0

        self.last_best_mean_reward = self.parent.best_mean_reward

        if self.verbose >= 1 and update_learning_rate:
            self._reduce_lr(self.optimizer)

        return True

    def _reduce_lr(self, optimizer):
        old_lr = self.model.learning_rate
        print("Old learning rate: ", old_lr)
        new_lr = max(old_lr * self.factor, self.min_lr)
        print("New learning rate: ", new_lr)
        if old_lr - new_lr > self.eps:
            self.model.learning_rate = new_lr
            self.model._setup_lr_schedule()
            self.model._update_learning_rate(optimizer)
            print(self.model.policy.optimizer)

            if self.verbose >= 1:
                print(f"Learning rate reduced to: {new_lr}")
                print(
                    "The stored learning rate in self.model.learning_rate: ",
                    self.model.learning_rate,
                )
