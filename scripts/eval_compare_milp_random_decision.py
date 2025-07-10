# battery storage optimisation with Reinforcement Learning
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from tqdm.auto import tqdm
from matplotlib import cm
from pickle import dump
from datetime import datetime
import stable_baselines3 as sb3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
import wandb
import time


from wandb.integration.sb3 import WandbCallback
import os
from tqdm import tqdm

# import custom classes:
try:
    from battery_efficiency import BatteryEfficiency
    from battery_environment import Battery
    from common import load_eval_settings_from_yaml as load_settings, update_settings
    print("Imports were successful.")
except ImportError:
    print("Imports were not successful.")
    sys.exit(1)

current_directory = os.getcwd()
print(current_directory)
if 'scripts' in current_directory:
    prefix = './'
else:
    prefix = './scripts/'

# Two options, last in folder or specific data, to use the last in folder, just leave the string empty

directory = 'da_id'
subdirectory = 'random_decision_agent'
subsubdirectory = 'random_decision'
# if folder name == '', the folders in between starting and ending date are used, else the last folder.
# get first 4 elements of directory string

selected_directories = []

# if folder_name is empty, the last folder will be used
directory_look_up = os.path.join (prefix, 'runs', directory, subdirectory, subsubdirectory)
print('directory_look_up: ', directory_look_up)
for folder in os.listdir(directory_look_up):
    selected_directories.append(folder)

if torch.cuda.is_available():
    print("CUDA is available!")
else:
    print("CUDA is not available.")
print(selected_directories)
if __name__ == '__main__':
    for selected_dict in selected_directories:
        print('selected_dict: ', selected_dict)
        milp_eval_dict = os.path.join(directory_look_up, selected_dict, 'milp_eval')
        config_path = os.path.join(directory_look_up, selected_dict, 'config.yaml')
        # check if folder "milp_eval" exists if not create it
        os.makedirs(milp_eval_dict, exist_ok=True)
        print('milp_eval_dict', milp_eval_dict)

        settings = load_settings(config_path)
        settings = update_settings(settings)
        settings['environment'].update({"episode_length": (292 * 24 * 4)})  # 292 in MILP comparison
        for i in range(1000):
            # load env and bring into specific mode, running through the eval data to see how it performs
            milp_eval_env = Battery(settings["environment"], train=False)
            milp_eval_env = sb3.common.monitor.Monitor(milp_eval_env, settings['milp_eval_logdir'])
            obs, info_ = milp_eval_env.reset()

            info = {'ts':               np.array([]),  # 'ts': 'timestamp'
                    'action_sum':       np.array([]),
                    'action_sum_kwh':   np.array([]),
                    'reward':           np.array([]),
                    'action_id':        np.array([]),
                    'action_da':        np.array([]),
                    'price_id':         np.array([]),
                    'price_da':         np.array([]),
                    'soc':              np.array([]),
                    'efficiency':       np.array([]),
                    'cycle_num':        np.array([]),
                    }
            start_time = time.time()
            action_space = milp_eval_env.action_space.n
            # Generate a random action
            # Evaluate the model
            while True:
                action =torch.tensor([np.random.randint(0, action_space)])
                obs, rewards, terminated, truncated, info_ = milp_eval_env.step(action)
                if info_['label'] == 1:
                    for key in info.keys():
                        info[key] = np.append(info[key], info_[key])
                #
                if terminated or truncated:
                    end_time = time.time()
                    info_path = os.path.join(milp_eval_dict, f"info_{i}.pkl")
                    execution_time = end_time - start_time
                    info["execution_time"] = execution_time
                    info['profit'] = info['action_da'] * info['price_da'] + info['action_id'] * info['price_id']
                    info['profit_cumsum'] = np.cumsum(info['profit'])
                    with open(info_path, "wb") as f:
                        dump(info, f)
                    print("model Name: ", i)
                    print("truncated: ", truncated, ", terminated: ", terminated)
                    print('Done')
                    print(f"The code executed in {execution_time} seconds")
                    print("cumulative profit: ", info['profit_cumsum'][-1])
                    print("-----------------------")
                    print("-----------------------")
                    del milp_eval_env
                    break
    print("Done")