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

# Two options, last in folder or specific data, to use the last in folder, just leave the string empty
directory = 'test'
# if folder name == '', the folders in between starting and ending date are used, else the last folder.
# get first 4 elements of directory string
folder_name = ''
starting_date_time = '2024-04-28_00-00'  # Corrected date
ending_date_time = '2024-04-29_08-00'  # Corrected date

start_date = datetime.strptime(starting_date_time, '%Y-%m-%d_%H-%M') if starting_date_time != '' else None
end_date = datetime.strptime(ending_date_time, '%Y-%m-%d_%H-%M') if ending_date_time != '' else None
selected_directories = []

# if folder_name is empty, the last folder will be used
if folder_name == '':
    if starting_date_time == '' and ending_date_time == '':
        selected_directories.append(os.listdir(f"./runs/da_id/{directory}")[-1])
    else:
        for folder in os.listdir(f"./runs/da_id/{directory}"):
            # Convert the folder name to a datetime object
            folder_date = datetime.strptime(folder, '%Y-%m-%d_%H-%M')

            # Check if the folder's date is within the range
            if start_date <= folder_date <= end_date:
                # If it is, add it to the list
                selected_directories.append(folder)
else:
    selected_directories.append(folder_name)

if torch.cuda.is_available():
    print("CUDA is available!")
else:
    print("CUDA is not available.")

if __name__ == '__main__':

    for selected_dict in selected_directories:
        print('selected_dict: ', selected_dict)
        model_dict = f"./runs/da_id/{directory}/{selected_dict}/model/"
        milp_eval_dict = f"./runs/da_id/{directory}/{selected_dict}/milp_eval"
        # check if folder "milp_eval" exists if not create it
        os.makedirs(milp_eval_dict, exist_ok=True)

        settings = load_settings(f"./runs/da_id/{directory}/{selected_dict}/config.yaml")
        settings = update_settings(settings)
        settings['environment'].update({"episode_length": (292 * 24 * 4)})  # 292 in MILP comparison
        for model_type in ['best_model', 'model']:
            # load env and bring into specific mode, running through the eval data to see how it performs
            milp_eval_env = Battery(settings["environment"], train=False)
            milp_eval_env = sb3.common.monitor.Monitor(milp_eval_env, settings['milp_eval_logdir'])
            obs, info_ = milp_eval_env.reset()
            # join settings['models_dir'] with best_model.zip
            model_path = os.path.join(model_dict, f"{model_type}.zip")
            # obs, info = eval_env.reset() # adding a variable to make clear it's the evaluation mode
            print('model_path: ', model_path)
            model = sb3.DQN.load(model_path)
            model.set_env(milp_eval_env)

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
            # Evaluate the model
            while True:
                action, _states = model.predict(obs, deterministic=True)
                obs, rewards, terminated, truncated, info_ = milp_eval_env.step(action)
                if info_['label'] == 1:
                    for key in info.keys():
                        info[key] = np.append(info[key], info_[key])
                #
                if terminated or truncated:
                    end_time = time.time()
                    info_path = os.path.join(milp_eval_dict, f"info_{model_type}.pkl")
                    execution_time = end_time - start_time
                    info["execution_time"] = execution_time
                    info['profit'] = info['action_da'] * info['price_da'] + info['action_id'] * info['price_id']
                    info['profit_cumsum'] = np.cumsum(info['profit'])
                    with open(info_path, "wb") as f:
                        dump(info_, f)
                    print("model Name: ", model_type)
                    print("truncated: ", truncated, "terminated: ", terminated)
                    print('Done')
                    print(f"The code executed in {execution_time} seconds")
                    print("cumulative reward: ", info['profit_cumsum'][-1])
                    print("-----------------------")
                    print("-----------------------")
                    del model
                    del milp_eval_env
                    break
    print("Done")