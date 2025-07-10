# battery storage optimisation with Reinforcement Learning
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import torch
import os
from pickle import dump, load
import sys
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import wandb

# import custom classes:
from battery_efficiency import BatteryEfficiency
from battery_degradation_func import calculate_degradation

# import model architecture from model scripts directory
sys.path.append('../data')
sys.path.append('./scripts')

from price_generator import price_generator_id_da as price_generator
class Battery(gym.Env):

    def __init__(self, env_settings, train=True, eval_const_deg=False, alpha_d_constant=0.0):

        self.pr = env_settings['battery_power']
        self.cr = env_settings['battery_capacity']
        self.cost = env_settings['battery_price']
        self.num_actions = env_settings['num_actions']
        self.ep_len = env_settings['episode_length']
        self.standby_loss = env_settings['standby_loss']
        self.train = train
        self.train_data_path = env_settings['train_data_path']
        self.test_data_path = env_settings['test_data_path']
        self.prices_id_norm = None
        self.prices_id_true = None
        self.prices_da_norm = None
        self.prices_da_true = None
        self.da_market_tracker_ts = 0
        self.da_market_tracker = False
        self.da_market_tracker_series = None
        self.da_plan = None
        self.alpha_d = 0  # degradation coefficient
        self.soc = 0  # variable to track soc
        self.soc_initial = 0.5  # initial state of charge
        self.ts = 0  # timestep within each episode
        self.ep = 0  # episode increment
        self.ep_pwr = 0  # total absolute power per each episode
        self.ep_end_kWh_remain = 0  # episode end charge
        self.kWh_cost = 127.59  # battery cost per kWh Euro, 139 US Dolar, Wechselkurs (0.92, 19.05.2024) [https://about.bnef.com/blog/lithium-ion-battery-pack-prices-hit-record-low-of-139-kwh/]
        self.done = False
        self.total_ts = 0
        self.cycle_num = 0
        self.ep_start_kWh_remain = self.cr
        self.seed = env_settings['seed']
        self.battery_degradation = env_settings['degradation']
        self.train_or_test = ('train' if self.train else 'test')
        self.eval_milp = env_settings['eval_milp'] if 'eval_milp' in env_settings else False
        self.calc_eff = env_settings['calc_eff'] if 'calc_eff' in env_settings else False
        self.hour_horizon = env_settings['hour_horizon']
        self.resolution = env_settings['resolution']  # QH or H
        self.resolution_per_hour = 4 if self.resolution == 'QH' else 1
        self.horizon = self.hour_horizon * self.resolution_per_hour
        self.num_episode = 0
        self.da_tracking = []
        self.id_tracking = []
        self.observation = {}
        self.new_ep = False
        self.da_reward_at_dec = env_settings['da_reward_at_dec'] if 'da_reward_at_dec' in env_settings else False
        self.printed_out = False
        # add a list to store the degradation coefficients
        self.degradation_coefficients = []
        self.alpha_d_constant = alpha_d_constant
        self.eval_const_deg = eval_const_deg

        # limits that will be checked inside of code to ensure that the battery is not overcharged or undercharged
        self.lim = {
            'bat_cap':      [0.0, self.cr],
            'bat_pow':      [-self.pr, self.pr],
            'soc':          [0.2, 1.0],
            'char_eff':     [0.0, 1.0],
            'dischar_eff':  [0.0, 1.0],
            'reward_func':  [-np.inf, np.inf]
        }
        # ---- Initialising environment settings ---- #
        # gymnasium specific observation space for the battery environment

        # ----- observation space for Intraday and DA Logic additional information ----
        # ---- combining the logic of both observation spaces ----
        self.observation_space = gym.spaces.Dict({
            # ---- general observation space ----
            'da_prices':    gym.spaces.Box(low=0, high=1, shape=(self.horizon,), dtype=np.float32), # tested
            'id_prices':    gym.spaces.Box(low=0, high=1, shape=(self.horizon,), dtype=np.float32), # tested
            'da_plan':      gym.spaces.Box(low=-1, high=1, shape=(self.horizon,), dtype=np.float32), # sparsely tested
            'soc':          gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32), # tested
            'alpha_d':      gym.spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            # ---- Request observation space ----
            'storage_time': gym.spaces.Discrete(self.horizon), # tested
            'power':        gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32), # tested
            'duration':     gym.spaces.Discrete(self.horizon), # tested
            'price':        gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32), # tested
            'label':        gym.spaces.Discrete(2) # tested
        })

        # keeping track of some variables not been changed every step
        self.observation = self.observation_space.sample()

        # ---- Creating the discretized action space ----
        num_actions = self.num_actions if self.num_actions > 3 else 3
        self.action_space = gym.spaces.Discrete(num_actions)
        self._action_to_direction = {i: np.around(-1 + 2 * i / (num_actions - 1), 6) for i in range(num_actions)}
        # ---- end of initialising observation and action space ----

        # initialise degradation class - battery capacity watts
        self.batt_deg = BatteryEfficiency(self.cr * 1000)  # watts as input

        # load new data generator
        self.price_generator = price_generator(train_path=self.train_data_path, test_path=self.test_data_path)

        # initialise observation
        self.observation = self.observation_space.sample()

    def step(self, action):
        # truncated boolean to check if the episode is truncated
        truncated = False

        if self.new_ep:  # if new episode, calculate the start capacity for degradation
            start_ep_capacity = calculate_degradation(self.cycle_num)
            self.ep_start_kWh_remain = (start_ep_capacity / 100) * self.cr
            self.new_ep = False

        # Check observation space for errors, da_prices, id_prices, soc, power.
        self.check_for_errors_in_observation_space(pre_da_market_tracker=True)

        # da_market_tracker == True means handling DA market decision
        if self.da_market_tracker:
            # ---- Add action to the planed power ----
            storage_time = self.observation['storage_time']
            duration = self.observation['duration']
            action_da = self._action_to_direction[action.item()]
            action_da_kw = action_da * self.pr
            action_da_kwh = action_da_kw * duration / self.resolution_per_hour
            # no physical action is taken
            efficiency = 0
            # Check if self.da_future_power_observation_space is not None
            if self.observation['da_plan'][storage_time:(storage_time + duration)].all() == 0:
                self.observation['da_plan'][storage_time:(storage_time + duration)] = action_da
            else:
                print("Error in updating the planed DA Schedule")
                truncated = True
                sys.exit(1)
            # ---- Calculating the price for the request power ----
            ts_price_da_mw = self.prices_da_true[self.ts + storage_time]
            ts_price_da_kw = ts_price_da_mw / 1000
            if self.da_reward_at_dec:
                reward = ts_price_da_kw * action_da_kwh - self.alpha_d * abs(action_da_kwh)
            else:
                reward = 0
            ## Check observation space for da_market_tracker, time stamp, duration, label, storage_time
            self.check_for_errors_in_observation_space(pre_da_market_tracker=False)

            self.da_market_tracker_ts += 1
            if self.da_market_tracker_ts >= 23:
                self.da_market_tracker = False

            info = {'label': 0,
                    'action_da': action_da,
                    'action_da_kwh': action_da_kwh,
                    'ts_price_da_kw': ts_price_da_kw,
                    'reward': reward,
                    }
        # Here to potential ways need to be respected.
        elif not self.da_market_tracker:   # ---- Handling the id auction ----
            # ---- Add action to the planed power ----
            storage_time = self.observation['storage_time']
            duration = self.observation['duration']
            action_sum = self._action_to_direction[action.item()]
            action_sum_kw = (action_sum * self.pr)
            action_sum_kwh = action_sum_kw * duration / self.resolution_per_hour
            action_da_kw = self.observation['da_plan'][0] * self.pr
            action_da_kwh = action_da_kw * 1 / self.resolution_per_hour # in this case only looking at one qh.
            # -- Check observation space for power, duration, label, storage_time
            self.check_for_errors_in_observation_space(pre_da_market_tracker=False)
            # efficiency calculation
            if self.calc_eff:
                efficiency = self.batt_deg.calc_efficiency_all(self.soc, 4 * action_sum_kw) # to get the power instead of energy
            else:
                efficiency = 1.0
            # added check for efficiency
            if action_sum > 0:
                truncated = False if self.lim['dischar_eff'][0] <= efficiency <= self.lim['dischar_eff'][1] else True
            elif action_sum < 0:
                truncated = False if self.lim['char_eff'][0] <= efficiency <= self.lim['char_eff'][1] else True
            action_sum_kwh_clipped = self._clip_action_batt_limits(action_sum_kwh, self.soc, efficiency)
            # calculating next SoC
            self.soc = self._next_soc(self.soc, efficiency, action_sum_kwh_clipped, self.standby_loss)
            # ---- Calculating the price for the request power ----
            action_id_kwh = action_sum_kwh_clipped - action_da_kwh
            # Furthermore I need to respect the degradation part
            action_kwh_deg = abs(action_sum_kwh_clipped) - abs(action_da_kwh)
            # ---- Calculating the price for the request power ----
            ts_price_id_mw = self.prices_id_true[self.ts]
            ts_price_da_mw = self.prices_da_true[self.ts]
            ts_price_id_kw = ts_price_id_mw / 1000
            ts_price_da_kw = ts_price_da_mw / 1000
            if self.da_reward_at_dec:
                reward = (ts_price_id_kw * action_id_kwh) - (self.alpha_d * action_kwh_deg)
            else:
                reward = (ts_price_id_kw * action_id_kwh + ts_price_da_kw * action_da_kwh) - (self.alpha_d * abs(action_sum_kwh_clipped))
            # determine if charge/discharge is out of bounds, i.e. <20% and >100% else fail episode
            truncated = True if np.around(self.soc, 2) < self.lim['soc'][0] or np.around(self.soc, 2) > self.lim['soc'][1] else truncated
            if truncated:
                print("error - limits breached")
                # updating
            self.ep_pwr += abs(action_sum_kwh_clipped)
            self.cycle_num += (abs(action_sum_kwh_clipped) / self.cr) / 2

            info = {
                'label':            np.array([1]),
                'ts':               np.array([self.ts]),
                'action_sum':       np.array([action]),
                'action_sum_kwh':   np.array([action_sum_kwh]),
                'reward':           np.array([reward]),
                'action_id':        np.array([action_id_kwh]),
                'action_da':        np.array([action_da_kwh]),
                'price_id':         np.array([ts_price_id_kw]),
                'price_da':         np.array([ts_price_da_kw]),
                'soc':              np.array([self.soc]),
                'efficiency':       np.array([efficiency]),
                'cycle_num':        np.array([self.cycle_num]),
                'alpha_d':          np.array([self.alpha_d]),
            }
            self.ts += 1
            self.total_ts += 1
            # I think I Need here to market checker to be set to True if
            self.da_market_tracker = self.da_market_tracker_series[self.ts]
        else:
            print("self.da_market_tracker is not a boolean value.")
            truncated = True

        # ---- Update the observation space at 12 pm ----
        if self.da_market_tracker:
            # ---- run through all da bids ----
            observation = self.get_observation_dict_step_da()
        elif not self.da_market_tracker:
            observation = self.get_observations_dict_step_ts()
            self.da_market_tracker_ts = 0
        else:
            print("self.da_market_tracker is not a boolean value.")
            truncated = True
        terminated = True if self.ts >= (self.ep_len - 1) else False
        self.num_episode += 1  # not equal to ts because da ahead biding is an episode but not time stamp.
        # scale reward for quicker learning
        reward = np.around(reward / 100, 4)

        # observation only contains the updated parameters of the observations space.
        self.observation.update(observation)
        self.done = terminated or truncated
        return self.observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)
        # update final charge for episode

        # calculating the end of episode capacity, needed for alpha_d calculation.
        end_ep_capacity = calculate_degradation(self.cycle_num)
        self.ep_end_kWh_remain = (end_ep_capacity / 100) * self.cr

        # calculate degradation coefficient for the up coming episode - iff battery degradation is enabled
        self._degrade_coeff()

        # ---- Reset some tracking variables ----
        # Reinitialise episode power (ep_pwr) to zero - needed for iteratively calculating degradation
        self.ep_pwr = 0
        self.num_step = 0
        self.ts = 0
        self.soc = self.soc_initial

        # Assume battery is refreshed after 4000 cycles
        if self.cycle_num >= 4000:
            self.cycle_num = 0

        # Grabbing prices for the episode.
        self.prices_id_norm, self.prices_id_true, self.prices_da_norm, self.prices_da_true, self.da_market_tracker_series \
            = self.price_generator.get_timeseries(self.ep_len, self.horizon, train_or_test=self.train_or_test)
        # ---- Resetting the main part the observation space ----
        self.observation.update(self.get_observation_dict_reset())
        self.da_market_tracker = self.da_market_tracker_series[self.ts]
        # Initial plan: First intraday request.

        # reinitialise episode parameters
        self.done = False
        self.new_ep = True
        self.ep += 1
        if not self.printed_out:
            print('self.da_reward_at_dec: ', self.da_reward_at_dec)
            self.printed_out = True
        info = {}
        return self.observation, info

    ################################### Subfunctions #########################################################################
    def get_observation_dict_reset(self):
        observation = {
            'da_prices':    self.prices_da_norm[self.ts:self.ts + self.horizon],
            'id_prices':    self.prices_id_norm[self.ts:self.ts + self.horizon],
            'da_plan':      np.array(np.zeros((self.horizon,)), dtype=np.float32),
            'soc':          np.array([self.soc_initial], dtype=np.float32), # does it need to be inital here?
            'alpha_d':      np.array([np.around(self.alpha_d,6)], dtype=np.float32),
            # ---- Request observation space ----
            'storage_time': 0,
            'power':        np.array([1], dtype=np.float32),
            'duration':     1,
            'price':        np.array([self.prices_id_norm[self.ts]], dtype=np.float32),
            'label':        1
        }
        return observation

    def get_observations_dict_step_ts(self):
        da_plan = self.observation["da_plan"]
        da_plan[:-1] = da_plan[1:]
        da_plan[-1] = 0
        observation = {
            'da_prices':    self.prices_da_norm[self.ts:self.ts + self.horizon],
            'id_prices':    self.prices_id_norm[self.ts:self.ts + self.horizon],
            'da_plan':      da_plan,
            'soc':          np.array([self.soc], dtype=np.float32),
            # ---- Request observation space ----
            'storage_time': 0,
            'power':        np.array([1], dtype=np.float32),
            'duration':     1,
            'price':        np.array([self.prices_id_norm[self.ts]], dtype=np.float32),
            'label':        1
        }
        return observation

    def get_observation_dict_step_da(self):
        # need to to shift the da_plan if its the first da request at 12 pm.
        if self.da_market_tracker_ts == 0: # only need to shift at first decision of da.
            da_plan = self.observation["da_plan"]
            da_plan[:-1] = da_plan[1:]
            da_plan[-1] = 0
        else:
            da_plan = self.observation["da_plan"]
        delta = 12 * self.resolution_per_hour
        observation = {
            'da_prices':    self.prices_da_norm[self.ts:self.ts + self.horizon],
            'id_prices':    self.prices_id_norm[self.ts:self.ts + self.horizon],
            'da_plan':      da_plan,
            'soc':          np.array([self.soc], dtype=np.float32),
            # ---- Request observation space ----
            'storage_time': delta + self.da_market_tracker_ts * 4,
            'power':        np.array([1], dtype=np.float32),
            'duration':     4,
            'price':        np.array([self.prices_da_norm[self.ts + delta + self.da_market_tracker_ts * 4]], dtype=np.float32),
            'label':        0
        }
        return observation

    def _next_soc(self, soc_t, efficiency, action, standby_loss): # action is
        e_ess = self.cr
        if np.around(soc_t, 2) == 0 and action == 0:
            next_soc = soc_t
        elif action < 0:  # action <0 means charge
            next_soc = soc_t - (1 / e_ess) * efficiency * (action)
        elif action > 0:
            next_soc = soc_t - (1 / e_ess) * (1 / efficiency) * (action)
        elif action == 0:
            next_soc = soc_t - (1 / e_ess) * (standby_loss) * (1 / self.resolution_per_hour)
        return np.around(next_soc, 6)

    def _degrade_coeff(self):
        if self.battery_degradation == True:
            if self.eval_const_deg == True:
                self.alpha_d = self.alpha_d_constant
            else:
                if self.ep_pwr == 0:
                    self.alpha_d = 0.0
                else:
                    self.alpha_d = ((self.ep_start_kWh_remain - self.ep_end_kWh_remain) / self.ep_pwr) * self.kWh_cost
                    self.degradation_coefficients.append(self.alpha_d)
        elif self.battery_degradation == False:
            self.alpha_d = 0.0
        else:
            print("No degradation coefficient found")
            sys.exit(1)

    def _clip_action_batt_limits(self, action_kwh, current_soc, efficiency):
        # clip charge / discharge relative to SoC limits
        upper_lim = ((current_soc - self.lim['soc'][0]) * self.cr) / (efficiency * 1)
        lower_lim = ((current_soc - self.lim['soc'][1]) * self.cr) / (efficiency * 1)

        # clip action to ensure within limits
        action_kwh_clipped = np.clip(action_kwh, lower_lim, upper_lim)
        return action_kwh_clipped


    def check_for_errors_in_observation_space(self, pre_da_market_tracker = True):
        if pre_da_market_tracker:
            if not np.array_equal(self.observation['da_prices'], self.prices_da_norm[self.ts: self.ts + self.horizon]):
                print("Error in da_prices")
                print(self.observation['da_prices'])
                print(self.prices_da_norm[self.ts: self.ts + self.horizon])
                truncated = True
            # checking if id_prices are right
            if not np.array_equal(self.observation['id_prices'],self.prices_id_norm[self.ts: self.ts + self.horizon]):
                print("Error in id_prices")
                print(self.observation['id_prices'])
                print(self.prices_id_norm[self.ts: self.ts + self.horizon])
                truncated = True
                sys.exit(1)
            if self.observation['soc'] != self.soc:
                print("Error in soc")
                truncated = True
                sys.exit(1)
            if self.observation['power'] != 1:
                print("Error in power")
                truncated = True
        else:
            if self.da_market_tracker:
                if self.da_market_tracker != self.da_market_tracker_series[self.ts]:
                    print("Error in da_market_tracker")
                    truncated = True
                if (self.ts + 48) % 96 != 0:
                    print("Error in time stamp")
                    truncated = True
                if self.observation['duration'] != 4:
                    print("Error in duration")
                    truncated = True
                if self.observation['label'] != 0:
                    print("Error in label")
                    truncated = True
                if self.observation['storage_time'] != (12 * self.resolution_per_hour) + self.da_market_tracker_ts * 4:
                    print("Error in storage time")
                    truncated = True
                if self.observation['price'] != self.prices_da_norm[self.ts + 12 * self.resolution_per_hour + self.da_market_tracker_ts * 4]:
                    print("Error in price")
                    truncated = True
            elif not self.da_market_tracker:
                if self.observation['power'] != 1:
                    print("Error in power")
                    truncated = True
                if self.observation['duration'] != 1:
                    print("Error in duration")
                    truncated = True
                if self.observation['label'] != 1:
                    print("Error in label")
                    truncated = True
                if self.observation['storage_time'] != 0:
                    print("Error in storage time")
                    truncated = True
            else:
                print("Error in da_market_tracker")
                truncated = True
                sys.exit(1)

    def get_average_degradation_coefficients(self):
        if len(self.degradation_coefficients) > 0:
            return sum(self.degradation_coefficients) / len(self.degradation_coefficients)
        else:
            return None
