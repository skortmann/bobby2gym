#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from battery_milp_da_id_gurobi_iterative import solve

data_df = pd.read_csv('../../data/processed_data/DE_DA_QH_ID_test_milp_adjusted_010_090.csv')

da_optimization_horizon = 24 * 4

battery_power = 10
battery_capacity = 20
soc_inital = 0.5 * battery_capacity
soc_minimal = 0.2 * battery_capacity
soc_maximal = 1.0 * battery_capacity
soc_final = 0.5 * battery_capacity

variables_df_da = solve(data_df, soc_initial=soc_inital, battery_power=battery_power, soc_minimal=soc_minimal, soc_maximal = soc_maximal, soc_final=soc_final, first_day_no_da=True, daywise_da=True, energy_in_id=np.zeros(len(data_df)),energy_out_id=np.zeros(len(data_df)))

variables_df = solve(data_df, battery_power=battery_power, soc_initial=soc_inital, soc_minimal=soc_minimal, soc_maximal=soc_maximal,soc_final=soc_final, first_day_no_da=True, energy_in_da=np.array(variables_df_da['energy_in_da']), energy_out_da=np.array(variables_df_da['energy_out_da']))

# where is this profit coming from?
variables_df['profit'] = variables_df['price_da'] * (variables_df['energy_out_da'] - variables_df['energy_in_da']) + variables_df['price_id'] * (variables_df['energy_out_id'] - variables_df['energy_in_id'])
variables_df['profit_cumsum'] = variables_df['profit'].cumsum()
# store results in csv and results as subdirectory MILP 
variables_df.to_csv('../../results/MILP/MILP_iterative_results.csv')