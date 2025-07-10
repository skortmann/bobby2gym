#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from battery_milp_da_id_gurobi import solve

data_df = pd.read_csv('../../data/processed_data/DE_DA_QH_ID_test_milp_adjusted_010_090.csv')

battery_power = 10
battery_capacity = 20
soc_initial = 0.5 * battery_capacity
soc_minimal = 0.2 * battery_capacity
soc_maximal = 1.0 * battery_capacity
soc_final = 0.5 * battery_capacity

variables_df = solve(data_df, battery_power=battery_power, soc_initial=soc_initial, soc_minimal=soc_minimal, soc_maximal=soc_maximal, soc_final=soc_final, first_day_no_da=True)

variables_df['profit'] = variables_df['price_id'] * (variables_df['energy_out_id'] - variables_df['energy_in_id']) + variables_df['price_da'] * (variables_df['energy_out_da'] - variables_df['energy_in_da'])
variables_df['profit_cumsum'] = variables_df['profit'].cumsum()

variables_df.to_csv('../../results/MILP/MILP_absolut_results.csv')
