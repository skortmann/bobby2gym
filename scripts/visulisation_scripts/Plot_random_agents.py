#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import enlopy as el
import os
import matplotlib
from matplotlib import pyplot as plt

visualisation_path = '../../visualisations'
filename = 'profit_distribution_random_agents.pdf'

save_to_csv = True
csv_directory = '../../results/CSV'

sum_ticks = 6
i_th = 64 # every 64 QH is displayed in the DataFrame.

colors = {3: ['#d8b365', '#f5f5f5', '#5ab4ac'], 4: ['#a6611a', '#dfc27d', '#80cdc1', '#018571'], 5: ['#a6611a', '#dfc27d', 'f5f5f5', '#80cdc1', '#018571']}
percentiles = [10, 25, 40, 50, 60, 75, 90]


input_data_path = '../../data/processed_data/DE_DA_QH_ID_test_adjusted_010_090.csv'
path = '../runs/da_id/random_decision_agent/random_decision/random_agent/milp_eval'

columns = []
for i in range(1000):
    # read the file
    file = os.path.join(path, f'info_{i}.pkl')
    if os.path.exists(file):
        info = pd.read_pickle(file)
        # Convert numpy ndarray to pandas Series and rename it
        series = pd.Series(np.around(info['profit_cumsum'], 2), name=f'profit_{i}')
        columns.append(series)

df_reward = pd.concat(columns, axis=1)

input_data = pd.read_csv(input_data_path, index_col=0)

# Trim the 'von' column in input_data to the same length as df_reward
# Access the 'von' index in input_data and trim it to the same length as df_reward
trimmed_von = input_data.index[:len(df_reward)]

# Set the index of df_reward to be the trimmed 'von' index
df_reward.index = pd.to_datetime(trimmed_von)
df_reward.index = pd.to_datetime(df_reward.index, unit='qh')

import matplotlib.pyplot as plt
import numpy as np

def calulate_percentiles(df, percentiles, colors, sum_ticks=5):
    import matplotlib.pyplot as plt
    import numpy as np

    # Always include the 50th percentile
    if 50 not in percentiles:
        percentiles.append(50)
        
    # Calculate percentiles and complementary percentiles
    data_percentiles = {percentile: np.percentile(df, percentile, axis=1) for percentile in percentiles}
    data_percentiles.update({100 - percentile: np.percentile(df, 100 - percentile, axis=1) for percentile in percentiles if percentile != 50})
    return percentiles, data_percentiles

    # Plot filled areas and line
def visualise_percentiles(df, data_percentiles, percentiles , colors, sum_ticks,  i_th=4): 
    for i, percentile in enumerate(sorted(percentiles)):
        if percentile != 50:
            plt.fill_between(df.index[::i_th], data_percentiles[percentile][::i_th], data_percentiles[100 - percentile][::i_th], color=colors[len(percentiles)][i], alpha=0.5, label=f'{percentile}th-{100 - percentile}th percentile')
    plt.plot(df.index[::i_th], data_percentiles[50][::i_th], color='black', label='50th percentile')
    x_ticks = int( len(df) /  (sum_ticks - 1))
    # plt.title('Statistical distribution of the profit of 1000 random decision agents')
    plt.xlabel('Time')
    plt.ylabel('Profit [€]')
    plt.xticks(df.index[::x_ticks])
    plt.legend()
    plt.savefig(os.path.join(visualisation_path, filename))
    plt.show()

percentiles, data_percentiles = calulate_percentiles(df_reward, percentiles, colors)

# Call the function
visualise_percentiles(df_reward, data_percentiles, percentiles, colors,sum_ticks=6, i_th=64)

def save_percentiles_to_csv(data_percentiles, df_input, i_th, directory):
    # Ensure the directory exists
    os.makedirs(directory, exist_ok=True)
    df = pd.DataFrame()
    for percentile, data in data_percentiles.items():
        # Downsample the data and add it to the DataFrame
        df[percentile] = data[::i_th]

    # Save the DataFrame to a CSV file
    df.index = df_input.index[::i_th]
    # and rename the index with 'Time'
    df.index.name = 'Time'
    df.to_csv(os.path.join(directory, 'percentiles_random_agent.csv'))
    return df.sort_index(axis=1)

if save_to_csv: 
    df = save_percentiles_to_csv(data_percentiles=data_percentiles,df_input=df_reward, i_th=i_th,directory=csv_directory)

