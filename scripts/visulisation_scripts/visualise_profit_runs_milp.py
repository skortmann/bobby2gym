import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import pickle

main_directory = '../runs/da_id'
folder = 'sweep'
subfolder = 'run_results_w_deg_06_03'

battery_capacity_mwh = 20
battery_power_mw = 10

degradation = True
add_milp_results = True
save_to_csv = True

visualisation_path = '../../visualisations'
subpath = "deg" if degradation else "wo_deg"
csv_directory = os.path.join('../../results/CSV', subpath)

colors = {3: ['#d8b365', '#f5f5f5', '#5ab4ac'], 4: ['#a6611a', '#dfc27d', '#80cdc1', '#018571'], 5: ['#a6611a', '#dfc27d', 'f5f5f5', '#80cdc1', '#018571']}

variable = 'efficiency' # ['action_sum_kwh', 'reward', 'action_id','action_da', 'price_id', 'price_da', 'soc', 'efficiency', 'cycle_num','alpha_d', 'execution_time', 'profit', 'profit_cumsum', 'reward_cumsum']
save_in_thousand = False if variable in ['profit_cumsum', 'reward_cumsum'] else True
filename = f"percentiles_results_{'deg' if degradation else 'wo_deg'}{'_in_thousand' if save_in_thousand else ''}"

sum_ticks = 6
i_th = 1
percentiles = [10, 25, 50, 75, 90]

path = os.path.join(main_directory, folder, subfolder)
input_data_path = '../../data/processed_data/DE_DA_QH_ID_test_adjusted_010_090.csv'
input_data = pd.read_csv(input_data_path, index_col=0)

milp_max_path = '../../results/MILP/MILP_absolut_results.csv'
milp_fair_comp_path = '../../results/MILP/MILP_iterative_results.csv'
# load the milp results 
milp_max = pd.read_csv(milp_max_path, index_col=0)[:-1] # caused by the agent not taking a last step to push information of choosen action to info dict. Potential fixes: change length of MILP opt, change waz of saving data in info dict. add one data information, copying last row, to make the agent take one more time step. 
milp_fair_comp = pd.read_csv(milp_fair_comp_path, index_col=0)[:-1] # some as above.

# for all files in the folder, load the data in the subfolder milp_eval/info_best_model.pkl. There is a dict stored with all necessary information, you you could create a dict as results and in this dict create a df with all keys stored in milp_eval/info_best_model.pkl
def get_results_fro_folder(path):
    results = {}
    for file in os.listdir(path):
        if os.path.isdir(os.path.join(path, file)):
            file_path = os.path.join(path, file, 'milp_eval/info_best_model.pkl')
            if os.path.exists(file_path):
                df = pd.DataFrame()
                # the key is the name of the column and then all their values
                # open pickle its a dict
                with open(file_path, 'rb') as f:
                    value = pickle.load(f)
                    for key in value.keys():
                        df[key] = value[key]
                results[file] = df
    return results

def get_df_tracked_variable_all_runs(results, variable, input_data):
    if variable not in results[list(results.keys())[0]].keys():
        print("Element not tracked in given results")
        return None
    df_variable = pd.DataFrame()
    for key in results.keys():
        df_variable[key] = results[key][variable]
    trimmed_von = input_data.index[:len(df_variable)]

    # Set the index of df_reward to be the trimmed 'von' index
    df_variable.index = pd.to_datetime(trimmed_von)
    df_variable.index = pd.to_datetime(df_variable.index, unit='qh')
    if variable in ['action_sum', 'action_sum_kwh', 'action_da', 'action_id']:
        df_variable = df_variable / 1000
    if variable in ['price_da', 'price_id']:
        df_variable = df_variable * 1000
    return df_variable

def adapt_milp_columns(milp, battery_power_mw, battery_capacity_mwh):
    milp['action_sum'] = milp['energy_out_sum'] - milp['energy_in_sum']
    milp['action_da'] = milp['energy_out_da'] - milp['energy_in_da']
    milp['action_id'] = milp['energy_out_id'] - milp['energy_in_id']
    milp['soc'] = milp['soc'] / battery_capacity_mwh
    return milp

def calculate_percentiles(df, percentiles):
    import matplotlib.pyplot as plt
    import numpy as np
    df_percentiles = pd.DataFrame()

    # Always include the 50th percentile
    if 50 not in percentiles:
        percentiles.append(50)

    # Calculate percentiles and complementary percentiles
    data_percentiles = {percentile: np.percentile(df, percentile, axis=1) for percentile in percentiles}
    data_percentiles.update({100 - percentile: np.percentile(df, 100 - percentile, axis=1) for percentile in percentiles if percentile != 50})
    for percentile, data in data_percentiles.items():
        df_percentiles[percentile] = data
    df_percentiles.index = df.index
    return percentiles, df_percentiles

    # Plot filled areas and line

def create_df_from_percentiles_dict(data_percentiles, df_input):
    # Ensure the directory exists
    df = pd.DataFrame()
    for percentile, data in data_percentiles.items():
        # Downsample the data and add it to the DataFrame
        df[percentile] = data

    return df

def add_milp_results_to_df(df, milp_max, milp_fair_comp, variable, factor=1):
    if variable == 'action_sum_kwh':
        variable = 'action_sum'
    if variable not in milp_max.columns:
        print("Variable not in MILP results")
        if variable in ['alpha_d', 'cycle_num', 'execution_time']:
            return df
        else:
            return None
    if variable not in milp_fair_comp.columns:
        print("Variable not in MILP results")
        if variable in ['alpha_d', 'cycle_num', 'execution_time']:
            return df
        else:
            return None
    #use factor to change unit of MILP results:
    df['MILP_max'] = milp_max[variable] * factor
    df['MILP_fair_comp'] = milp_fair_comp[variable] * factor
    df = df.iloc[::i_th]
    return df

    # Save the DataFrame to a CSV file
    df.index = df_input.index
    # and rename the index with 'Time'
    df.index.name = 'Time'
    return df.sort_index(axis=1)
# Call the function
# visualise_percentiles(df=df_variable,data_percentiles=data_percentiles,percentiles=percentiles,colors=colors,sum_ticks=6, visualisation_path=visualisation_path, filename=filename ,i_th=64)

def get_df_variable_percentiles_all_runs(path, variable,percentiles, input_data, milp_max, milp_fair_comp, add_milp_results=False):
    results = get_results_fro_folder(path)
    df_variable = get_df_tracked_variable_all_runs(results, variable, input_data)
    percentiles, df_variable_percentiles = calculate_percentiles(df_variable, percentiles)
    milp_max.index = df_variable.index
    milp_fair_comp.index = df_variable.index

    if add_milp_results:
        if variable == 'soc':
            df_variable_percentiles = add_milp_results_to_df(df_variable_percentiles, milp_max, milp_fair_comp, variable, factor=1)
        elif variable in ['action_sum', 'action_da', 'action_id']:
            df_variable_percentiles = add_milp_results_to_df(df_variable_percentiles, milp_max, milp_fair_comp, variable, factor=1)
        elif variable in ['price_da', 'price_id']:
            df_variable_percentiles = add_milp_results_to_df(df_variable_percentiles, milp_max, milp_fair_comp, variable, factor=1)
        else:
            df_variable_percentiles = add_milp_results_to_df(df_variable_percentiles, milp_max, milp_fair_comp, variable)
    else:
        print("No MILP results added to the DataFrame")
    return df_variable_percentiles
    # Continue to adpapt the columns.

def get_keys_of_result_dict_via_path(path):
    results = get_results_fro_folder(path)
    return results[list(results.keys())[0]].keys()

def visualise_percentiles(df, data_percentiles, percentiles , colors, sum_ticks, visualisation_path, filename, i_th=4, save_fig_as_pdf=False):
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
    if save_fig_as_pdf:
        plt.savefig(os.path.join(visualisation_path, f'{filename}.pdf'))
    plt.show()

def save_df_to_csv(df,save_in_thousand, csv_directory, filename, variable):
    if variable == 'action_sum_kwh':
        variable = 'action_sum'
    if not os.path.exists(csv_directory):
        os.makedirs(csv_directory)
    for column in df.columns:
        if column == 'Time':
            continue
        if save_in_thousand:
            df[column] = df[column] / 1000
    if save_in_thousand:
        df.to_csv(os.path.join(csv_directory, f'{filename}_{variable}_in_thousand.csv'))
    else:
        df.to_csv(os.path.join(csv_directory, f'{filename}_{variable}.csv'))


# executing the functions

milp_max = adapt_milp_columns(milp_max,battery_power_mw=10, battery_capacity_mwh=battery_capacity_mwh)
milp_fair_comp = adapt_milp_columns(milp_fair_comp,battery_power_mw=10, battery_capacity_mwh=battery_capacity_mwh)
df_percentiles = get_df_variable_percentiles_all_runs(path, variable, percentiles, input_data, milp_max, milp_fair_comp, add_milp_results=add_milp_results)

if save_to_csv: 
    save_df_to_csv(df_percentiles, save_in_thousand, csv_directory, filename, variable)
