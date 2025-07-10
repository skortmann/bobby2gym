import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

resolution_per_hour = 4
horizon_hours = 36
write_to_file = False
horizon = horizon_hours * resolution_per_hour

data_path_df_DA_QH = r'../../data/processed_data/DE_DA_QH_fulldf_adjusted_010_090.csv'
path_full_data_id = r'../../data/processed_data/DE_ID_fulldf_adjusted_010_090.csv'

df_DA_QH = pd.read_csv(data_path_df_DA_QH)
df_ID = pd.read_csv(path_full_data_id)

von_check = df_DA_QH['von'].equals(df_ID['von'])
bis_check = df_DA_QH['bis'].equals(df_ID['bis'])

print("Von columns are the same: ", von_check)
print("Bis columns are the same: ", bis_check)

# Rename columns for df_DA_QH
df_DA_QH = df_DA_QH.rename(columns={'preis': 'price_da_qh', 'preis_normalized': 'price_da_qh_norm',"ID": "ID_da_qh"})

# Rename columns for df_ID
df_ID = df_ID.rename(columns={'preis': 'price_id', 'preis_normalized': 'price_id_norm', "ID" : "ID_id"})

# Merge the two dataframes
df_merged = pd.merge(df_DA_QH, df_ID, on=['von', 'bis', 'da_market_trading'], how='inner')
df_merged = df_merged.drop(columns=['#_x', '#_y'])
# reorder columns
df_merged = df_merged[['von', 'bis', 'da_market_trading', 'price_da_qh', 'price_da_qh_norm', 'ID_da_qh', 'price_id', 'price_id_norm', 'ID_id']]

# ## Splitting the df in test, train and milp data once again.
# leads to a split around 80 / 20, but fitting the weeks.
hours_per_day = 24
days_per_week = 7
qh_per_hour = 4
ts_taken_into_account = hours_per_day * days_per_week * qh_per_hour

num_rows = len(df_merged)
test_split_02 = ( num_rows / ts_taken_into_account + horizon / ts_taken_into_account) * 0.2
# round test_split_02 up to next integer 
test_split_02_round = int(np.ceil(test_split_02))
# plus one because the agent needs to step before receiving reward. 
test_size = test_split_02_round * ts_taken_into_account + 1

quotient, remainder = divmod(horizon, hours_per_day * qh_per_hour)
days_to_ignore = quotient + (1 if remainder else 0)
ts_to_ignore = days_to_ignore * hours_per_day * qh_per_hour

train_data = df_merged[:-test_size]
test_data = df_merged[-test_size:]
test_data_milp = test_data[:-ts_to_ignore-1]

# and reset the index
train_data = train_data.reset_index(drop=True)
test_data = test_data.reset_index(drop=True)
test_data_milp = test_data_milp.reset_index(drop=True)

if write_to_file:
    train_data.to_csv('../../data/processed_data/DE_DA_QH_ID_train_adjusted_010_090.csv', index=False)
    test_data.to_csv('../../data/processed_data/DE_DA_QH_ID_test_adjusted_010_090.csv', index=False)
    test_data_milp.to_csv('../../data/processed_data/DE_DA_QH_ID_test_milp_adjusted_010_090.csv', index=False)
else: 
    print('Data not written to file, boolean write_to_file is set to False')

# check if the data is stored correctly
train_data_2 = pd.read_csv('../../data/processed_data/DE_DA_QH_ID_train_adjusted_010_090.csv')
test_data_2 = pd.read_csv('../../data/processed_data/DE_DA_QH_ID_test_adjusted_010_090.csv')
test_data_milp_2 = pd.read_csv('../../data/processed_data/DE_DA_QH_ID_test_milp_adjusted_010_090.csv')

print("------------------------------------")

print("check if the data is created and stored correctly")
print("train equals :", train_data_2.equals(train_data),", test equals: ", test_data_2.equals(test_data),", Test MILP equals: ", test_data_milp_2.equals(test_data_milp))


