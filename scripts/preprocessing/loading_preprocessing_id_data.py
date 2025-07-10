#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

resolution_per_hour = 4
horizon_hours = 24
write_to_file = False
write_to_file_pre_split = False
horizon = horizon_hours * resolution_per_hour

path = r'../../data/raw_data/iaew_marktdaten_preiintraday_auktion_epex.csv'
data = pd.read_csv(path)

# can i check if von is always followed by the next full hour ? 
data_id = data.sort_values(['#', 'von'])

# Identify the duplicate rows based on the 'von' column
duplicates = data_id[data_id.duplicated(subset='von', keep=False)]

indices_to_drop = []
for index, row in duplicates.iterrows():
    # get the index of the row with the lower volume
    index_to_drop = data_id[(data_id['von'] == row['von']) & (data_id['volumen'] < row['volumen'])].index
    indices_to_drop.append(index_to_drop)
indices_to_drop = [x[0] for x in indices_to_drop if x.size > 0]

# drop the duplicates
data_id = data_id.drop(indices_to_drop)

data_id['von_dt'] = pd.to_datetime(data_id['von'])
data_id['bis_dt'] = pd.to_datetime(data_id['bis'])
data_id = data_id.sort_values('von')
# Calculate the time difference between two consecutive rows
# Calculate the difference between 'bis_dt' of the current row and 'von_dt' of the next row
data_id['diff_bis_von_next'] = (data_id['von_dt'].shift(-1) - data_id['bis_dt']).dt.total_seconds() / 60
# Calculate the difference between 'von_dt' and 'bis_dt' of the same row
data_id['diff_von_bis'] = (data_id['bis_dt'] - data_id['von_dt']).dt.total_seconds() / 60
data_id.loc[data_id.index[-1], 'diff_bis_von_next'] = 0

indizes_bigger_diff = data_id[data_id['diff_bis_von_next'] != 0].index
# then add to the list the next index too and later sort them again
indizes_bigger_diff_2 = indizes_bigger_diff.append(indizes_bigger_diff + 1)
indizes_bigger_diff_2 = indizes_bigger_diff_2.sort_values()
indizes_bigger_diff_2

#
if indizes_bigger_diff.empty == False:
    print('Check data preprocessing. Some gap inbetween data! The last value is not checked, because it has no successor.')
    print (data_id.loc[indizes_bigger_diff_2, ['von', 'bis', 'diff_bis_von_next']])
else:
    print('No Time difference between rows.')

print("------------------------------------")

# Create an empty list to store the new rows
new_rows = []
indices_to_drop_2 = []

# Checking if any row is longer than 15 minutes
if data_id[data_id['diff_von_bis'] != 0] is not None:
    print("Check data preprocessing. Some values are longer than 15 minutes!")
else:
    print("Each row is 15 minutes long.")

# Iterate over the rows of the DataFrame
for index, row in data_id.iterrows():
    # Check if 'diff_von_bis' is not equal to 15
    if row['diff_von_bis'] != 15:
        indices_to_drop_2.append(index)
        # Calculate the number of new rows needed
        num_new_rows = int(row['diff_von_bis'] / 15)

        # Create the new rows with the same volume and price
        for i in range(0, num_new_rows):
            new_row = row.copy()
            new_row['von_dt'] = new_row['von_dt'] + pd.Timedelta(minutes=15*(i))
            new_row['bis_dt'] = new_row['von_dt'] + pd.Timedelta(minutes=15*(1)) # i am just creating a 15 minutes difference from the start
            new_row['von'] = new_row['von_dt'].strftime('%Y-%m-%d %H:%M:%S')
            new_row['bis'] = new_row['bis_dt'].strftime('%Y-%m-%d %H:%M:%S')
            new_rows.append(new_row)
dates = data_id.loc[indices_to_drop_2, ['von', 'bis']]
# Convert the list of new rows into a DataFrame
data_id = data_id.drop(indices_to_drop_2)
    # Concatenate the new rows to the DataFrame
new_rows_df = pd.DataFrame(new_rows)

data_id = pd.concat([data_id, new_rows_df], ignore_index=True)

# Sort the DataFrame by 'von'
data_id = data_id.sort_values('von')


data_id['diff_von_bis'] = (data_id['bis_dt'] - data_id['von_dt']).dt.total_seconds() / 60
data_id['diff_bis_von_next'] = (data_id['von_dt'].shift(-1) - data_id['bis_dt']).dt.total_seconds() / 60
data_id.loc[data_id.index[-1], 'diff_bis_von_next'] = 0

# show rows where diff_von_bis is not 15 and the consecutive 4 rows
indeces_diff_von_bis = data_id[data_id['diff_von_bis'] != 15].index
# indeces_diff_von_bis should be empty now. if its not empty, the if not
if not indeces_diff_von_bis.empty:
    print("The Epsiodes being longer than 15 minutes are not corrected properly.")
else:
    print("Filling rows of day time shift sucessfull")

print("------------------------------------")

indizes_bigger_diff = data_id[data_id['diff_bis_von_next'] != 0].index

if indizes_bigger_diff.empty == False:
    print('Check data preprocessing. Some gap inbetween data! The last value is not checked, because it has no successor.')
    print (data_id.loc[indizes_bigger_diff_2, ['von', 'bis', 'diff_bis_von_next']])
else:
    print('No Time difference between rows.')

print("------------------------------------")
# check if beside the last value of the data frame any "diff_bis_von_next" is nan




## redo after correction of data.
data_id['diff_bis_von_next'] = (data_id['von_dt'].shift(-1) - data_id['bis_dt']).dt.total_seconds() / 60
# Calculate the difference between 'von_dt' and 'bis_dt' of the same row
data_id['diff_von_bis'] = (data_id['bis_dt'] - data_id['von_dt']).dt.total_seconds() / 60
data_id.loc[data_id.index[-1], 'diff_bis_von_next'] = 0


# Recheck if data is been processed correctly
print("------------------------------------")
print("Recheck if data is been processed correctly")
print("------------------------------------")

# check if beside the last value of the data frame any "diff_bis_von_next" is nan

indizes_bigger_diff = data_id[data_id['diff_bis_von_next'] != 0].index
# then add to the list the next index too and later sort them again
indizes_bigger_diff_2 = indizes_bigger_diff.append(indizes_bigger_diff + 1)
indizes_bigger_diff_2 = indizes_bigger_diff_2.sort_values()

#
if indizes_bigger_diff.empty == False:
    print('Check data preprocessing. Some values are missing! The last value is not checked, because it has no successor.')
    print (data_id.loc[indizes_bigger_diff_2, ['von', 'bis', 'diff_bis_von_next']])
else:
    print('No Time difference between rows.')
print("------------------------------------")

# Calculate the 0.1 and 0.9 quantiles
q10, q90 = np.around(data_id["preis"].quantile([0.1, 0.9]), 2)
print('quantile 10: ',  q10,'quantile 90: ', q90)

data_id["preis"] = data_id["preis"].apply(lambda x: q10 if x < q10 else q90 if x > q90 else x)

# Create a MinMaxScaler object
scaler = MinMaxScaler()

# Fit the scaler to the 'preis' column and transform it
data_id['preis_normalized'] = scaler.fit_transform(data_id[['preis']])

# Round the 'preis_normalized' column to 4 decimal places
data_id['preis_normalized'] = data_id['preis_normalized'].round(2)

# add a column called da_market_trading, which is a boolean True if time is 12:00:00 and False otherwise
data_id['da_market_trading'] = data_id['von_dt'].dt.time == pd.to_datetime('12:00:00').time()

# leads to a split around 80 / 20, but fitting the weeks.
num_rows = len(data_id)
test_split_02 = num_rows / 24 / 7 * 0.2
# round test_split_02 up to next integer 
test_split_02_round = int(np.ceil(test_split_02))
# plus one because the agent needs to step before receiving reward. 
test_size = test_split_02_round * 24 * 7 + horizon + 1



data_id.sort_values('von_dt')
data_id = data_id.reset_index(drop=True)
data_id.drop(columns=['marktgebiet', 'volumen', 'diff_bis_von_next', 'diff_von_bis', 'von_dt', 'bis_dt'], inplace=True)



train_data = data_id[:-test_size]
test_data = data_id[-test_size:]
test_data_milp = test_data[:-horizon-1]

# sort the data by 'von'
train_data = train_data.sort_values('#')
test_data = test_data.sort_values('#')
test_data_milp = test_data_milp.sort_values('#')
# and reset the index
train_data = train_data.reset_index(drop=True)
test_data = test_data.reset_index(drop=True)
test_data_milp = test_data_milp.reset_index(drop=True)

path_full_data_id = r'../../data/processed_data/DE_ID_fulldf_adjusted_010_090.csv'
if write_to_file_pre_split:
    data_id.to_csv(path_full_data_id, index=False)
else:
    print("to write full dataframe for combining with intraday data, set boolean 'write_to_file_pre_split' to TRUE")

data_id_2 = pd.read_csv(path_full_data_id)

print("------------------------------------")
print("check if the data is created and stored correctly")
print("data equals :", data_id_2.equals(data_id))
print("------------------------------------")


if write_to_file:
    train_data.to_csv('../../data/processed_data/DE_ID_train_adjusted_010_090.csv', index=False)
    test_data.to_csv('../../data/processed_data/DE_ID_test_adjusted_010_090.csv', index=False)
    test_data_milp.to_csv('../../data/processed_data/DE_ID_test_milp_adjusted_010_090.csv', index=False)
else: 
    print('Data not written to file, boolean write_to_file is set to False')

# check if the data is stored correctly
train_data_2 = pd.read_csv('../../data/processed_data/DE_ID_train_adjusted_010_090.csv')
test_data_2 = pd.read_csv('../../data/processed_data/DE_ID_test_adjusted_010_090.csv')
test_data_milp_2 = pd.read_csv('../../data/processed_data/DE_ID_test_milp_adjusted_010_090.csv')

print("------------------------------------")
# compare the data
print("check if the data is created and stored correctly")
print("train equals :", train_data_2.equals(train_data),", test equals: ", test_data_2.equals(test_data),", Test MILP equals: ", test_data_milp_2.equals(test_data_milp))
