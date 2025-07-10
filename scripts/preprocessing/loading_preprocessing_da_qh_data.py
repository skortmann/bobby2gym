
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


#read excel from  I:\AEV\Studierende\MA_Zoller\09_Daten\Austausch_IAEW\iaew_marktdaten_preiotmarkt_dayahead_epex
resolution_per_hour = 4
horizon_hours = 24
write_to_file = False
write_to_file_pre_split = True
horizon = horizon_hours * resolution_per_hour

path = r'../../data/raw_data/iaew_marktdaten_preiotmarkt_dayahead_epex.csv'
data = pd.read_csv(path)

# reduce data to data with produkt = Stundenkontrakt
data_da = data[data["produkt"] == "Stundenkontrakt"]

# reduce data_h to marktgebiet = DE-LU-AT and marktgebiet = DE-LU 
data_da = data_da[(data_da["marktgebiet"] == "DE-LU-AT") | (data_da["marktgebiet"] == "DE-LU")]

duplicates = data_da[data_da.duplicated(subset='von', keep=False)]

indices_to_drop = []
for index, row in duplicates.iterrows():
    # get the index of the row with the lower volume
    index_to_drop = data_da[(data_da['von'] == row['von']) & (data_da['volumen'] < row['volumen'])].index
    indices_to_drop.append(index_to_drop)
indices_to_drop = [x[0] for x in indices_to_drop if x.size > 0]

print("Len of data_da Dataframe: ", len(data_da))
data_da = data_da.drop(indices_to_drop)
print("Len of data_da Dataframe after deleting duplicates: ", len(data_da), ", deleted rows: ", len(indices_to_drop))
print("------------------------------------")

data_da['von_dt'] = pd.to_datetime(data_da['von'])
data_da['bis_dt'] = pd.to_datetime(data_da['bis'])
data_da = data_da.sort_values('von_dt')
# Calculate the time difference between two consecutive rows
# Calculate the difference between 'bis_dt' of the current row and 'von_dt' of the next row
data_da['diff_bis_von_next'] = (data_da['von_dt'].shift(-1) - data_da['bis_dt']).dt.total_seconds() / 60
# Calculate the difference between 'von_dt' and 'bis_dt' of the same row
data_da['diff_von_bis'] = (data_da['bis_dt'] - data_da['von_dt']).dt.total_seconds() / 60
data_da.loc[data_da.index[-1], 'diff_bis_von_next'] = 0

print("Checking if if the df has missing time steps.")
# check if beside the last value of the data frame any "diff_bis_von_next" is nan 
if (data_da['diff_bis_von_next'] != 0).sum() >= 1: 
    print("There are some missing timeseries in the df, which will be filled.")
    print(data_da[data_da['diff_bis_von_next'] != 0][['von', 'bis', 'diff_bis_von_next']])
else: 
    print("Time Difference between rows calculated correctly.")
    # change data_id['diff_bis_von_next'] from last row to 0
print("------------------------------------")

# data_da where diff_bis_von_next is not 0 
new_rows = []
indices_to_drop_2 = []

print("Correcting both missing or too long ts")

for index, row in data_da.iterrows():
    # need to adapt the df from sql database. in 2018 - 2022 two sunrise shifts are handled as missung time steps, two are handled as 120 minutes periods. 
    if row['diff_bis_von_next'] != 0:
        # handlung missung ts 
        missed_ts = int(row['diff_bis_von_next']/60)
        print("missing ts in between rows:")
        print(data_da.loc[index: index + 1][['von', 'bis', 'diff_bis_von_next']])
        print("Number of new created rows: ", missed_ts, ", deleted row: 1.")
        print("------------------------------------")
        for i in range(0, missed_ts):
            new_row = row.copy()
            new_row['von_dt'] = new_row['bis_dt'] + pd.Timedelta(minutes=60*(i))
            new_row['bis_dt'] = new_row['von_dt'] + pd.Timedelta(minutes=60)
            new_rows.append(new_row)
    # This is handeling those sun time shifts, where the period is longer than 60 minutes.
    # I assume it always to be a mulitplicate of 60 minutes.
    
    if row['diff_von_bis'] != 60:
        # easiest way is after handeling doublets, to delete the original entry of the df.
        indices_to_drop_2.append(index)
        # Calculate the number of new rows needed
        num_new_rows = int(row['diff_von_bis'] / 60)
        print(data_da.loc[index: index + 1][['von', 'bis', 'diff_von_bis']])
        print("Number of new created rows: ", num_new_rows, ", deleted row: 1.")
        print("------------------------------------")

        # Create the new rows with the same volume and price
        for i in range(0, num_new_rows):
            new_row = row.copy()
            new_row['von_dt'] = new_row['von_dt'] + pd.Timedelta(minutes=60*(i))
            new_row['bis_dt'] = new_row['von_dt'] + pd.Timedelta(minutes=60*(1)) # i am just creating a 15 minutes difference from the start 
            new_row['von'] = new_row['von_dt'].strftime('%Y-%m-%d %H:%M:%S')
            new_row['bis'] = new_row['bis_dt'].strftime('%Y-%m-%d %H:%M:%S')
            new_rows.append(new_row)
# Convert the list of new rows into a DataFrame
data_da = data_da.drop(indices_to_drop_2)
    # Concatenate the new rows to the DataFrame
new_rows_df = pd.DataFrame(new_rows)
data_da = pd.concat([data_da, new_rows_df], ignore_index=True)

# Sort the DataFrame by 'von' 
data_da = data_da.sort_values('von_dt')
data_da = data_da.reset_index(drop=True)

# Recheck if procedure was sucessful 
data_da['diff_bis_von_next'] = (data_da['von_dt'].shift(-1) - data_da['bis_dt']).dt.total_seconds() / 60
# data_id['diff_bis_von_next'] = data_id['diff_bis_von_next'].fillna(0)
# Calculate the difference between 'von_dt' and 'bis_dt' of the same row
data_da['diff_von_bis'] = (data_da['bis_dt'] - data_da['von_dt']).dt.total_seconds() / 60
data_da.loc[data_da.index[-1], 'diff_bis_von_next'] = 0

data_da = data_da.sort_values('von')
data_da = data_da.reindex()
indizes_bigger_diff = data_da[data_da['diff_bis_von_next'] != 0].index
# then add to the list the next index too and later sort them again 
indizes_bigger_diff_2 = indizes_bigger_diff.append(indizes_bigger_diff + 1)
indizes_bigger_diff_2 = indizes_bigger_diff_2.sort_values()

print("------------------------------------")
print('Rechecking if missing time steps exists.')
if indizes_bigger_diff.empty == False:
    print('Check data preprocessing. Some values are missing! The last value is not checked, because it has no successor.')
    print (data_da.loc[indizes_bigger_diff_2, ['von', 'bis', 'diff_bis_von_next']])
else: 
    print('No Time difference between rows.')
print("------------------------------------")

def create_new_rows(row):
    # Create four new rows for each hour
    new_rows = []
    for i in range(4):
        new_row = row.copy()
        new_row['von_dt'] = row['von_dt'] + pd.Timedelta(minutes=15*i)
        new_row['bis_dt'] = new_row['von_dt'] + pd.Timedelta(minutes=15)
        new_row['von'] = new_row['von_dt'].strftime('%Y-%m-%d %H:%M:%S')
        new_row['bis'] = new_row['bis_dt'].strftime('%Y-%m-%d %H:%M:%S')
        new_row["volumen"] = np.around(new_row["volumen"] / 4, 2)
        new_rows.append(new_row)
    return new_rows

# Reducing one hour blocks to qh blocks
print("Reducing one hour blocks to 15 minute blocks.")
print("------------------------------------")

# Use apply to create new rows for all but the last row
new_rows = data_da[:-1].apply(create_new_rows, axis=1)

# Flatten the list of lists
new_rows = [item for sublist in new_rows for item in sublist]

# For the last row, create a new row with a 15-minute duration
last_row = data_da.iloc[-1].copy()
last_row['von_dt'] = last_row['von_dt']
last_row['bis_dt'] = last_row['von_dt'] + pd.Timedelta(minutes=15)
last_row['von'] = last_row['von_dt'].strftime('%Y-%m-%d %H:%M:%S')
last_row['bis'] = last_row['bis_dt'].strftime('%Y-%m-%d %H:%M:%S')
last_row["volumen"] = last_row["volumen"] / 4
new_rows.append(last_row)

# Convert the list of new rows into a DataFrame
data_da_qh = pd.DataFrame(new_rows)

# Sort the new DataFrame by 'von'
data_da_qh = data_da_qh.sort_values('von_dt')

# Reset the index of the new DataFrame
data_da_qh.reset_index(drop=True, inplace=True)

data_da_qh['diff_bis_von_next'] = (data_da_qh['von_dt'].shift(-1) - data_da_qh['bis_dt']).dt.total_seconds() / 60
# Calculate the difference between 'von_dt' and 'bis_dt' of the same row
data_da_qh['diff_von_bis'] = (data_da_qh['bis_dt'] - data_da_qh['von_dt']).dt.total_seconds() / 60
data_da_qh.loc[data_da_qh.index[-1], 'diff_bis_von_next'] = 0


indizes_bigger_diff = data_da_qh[data_da_qh['diff_bis_von_next'] != 0].index
# then add to the list the next index too and later sort them again
indizes_bigger_diff_2 = indizes_bigger_diff.append(indizes_bigger_diff + 1)
indizes_bigger_diff_2 = indizes_bigger_diff_2.sort_values()

#
if indizes_bigger_diff.empty == False:
    print('Check data preprocessing. Some values are missing! The last value is not checked, because it has no successor.')
    print (data_da_qh.loc[indizes_bigger_diff_2, ['von', 'bis', 'diff_bis_von_next']])
else:
    print('No Time difference between rows.')
print("------------------------------------")

# Calculate the 0.1 and 0.9 quantiles
q10, q90 = np.around(data_da["preis"].quantile([0.1, 0.9]), 2)
print("Quantiles 10 / 90: ", q10, q90)
print("------------------------------------")

data_da_qh["preis"] = data_da_qh["preis"].apply(lambda x: q10 if x < q10 else q90 if x > q90 else x)

# Create a MinMaxScaler object
scaler = MinMaxScaler()

# Fit the scaler to the 'preis' column and transform it
data_da_qh['preis_normalized'] = scaler.fit_transform(data_da_qh[['preis']])

# Round the 'preis_normalized' column to 4 decimal places
data_da_qh['preis_normalized'] = data_da_qh['preis_normalized'].round(2)

data_da_qh['da_market_trading'] = data_da_qh['von_dt'].dt.time == pd.to_datetime('12:00:00').time()

data_da_qh.drop(columns = ['produkt', 'marktgebiet', 'volumen', 'von_dt', 'bis_dt', 'diff_von_bis', 'diff_bis_von_next'], inplace = True)


# leads to a split around 80 / 20, but fitting the weeks. 
num_rows = len(data_da_qh)
test_split_02 = num_rows / 24 / 7 * 0.2
# round test_split_02 up to next integer 
test_split_02_round = int(np.ceil(test_split_02))
# plus one because the agent needs to step before receiving reward. 
test_size = test_split_02_round * 24 * 7 + horizon + 1
test_size

train_data = data_da_qh[:-test_size]
test_data = data_da_qh[-test_size:]
test_data_milp = test_data[:-horizon-1]

# sort the data by 'von'
train_data = train_data.sort_values('von')
test_data = test_data.sort_values('von')
test_data_milp = test_data_milp.sort_values('von')
# and reset the index
train_data = train_data.reset_index(drop=True)
test_data = test_data.reset_index(drop=True)
test_data_milp = test_data_milp.reset_index(drop=True)

full_data_path = r'../../data/processed_data/DE_DA_QH_fulldf_adjusted_010_090.csv'
if write_to_file_pre_split:
    data_da_qh.to_csv(full_data_path, index=False)
else:
    print("to write full dataframe for combining with intraday data, set boolean 'write_to_file_pre_split' to TRUE")


print("Check if the full df data is created and stored correctly")
data_da_qh_2 = pd.read_csv(full_data_path)
print("data equals :", data_da_qh_2.equals(data_da_qh))
print("------------------------------------")

if write_to_file:
    train_data.to_csv(r'C:\Users\n.zoller\PycharmProjects\battery-optimisation-with-drl\data\processed_data_norm\DE_DA_QH_train_adjusted_010_090.csv', index=False)
    test_data.to_csv(r'C:\Users\n.zoller\PycharmProjects\battery-optimisation-with-drl\data\processed_data_norm\DE_DA_QH_test_adjusted_010_090.csv', index=False)
    test_data_milp.to_csv(r'C:\Users\n.zoller\PycharmProjects\battery-optimisation-with-drl\data\processed_data_norm\DE_DA_QH_test_milp_adjusted_010_090.csv', index=False)
else: 
    print('train, test and MILP Files files not written to storage, boolean write_to_file is set to False')

# check if the data is stored correctly
train_data_2 = pd.read_csv('../../data/processed_data/DE_DA_QH_train_adjusted_010_090.csv')
test_data_2 = pd.read_csv('../../data/processed_data/DE_DA_QH_test_adjusted_010_090.csv')
test_data_milp_2 = pd.read_csv('../../data/processed_data/DE_DA_QH_test_milp_adjusted_010_090.csv')

# compare the data
print("check if the train, test and MILP files are created and stored correctly")
print("train equals :", train_data_2.equals(train_data),", test equals: ", test_data_2.equals(test_data),", Test MILP equals: ", test_data_milp_2.equals(test_data_milp))

