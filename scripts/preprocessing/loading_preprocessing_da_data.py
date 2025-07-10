#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# boolean to write the data to file
write_to_file = False
horizon = 24 # horizon of the agent. is important for spliting test and train data to fit to weekly data.

path = r'../../data/raw_data/iaew_marktdaten_preiotmarkt_dayahead_epex.csv'
data = pd.read_csv(path)

# reduce data to data with produkt = Stundenkontrakt
data_h = data[data["produkt"] == "Stundenkontrakt"]

# reduce data_h to marktgebiet = DE-LU-AT and marktgebiet = DE-LU 
data_h = data_h[(data_h["marktgebiet"] == "DE-LU-AT") | (data_h["marktgebiet"] == "DE-LU")]

# can i check if von is always followed by the next full hour ?
data_h = data_h.sort_values(['von', '#'])

# Calculate the 0.1 and 0.9 quantiles
q10, q90 = np.around(data_h["preis"].quantile([0.1, 0.9]), 2)

data_h["preis"] = data_h["preis"].apply(lambda x: q10 if x < q10 else q90 if x > q90 else x)

# Create a MinMaxScaler object
scaler = MinMaxScaler()

# normailzed data.
data_h['preis_normalized'] = scaler.fit_transform(data_h[['preis']]).round(2)

data_h.drop(columns = ['ID', 'produkt', 'marktgebiet', 'volumen'], inplace = True)

# leads to a split around 80 / 20, but fitting the weeks. 
num_rows = len(data_h)
test_split_02 = num_rows / 24 / 7 * 0.2
# round test_split_02 up to next integer 
test_split_02_round = int(np.ceil(test_split_02))

# plus one because the agent needs to step before receiving reward. 
test_size = test_split_02_round * 24 * 7 + horizon + 1
print('Calculated Size - train: ', num_rows - test_size,'test: ', test_size)
train_data = data_h[:-test_size]
test_data = data_h[-test_size:]
test_data_milp = test_data[:-horizon-1]
# sort the data by 'von'
train_data = train_data.sort_values(['von', '#'])
test_data = test_data.sort_values(['von', '#'])
test_data_milp = test_data_milp.sort_values(['von', '#'])

# and reset the index
train_data = train_data.reset_index(drop=True)
test_data = test_data.reset_index(drop=True)
test_data_milp = test_data_milp.reset_index(drop=True)

print('Executed Size - train: ', len(train_data),'test: ', len(test_data), 'test_milp: ', len(test_data_milp))
print("MILP size smaller, because the horizon is not considered to be needed for MILP evaluation.")

if write_to_file:
    train_data.to_csv(r'C:\Users\n.zoller\PycharmProjects\battery-optimisation-with-drl\data\processed_data_norm\DE_DA_train_adjusted_010_090.csv', index=False)
    test_data.to_csv(r'C:\Users\n.zoller\PycharmProjects\battery-optimisation-with-drl\data\processed_data_norm\DE_DA_test_adjusted_010_090.csv', index=False)
    test_data_milp.to_csv(r'C:\Users\n.zoller\PycharmProjects\battery-optimisation-with-drl\data\processed_data_norm\DE_DA_test_milp_adjusted_010_090.csv', index=False)
else:
    print('Data not written to file, boolean write_to_file is set to False')


# testing if its still the same.
train_data_2 = pd.read_csv(r'C:\Users\n.zoller\PycharmProjects\battery-optimisation-with-drl\data\processed_data_norm\DE_DA_train_adjusted_010_090.csv')
test_data_2 = pd.read_csv(r'C:\Users\n.zoller\PycharmProjects\battery-optimisation-with-drl\data\processed_data_norm\DE_DA_test_adjusted_010_090.csv')
test_data_milp_2 = pd.read_csv(r'C:\Users\n.zoller\PycharmProjects\battery-optimisation-with-drl\data\processed_data_norm\DE_DA_test_milp_adjusted_010_090.csv')

# compare the data
print("check if the data is created and stored correctly")
print("train equals :", train_data_2.equals(train_data),", test equals: ", test_data_2.equals(test_data),", Test MILP equals: ", test_data_milp_2.equals(test_data_milp))

