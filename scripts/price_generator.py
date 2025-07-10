import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os



class price_generator():
    def __init__(self, train_path, test_path):
        try:
            train_prices_df = pd.read_csv(train_path)
            test_prices_df = pd.read_csv(test_path)
        except:
            print('Price Generator: Could not read the file')
            print("train_path: ", train_path)
            print("test_path: ", test_path)
            current_dic = os.getcwd()
            print("current_dic: ", current_dic)
            sys.exit(1)

        train_prices = train_prices_df.loc[:, 'preis'].to_numpy(dtype=np.float32)
        test_prices = test_prices_df.loc[:, 'preis'].to_numpy(dtype=np.float32)
        train_prices_norm = train_prices_df.loc[:, 'preis_normalized'].to_numpy(dtype=np.float32)
        test_prices_norm = test_prices_df.loc[:, 'preis_normalized'].to_numpy(dtype=np.float32)
        self.prices_true = {'train': train_prices, 'test': test_prices}
        self.prices_norm = {'train': train_prices_norm, 'test': test_prices_norm}

    def get_timeseries(self, length, train_or_test='train'):
        price_type_true = self.prices_true[train_or_test]
        price_type_norm = self.prices_norm[train_or_test]
        start_index = np.random.randint(0, len(price_type_true) - length)
        return price_type_norm[start_index:(start_index + length)], price_type_true[start_index:(start_index + length)]

class price_generator_id_da():
    def __init__(self, train_path, test_path):
        try:
            train_prices_df = pd.read_csv(train_path)
            test_prices_df = pd.read_csv(test_path)
        except:
            print('Price generator: Could not read the file')
            print("train_path: ", train_path)
            print("test_path: ", test_path)
            sys.exit(1)

        train_prices_id_norm = train_prices_df.loc[:, 'price_id_norm'].to_numpy(dtype=np.float32)
        train_prices_da_norm = train_prices_df.loc[:, 'price_da_qh_norm'].to_numpy(dtype=np.float32)
        train_prices_id_true = train_prices_df.loc[:, 'price_id'].to_numpy(dtype=np.float32)
        train_prices_da_true = train_prices_df.loc[:, 'price_da_qh'].to_numpy(dtype=np.float32)
        train_da_market_tracker = train_prices_df.loc[:, 'da_market_trading'].to_numpy(dtype=np.float32)

        test_prices_id_norm = test_prices_df.loc[:, 'price_id_norm'].to_numpy(dtype=np.float32)
        test_prices_da_norm = test_prices_df.loc[:, 'price_da_qh_norm'].to_numpy(dtype=np.float32)
        test_prices_id_true = test_prices_df.loc[:, 'price_id'].to_numpy(dtype=np.float32)
        test_prices_da_true = test_prices_df.loc[:, 'price_da_qh'].to_numpy(dtype=np.float32)
        test_da_market_tracker = test_prices_df.loc[:, 'da_market_trading'].to_numpy(dtype=np.float32)

        self.prices_id_true = {'train': train_prices_id_true, 'test': test_prices_id_true}
        self.prices_id_norm = {'train': train_prices_id_norm, 'test': test_prices_id_norm}
        self.prices_da_true = {'train': train_prices_da_true, 'test': test_prices_da_true}
        self.prices_da_norm = {'train': train_prices_da_norm, 'test': test_prices_da_norm}
        self.da_market_tracker = {'train': train_da_market_tracker, 'test': test_da_market_tracker}

    def get_timeseries(self, length, horizon, train_or_test= 'train'):
        price_type_id_norm = self.prices_id_norm[train_or_test]
        price_type_id_true = self.prices_id_true[train_or_test]
        price_type_da_norm = self.prices_da_norm[train_or_test]
        price_type_da_true = self.prices_da_true[train_or_test]
        da_market_tracker = self.da_market_tracker[train_or_test]
        # start_index
        midnight = (len(price_type_id_true) - length - horizon)
        if midnight >= 0:
            midnight_starter = midnight // (24 * 4)
            if midnight_starter == 0:
                start_index_day_index = 0
            else:
                start_index_day_index = np.random.randint(0, midnight_starter + 1)
            start_index = start_index_day_index * (24 * 4)
        else:
            print('The length of the time series is too long')
            sys.exit(1)

        return (price_type_id_norm[start_index:(start_index + length + horizon)],
                price_type_id_true[start_index:(start_index + length + horizon)],
                price_type_da_norm[start_index:(start_index + length + horizon)],
                price_type_da_true[start_index:(start_index + length + horizon)],
                da_market_tracker[start_index:(start_index + length + horizon)])
