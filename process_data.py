import pandas as pd
from scipy.stats import zscore
import numpy as np
import os

# process bitcoin data into sorted dates
def process_data(sequence_length=30):
    
    print("Current Working Directory:", os.getcwd())
    # read in the data file
    file_name_df = pd.read_csv(f"./CPT_S-315-Crypto-Project/data/coin_Bitcoin.csv")
    # sort by date
    file_name_df['Date'] = pd.to_datetime(file_name_df['Date'])
    # sort by date(ascending), set index back to original, and drop empty columns
    file_name_df = file_name_df.sort_values(by='Date', ascending=True).reset_index(drop=True)
    #print("before normalization:")
    #print(file_name_df.head())

    # standardize with z-score standardization

    # standardize high, low, open, close, marketvolume, and market cap
    file_name_df['High'] = zscore(file_name_df['High'])
    file_name_df['Low'] = zscore(file_name_df['Low'])
    file_name_df['Open'] = zscore(file_name_df['Open'])
    file_name_df['Close'] = zscore(file_name_df['Close'])
    file_name_df['Volume'] = zscore(file_name_df['Volume'])
    file_name_df['Marketcap'] = zscore(file_name_df['Marketcap'])
    #print("after normalization:")
    #print(file_name_df.head())
    features = ['High', 'Low', 'Open', 'Close', 'Volume', 'Marketcap']
    data = file_name_df[features].values

    # Create sequences
    features = ['High', 'Low', 'Open', 'Close', 'Volume', 'Marketcap']
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])  # Previous `sequence_length` days
        y.append(data[i+sequence_length][3])  # Predict 'Close' price
    print(str(np.array(X)) + str(np.array(y)))
    return np.array(X), np.array(y)