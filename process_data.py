import numpy as np
import pandas as pd
from scipy.stats import zscore


# process bitcoin data into sorted dates
def process_data(sequence_length=30):
    
    # read in the data file
    file_name_df = pd.read_csv("./data/coin_Bitcoin.csv")
    # sort by date
    file_name_df['Date'] = pd.to_datetime(file_name_df['Date'])
    # sort by date(ascending), set index back to original, and drop empty columns
    file_name_df = file_name_df.sort_values(by='Date', ascending=True).reset_index(drop=True)

    # standardize with z-score standardization

    # standardize high, low, open, close, marketvolume, and market cap
    file_name_df['High'] = zscore(file_name_df['High'])
    file_name_df['Low'] = zscore(file_name_df['Low'])
    file_name_df['Open'] = zscore(file_name_df['Open'])
    file_name_df['Close'] = zscore(file_name_df['Close'])
    file_name_df['Volume'] = zscore(file_name_df['Volume'])
    file_name_df['Marketcap'] = zscore(file_name_df['Marketcap'])
    features = ['High', 'Low', 'Open', 'Close', 'Volume', 'Marketcap']
    data = file_name_df[features].values
    #print("After normalization: ")
    #print(file_name_df.head())
    print("Before sequences: ", file_name_df.head())

    # Create sequences
    features = ['High', 'Low', 'Open', 'Close', 'Volume', 'Marketcap']
    x, y = [], []
    for i in range(len(data) - sequence_length):
        x.append(data[i:i+sequence_length])  # Previous `sequence_length` days
        y.append(data[i+sequence_length][3])  # Predict 'Close' price
    print("After sequences: ", file_name_df.head())
    return np.array(x), np.array(y)
