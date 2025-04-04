import numpy as np
import pandas as pd
from scipy.stats import zscore


# process bitcoin data into sorted dates with normalized values
def process_data():
    
    # read in the data file
    file_name_df = pd.read_csv("./data/coin_Bitcoin.csv")
    # sort by date
    file_name_df['Date'] = pd.to_datetime(file_name_df['Date'])
    # sort by date(ascending), set index back to original, and drop empty columns
    file_name_df = file_name_df.sort_values(by='Date', ascending=True).reset_index(drop=True)

    # save the original close stats before we normalize
    original_close_mean = file_name_df['Close'].mean()
    original_close_std = file_name_df['Close'].std()

    # standardize with z-score standardization

    # standardize high, low, open, close, marketvolume, and market cap
    file_name_df['High'] = zscore(file_name_df['High'])
    file_name_df['Low'] = zscore(file_name_df['Low'])
    file_name_df['Open'] = zscore(file_name_df['Open'])
    file_name_df['Close'] = zscore(file_name_df['Close'])
    file_name_df['Volume'] = zscore(file_name_df['Volume'])
    file_name_df['Marketcap'] = zscore(file_name_df['Marketcap'])
    #print("After normalization: ")
    #print(file_name_df.head())

    # convert it into a numpy array before returning
    np_processed = file_name_df.to_numpy()

    return { "numpy": np_processed, "df": file_name_df, "close_mean": original_close_mean, "close_std": original_close_std}
