import numpy as np


# process bitcoin data into sorted dates
def sequence_data(sequence_length, processed_df):
    features = ['High', 'Low', 'Open', 'Close', 'Volume', 'Marketcap']
    data = processed_df[features].values
    
    # Create sequences
    features = ['High', 'Low', 'Open', 'Close', 'Volume', 'Marketcap']
    x, y = [], []
    for i in range(len(data) - sequence_length):
        x.append(data[i:i+sequence_length])  # Previous `sequence_length` days
        y.append(data[i+sequence_length][3])  # Predict 'Close' price

    return np.array(x), np.array(y)
