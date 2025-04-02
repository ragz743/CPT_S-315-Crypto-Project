import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Read data from Bitcoin and Ethereum CSVs and put them into dataframes
BitcoinCSV_df = pd.read_csv("coin_Bitcoin.csv")
EthereumCSV_df = pd.read_csv("coin_Ethereum.csv")

# Convert Date from a string to Datetime
BitcoinCSV_df['Date'] = pd.to_datetime(BitcoinCSV_df['Date'])
EthereumCSV_df['Date'] = pd.to_datetime(EthereumCSV_df['Date'])

# Remove duplicate Dates from both dataframes
BitcoinCSV_df = BitcoinCSV_df.drop_duplicates(subset=['Date'], keep='first')
EthereumCSV_df = EthereumCSV_df.drop_duplicates(subset=['Date'], keep='first')

# Sort Dates in order
BitcoinCSV_df.sort_values('Date')
EthereumCSV_df.sort_values('Date')

# Check if data is missing in any of the columns in both dataframes and replace them using backward filling
emptyVals = ['High','Low','Open','Close','Volume','Marketcap']
for col in emptyVals:
    BitcoinCSV_df[col] = BitcoinCSV_df[col].bfill()
    EthereumCSV_df[col] = EthereumCSV_df[col].bfill()

# ---------- RANDOM FOREST USING CLOSE ----------

# For Close_bit_tag1, shift the Close column down by 1 row
# For Close_bit_tag2, shift the Close column down by 2 rows
BitcoinCSV_df['Close_bit_tag1'] = BitcoinCSV_df['Close'].shift(1)
BitcoinCSV_df['Close_bit_tag2'] = BitcoinCSV_df['Close'].shift(2)

EthereumCSV_df['Close_eth_tag1'] = EthereumCSV_df['Close'].shift(1)
EthereumCSV_df['Close_eth_tag2'] = EthereumCSV_df['Close'].shift(2)

# Creating training data, with x_train using Close_bit_tag1 and Close_bit_tag2, and y_train using 'Close'
x_train1, x_test1, y_train1, y_test1 = train_test_split(BitcoinCSV_df[['Close_bit_tag1', 'Close_bit_tag2']], BitcoinCSV_df['Close'], test_size=0.2, random_state=42)
x_train2, x_test2, y_train2, y_test2 = train_test_split(EthereumCSV_df[['Close_eth_tag1', 'Close_eth_tag2']], EthereumCSV_df['Close'], test_size=0.2, random_state=42)

# Create a random forest using 100 trees and 42 random selection, as well as fit x_train and y_train
TrainingTree1 = RandomForestRegressor(n_estimators=100, random_state=42)
TrainingTree1.fit(x_train1, y_train1)

TrainingTree2 = RandomForestRegressor(n_estimators=100, random_state=42)
TrainingTree2.fit(x_train2, y_train2)

# Create a prediction model by predicting x_train on the training tree
PredictionModel1 = TrainingTree1.predict(x_test1)
r2_1 = r2_score(y_test1, PredictionModel1)
print("Accuracy of Bitcoin Close Prediction Model using R^2 Score: " + str(r2_1))

PredictionModel2 = TrainingTree2.predict(x_test2)
r2_2 = r2_score(y_test2, PredictionModel2)
print("Accuracy of Ethereum Close Prediction Model using R^2 Score: " + str(r2_2))

# ---------- RANDOM FOREST USING HIGH ----------
