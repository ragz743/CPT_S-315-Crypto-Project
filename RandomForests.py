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

# For High_bit_tag1, shift the High column down by 1 row
# For High_bit_tag2, shift the High column down by 2 rows
BitcoinCSV_df['High_bit_tag1'] = BitcoinCSV_df['High'].shift(1)
BitcoinCSV_df['High_bit_tag2'] = BitcoinCSV_df['High'].shift(2)

EthereumCSV_df['High_eth_tag1'] = EthereumCSV_df['High'].shift(1)
EthereumCSV_df['High_eth_tag2'] = EthereumCSV_df['High'].shift(2)

# Creating training data, with x_train using Close_bit_tag1 and Close_bit_tag2, and y_train using 'Close'
x_train3, x_test3, y_train3, y_test3 = train_test_split(BitcoinCSV_df[['High_bit_tag1', 'High_bit_tag2']], BitcoinCSV_df['High'], test_size=0.2, random_state=42)
x_train4, x_test4, y_train4, y_test4 = train_test_split(EthereumCSV_df[['High_eth_tag1', 'High_eth_tag2']], EthereumCSV_df['High'], test_size=0.2, random_state=42)

# Create a random forest using 100 trees and 42 random selection, as well as fit x_train and y_train
TrainingTree3 = RandomForestRegressor(n_estimators=100, random_state=42)
TrainingTree3.fit(x_train3, y_train3)

TrainingTree4 = RandomForestRegressor(n_estimators=100, random_state=42)
TrainingTree4.fit(x_train4, y_train4)

# Create a prediction model by predicting x_train on the training tree
PredictionModel3 = TrainingTree3.predict(x_test3)
r2_3 = r2_score(y_test3, PredictionModel3)
print("Accuracy of Bitcoin High Prediction Model using R^2 Score: " + str(r2_3))

PredictionModel4 = TrainingTree4.predict(x_test4)
r2_4 = r2_score(y_test4, PredictionModel4)
print("Accuracy of Ethereum High Prediction Model using R^2 Score: " + str(r2_4))

# ---------- RANDOM FOREST USING LOW ----------

# For Low_bit_tag1, shift the Low column down by 1 row
# For Low_bit_tag2, shift the Low column down by 2 rows
BitcoinCSV_df['Low_bit_tag1'] = BitcoinCSV_df['Low'].shift(1)
BitcoinCSV_df['Low_bit_tag2'] = BitcoinCSV_df['Low'].shift(2)

EthereumCSV_df['Low_eth_tag1'] = EthereumCSV_df['Low'].shift(1)
EthereumCSV_df['Low_eth_tag2'] = EthereumCSV_df['Low'].shift(2)

# Creating training data, with x_train using Close_bit_tag1 and Close_bit_tag2, and y_train using 'Close'
x_train5, x_test5, y_train5, y_test5 = train_test_split(BitcoinCSV_df[['Low_bit_tag1', 'Low_bit_tag2']], BitcoinCSV_df['Low'], test_size=0.2, random_state=42)
x_train6, x_test6, y_train6, y_test6 = train_test_split(EthereumCSV_df[['Low_eth_tag1', 'Low_eth_tag2']], EthereumCSV_df['Low'], test_size=0.2, random_state=42)

# Create a random forest using 100 trees and 42 random selection, as well as fit x_train and y_train
TrainingTree5 = RandomForestRegressor(n_estimators=100, random_state=42)
TrainingTree5.fit(x_train5, y_train5)

TrainingTree6 = RandomForestRegressor(n_estimators=100, random_state=42)
TrainingTree6.fit(x_train6, y_train6)

# Create a prediction model by predicting x_train on the training tree
PredictionModel5 = TrainingTree5.predict(x_test5)
r2_5 = r2_score(y_test5, PredictionModel5)
print("Accuracy of Bitcoin Low Prediction Model using R^2 Score: " + str(r2_5))

PredictionModel6 = TrainingTree6.predict(x_test6)
r2_6 = r2_score(y_test6, PredictionModel6)
print("Accuracy of Ethereum Low Prediction Model using R^2 Score: " + str(r2_6))

# ---------- RANDOM FOREST USING VOLUME ----------


# ---------- RANDOM FOREST USING MARKETCAP ----------