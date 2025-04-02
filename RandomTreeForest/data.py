import pandas as pd
import numpy as np
import  matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error


bit_Coin_File = pd.read_csv("../Data/coin_Bitcoin.csv")
Ethereum_File = pd.read_csv("../Data/coin_Ethereum.csv")

BitCoin = bit_Coin_File
Ethereum = Ethereum_File
#
#
BitCoin['Date'] = pd.to_datetime(BitCoin['Date'])
Ethereum['Date'] = pd.to_datetime(Ethereum['Date'])

##Clean Here In the Future if dublicates by datte



BitCoin.sort_values('Date')
Ethereum.sort_values('Date')

#Check if any data is missing all catagorys




BitCoin['Close_bit_tag1'] = BitCoin['Close'].shift(1)
BitCoin['Close_bit_tag2'] = BitCoin['Close'].shift(2)

Ethereum['Close_eth_tag1'] = Ethereum['Close'].shift(1)
Ethereum['Close_eth_tag2'] = Ethereum['Close'].shift(2)


x_train = BitCoin[['Close_bit_tag1', 'Close_bit_tag2']]
y_train = BitCoin['Close']



TreeTrain = RandomForestRegressor(n_estimators=100,random_state=40)

TreeTrain.fit(x_train, y_train)

Predicted = TreeTrain.predict(x_train)
print(Predicted)