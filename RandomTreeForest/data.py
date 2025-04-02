import pandas as pd
import numpy as np
import  matplotlib.pyplot as plt
from libpasteurize.fixes.feature_base import Features
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

bit_Coin_File = pd.read_csv("../Data/coin_Bitcoin.csv")
Ethereum_File = pd.read_csv("../Data/coin_Ethereum.csv")



def makeGraph(fileInput,TestSize,y_test,Predictions,):
    plt.figure(figsize=(10, 6))
    plt.plot(TestSize['Date'], y_test, label='Actual Close Price', marker='o')
    plt.plot(TestSize['Date'], Predictions, label='Predicted Close Price', marker='x')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.title(f'FileName here Actual vs Predicted Close Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def accuracy(predictions, y_test):
    AllowedError = .05

    correctPredictions = np.mean((np.abs(predictions-y_test)/y_test) <= AllowedError)
    Accuracy = correctPredictions * 100
    return Accuracy

def r2(predictions, y_test):
    return r2_score(y_test,predictions)


def RunProgram(fileInput, target):

    fileInput['Date'] = pd.to_datetime(fileInput['Date'])

    fileInput.sort_values(['Date'],inplace=True)

    fileInput['Open1'] = fileInput['Open'].shift(1)
    fileInput['High1'] = fileInput['High'].shift(1)
    fileInput['Low1'] = fileInput['Low'].shift(1)
    fileInput['Close1'] = fileInput['Close'].shift(1)
    fileInput['High2'] = fileInput['High'].shift(2)
    fileInput['Low2'] = fileInput['Low'].shift(2)
    fileInput['Close2'] = fileInput['Close'].shift(2)
    fileInput['Open2'] = fileInput['Open'].shift(2)

    FeaturesToTrain = ['Open1','High1','Low1','Close1','High2','Low2','Close2','Open2']
    target = target

    x_train = fileInput[FeaturesToTrain]
    y_train = fileInput[target]

    makeTree = RandomForestRegressor(n_estimators=100,random_state=42)
    makeTree.fit(x_train, y_train)

    testData = int(len(fileInput)*0.8)
    TestSize = fileInput.iloc[testData:]
    x_test = TestSize[FeaturesToTrain]
    y_test = TestSize[target]
    predictions = makeTree.predict(x_test)
    # mse = mean_squared_error(y_test, predictions)
    # mae = mean_absolute_error(y_test, predictions)

    makeGraph(fileInput,TestSize,y_test,predictions,)
    # print(mae)
    # print(mse)

    accSize = accuracy(predictions, y_test)
    r2 = r2_score(y_test,predictions)
    print(accSize)
    print(r2)






RunProgram(bit_Coin_File, "Close")
print("")
RunProgram(Ethereum_File, "Close")

