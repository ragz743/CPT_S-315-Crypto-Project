import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

bit_Coin_File = pd.read_csv("../Data/coin_Bitcoin.csv")
Ethereum_File = pd.read_csv("../Data/coin_Ethereum.csv")


def makeGraph(fileName, test_data, y_test, predictions):
    plt.figure(figsize=(10, 6))
    plt.plot(test_data['Date'], y_test, label='Actual Price', marker='o')
    plt.plot(test_data['Date'], predictions, label='Predicted Price', marker='x')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title(f'{fileName} - Actual vs Predicted Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def accuracy(predictions, y_test):
    allowed_error = 0.10
    correct_predictions = np.mean((np.abs(predictions - y_test) / y_test) <= allowed_error)
    return correct_predictions * 100


def RunProgram(fileInput, target, fileName):
    fileInput['Date'] = pd.to_datetime(fileInput['Date'])
    fileInput.sort_values(['Date'], inplace=True)

    fileInput['Open1'] = fileInput['Open'].shift(1)
    fileInput['High1'] = fileInput['High'].shift(1)
    fileInput['Low1'] = fileInput['Low'].shift(1)
    fileInput['Close1'] = fileInput['Close'].shift(1)
    fileInput['High2'] = fileInput['High'].shift(2)
    fileInput['Low2'] = fileInput['Low'].shift(2)
    fileInput['Close2'] = fileInput['Close'].shift(2)
    fileInput['Open2'] = fileInput['Open'].shift(2)


    fileInput.dropna(inplace=True)
    split_index = int(len(fileInput) * 0.8)
    train_data = fileInput.iloc[:split_index]
    test_data = fileInput.iloc[split_index:]

    #features = ['Close1','Close2']
    features = ['Open1', 'High1', 'Low1', 'Close1', 'High2', 'Low2', 'Close2', 'Open2']

    x_train = train_data[features]
    y_train = train_data[target]
    x_test = test_data[features]
    y_test = test_data[target]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    makeGraph(fileName, test_data, y_test, predictions)


    acc = accuracy(predictions, y_test)
    r2 = r2_score(y_test, predictions)
    print(f"{fileName} Accuracy: {acc:.2f}%")
    print(f"{fileName} R^2 Score: {r2:.4f}")



RunProgram(bit_Coin_File, "Close", "Bitcoin")
print()
RunProgram(Ethereum_File, "Close", "Ethereum")
