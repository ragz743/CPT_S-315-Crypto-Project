import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def evaluate_predictions(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    # Avoid division by zero
    mape = np.mean(np.abs((actual - predicted) / np.where(actual == 0, 1, actual))) * 100
    return mae, rmse, mape
