import matplotlib.pyplot as plt


def plot_predictions(dates, actual, predicted):
    plt.figure(figsize=(12, 6))
    plt.plot(dates, actual, label='Actual Prices')
    plt.plot(dates, predicted, label='Predicted Prices', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Bitcoin Price')
    plt.title('Bitcoin Price Prediction vs Actual')
    plt.legend()
    plt.show()
