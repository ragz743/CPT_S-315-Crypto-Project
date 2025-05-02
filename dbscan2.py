import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from matplotlib.lines import Line2D
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from datetime import timedelta

df = pd.read_csv("coin_Bitcoin.csv")
df2 = pd.read_csv("coin_Ethereum.csv")
# Preprocessing
# Filter out rows with zero or missing volume
df = df[df['Volume'] > 0].copy()
df = df.dropna()

df2 = df2[df2['Volume'] > 0].copy()
df2 = df2.dropna()

# Create a 30-day trend feature based on the difference in closing price
window_size = 30
df['Trend'] = df['Close'].rolling(window=window_size).apply(lambda x: x.iloc[-1] - x.iloc[0])
df['Trend'].fillna(0, inplace=True)

window_size = 30
df2['Trend'] = df2['Close'].rolling(window=window_size).apply(lambda x: x.iloc[-1] - x.iloc[0])
df2['Trend'].fillna(0, inplace=True)

# Define bull (1) and bear (0) markets based on trend direction
df['Market_Label'] = (df['Trend'] > 0).astype(int)
df2['Market_Label'] = (df2['Trend'] > 0).astype(int)

plot_df = df.copy()
# print(plot_df['Cluster'].unique())
plot_df['Color'] = plot_df['Market_Label'].map({0: 'red', 1: 'green'})

fig, ax = plt.subplots(figsize=(14, 6))

# Color line segments based on bull/bear trend
for i in range(1, len(plot_df)):
    color = 'green' if plot_df.iloc[i]['Market_Label'] == 1 else 'red'
    ax.plot(plot_df['Date'].iloc[i-1:i+1], plot_df['Close'].iloc[i-1:i+1], color=color, linewidth=1.5)

# Format x-axis
plt.xticks([])

ax.set_title("Bull (Green) vs Bear (Red) Trends")
ax.set_xlabel("Date")
ax.set_ylabel("Close Price")

# Cluster legend
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Bull Market', markerfacecolor='green', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='Bear Market', markerfacecolor='red', markersize=10)
]
ax.legend(handles=legend_elements, title="Market Trend")

plt.tight_layout()
# plt.show()


df_forecast = df.copy()
df_forecast['Close_Lag_1'] = df_forecast['Close'].shift(1)
df_forecast['Close_Lag_2'] = df_forecast['Close'].shift(2)
df_forecast['Volume_Lag_1'] = df_forecast['Volume'].shift(1)
df_forecast['Marketcap_Lag_1'] = df_forecast['Marketcap'].shift(1)
df_forecast['Market_Label'] = df_forecast['Market_Label'].shift(1)

df2_forecast = df2.copy()
df2_forecast['Close_Lag_1'] = df2_forecast['Close'].shift(1)
df2_forecast['Close_Lag_2'] = df2_forecast['Close'].shift(2)
df2_forecast['Volume_Lag_1'] = df2_forecast['Volume'].shift(1)
df2_forecast['Marketcap_Lag_1'] = df2_forecast['Marketcap'].shift(1)
df2_forecast['Market_Label'] = df2_forecast['Market_Label'].shift(1)

# Drop rows with NaNs from shifting
df_forecast.dropna(inplace=True)

df2_forecast.dropna(inplace=True)

# Features for regression
forecast_features = ['Close_Lag_1', 'Close_Lag_2', 'Volume_Lag_1', 'Marketcap_Lag_1', 'Market_Label']
X_reg = df_forecast[forecast_features]
y_reg = df_forecast['Close']

forecast_features2 = ['Close_Lag_1', 'Close_Lag_2', 'Volume_Lag_1', 'Marketcap_Lag_1', 'Market_Label']
X_reg2 = df2_forecast[forecast_features2]
y_reg2 = df2_forecast['Close']

# Time-based train/test split
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, shuffle=False)

X_train_reg2, X_test_reg2, y_train_reg2, y_test_reg2 = train_test_split(X_reg2, y_reg2, test_size=0.2, shuffle=False)

# Train linear regression model
linreg = LinearRegression()
linreg.fit(X_train_reg, y_train_reg)
y_pred_reg = linreg.predict(X_test_reg)

linreg2 = LinearRegression()
linreg2.fit(X_train_reg2, y_train_reg2)
y_pred_reg2 = linreg2.predict(X_test_reg2)

# Evaluate
mse = mean_squared_error(y_test_reg, y_pred_reg)
r2 = r2_score(y_test_reg, y_pred_reg)

mse2 = mean_squared_error(y_test_reg2, y_pred_reg2)
r22 = r2_score(y_test_reg2, y_pred_reg2)

print(r2)
print(r22)
print(mse)
print(mse2)

plot_df = df_forecast.iloc[-len(y_test_reg):].copy()
plot_df['Predicted_Close'] = y_pred_reg
plot_df['Actual_Close'] = y_test_reg.values

# Plot actual vs predicted close prices
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(plot_df['Date'], plot_df['Actual_Close'], label='Actual Close', linewidth=2)
ax.plot(plot_df['Date'], plot_df['Predicted_Close'], label='Predicted Close', linestyle='--', linewidth=2)

ax.set_title('Actual vs Predicted Close Prices (BitCoin)')
ax.set_xlabel('Date')
ax.set_ylabel('Close Price')
ax.legend()
ax.grid(True)
plt.xticks([])
plt.tight_layout()
# plt.show()

plot_df2 = df2_forecast.iloc[-len(y_test_reg2):].copy()
plot_df2['Predicted_Close'] = y_pred_reg2
plot_df2['Actual_Close'] = y_test_reg2.values

# Plot actual vs predicted close prices
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(plot_df2['Date'], plot_df2['Actual_Close'], label='Actual Close', linewidth=2)
ax.plot(plot_df2['Date'], plot_df2['Predicted_Close'], label='Predicted Close', linestyle='--', linewidth=2)

ax.set_title('Actual vs Predicted Close Prices (Ethereum)')
ax.set_xlabel('Date')
ax.set_ylabel('Close Price')
ax.legend()
ax.grid(True)
plt.xticks([])
plt.tight_layout()
# plt.show()

plot_df['Residual'] = plot_df['Actual_Close'] - plot_df['Predicted_Close']

fig, axs = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# Plot predictions vs actual
axs[0].plot(plot_df['Date'], plot_df['Actual_Close'], label='Actual Close', linewidth=2)
axs[0].plot(plot_df['Date'], plot_df['Predicted_Close'], label='Predicted Close', linestyle='--', linewidth=2)
axs[0].set_title("Actual vs Predicted Close Prices")
axs[0].set_ylabel("Close Price")
axs[0].legend()
axs[0].grid(True)

# Plot residuals
axs[1].plot(plot_df['Date'], plot_df['Residual'], label='Residual (Error)', color='red')
axs[1].set_title("Prediction Error (Residuals)")
axs[1].set_ylabel("Error")
axs[1].set_xlabel("Date")
axs[1].grid(True)

plt.xticks([])

plt.tight_layout()
# plt.show()

future_days = 30

# Ensure 'Date' column is datetime
df_forecast['Date'] = pd.to_datetime(df_forecast['Date'])
last_row = df_forecast.iloc[-1]
last_date = last_row['Date']

# Reinitialize values for future forecasting
close_lag_1 = last_row['Close']
close_lag_2 = last_row['Close_Lag_1']
volume_lag_1 = last_row['Volume']
marketcap_lag_1 = last_row['Marketcap']
last_market_label = int(last_row['Market_Label'])

# Predict forward 30 days
future_predictions = []
future_dates = []

for i in range(future_days):
    X_future = pd.DataFrame([{
    'Close_Lag_1': close_lag_1,
    'Close_Lag_2': close_lag_2,
    'Volume_Lag_1': volume_lag_1,
    'Marketcap_Lag_1': marketcap_lag_1,
    'Market_Label' : last_market_label
    }])
    pred_close = linreg.predict(X_future)[0]
    close_lag_2 = close_lag_1
    close_lag_1 = pred_close

    # Save prediction
    next_date = last_date + timedelta(days=1 + i)
    future_dates.append(next_date)
    future_predictions.append(pred_close)

# Create DataFrame
future_df = pd.DataFrame({
    'Date': future_dates,
    'Close': future_predictions,
    'Type': 'Forecast'
})

# Combine with historical close data for plotting
historical_df = plot_df[['Date', 'Actual_Close']].rename(columns={'Actual_Close': 'Close'})
historical_df['Type'] = 'Historical'
historical_df['Date'] = pd.to_datetime(historical_df['Date'])

combined_df = pd.concat([historical_df, future_df], ignore_index=True)

# Plot the extended forecast
fig, ax = plt.subplots(figsize=(14, 6))
for label, group in combined_df.groupby('Type'):
    ax.plot(group['Date'], group['Close'], label=label, linestyle='--' if label == 'Forecast' else '-', linewidth=2)

ax.set_title("Bitcoin 30-Day Price Forecast")
ax.set_xlabel("Date")
ax.set_ylabel("Close Price")
ax.legend()
ax.grid(True)
plt.xticks([])
plt.tight_layout()
plt.show()