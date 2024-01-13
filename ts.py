# Data manipulation
# ==============================================================================
import numpy as np
import pandas as pd

# Plots
# ==============================================================================
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['font.size'] = 10

# Modeling and Forecasting
# ==============================================================================
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.ForecasterAutoregCustom import ForecasterAutoregCustom
from skforecast.ForecasterAutoregDirect import ForecasterAutoregDirect
from skforecast.model_selection import grid_search_forecaster
from skforecast.model_selection import backtesting_forecaster
from skforecast.utils import save_forecaster
from skforecast.utils import load_forecaster

# Warnings configuration
# ==============================================================================
import warnings

fix_step = 36

def predict(df):
    # Assuming you want to predict 'net_count'
    target_variable = 'net_count'

    # Create and train forecaster
    forecaster = ForecasterAutoreg(
        regressor=RandomForestRegressor(random_state=123),
        lags=6  # Adjust the number of lags as needed
    )

    forecaster.fit(y=df[target_variable])

    # Predictions
    steps = fix_step  # Adjust the number of steps as needed
    predictions = forecaster.predict(steps=steps)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df[target_variable], label='Original Data')
    plt.plot(pd.date_range(start=df.index[-1], periods=steps + 1, freq='S')[1:], predictions, label='Forecast', color='red')
    plt.title('Time Series Data with Forecast')
    plt.xlabel('Time')
    plt.ylabel(target_variable)
    plt.legend()
    plt.show()

    # Print the predictions
    print("Predictions:")
    print(predictions)

def main():
    # Data download
    # ==============================================================================
    url = 'https://raw.githubusercontent.com/JoaquinAmatRodrigo/skforecast/master/data/h2o_exog.csv'
    data = pd.read_csv(url, sep=',')

    # Data preparation
    # ==============================================================================
    data = data.rename(columns={'fecha': 'date'})
    data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')
    data = data.set_index('date')
    data = data.rename(columns={'x': 'y'})
    data = data.asfreq('MS')
    data = data.sort_index()
    data.head()

    print(f'Number of rows with missing values: {data.isnull().any(axis=1).mean()}')

    # Verify that a temporary index is complete
    # ==============================================================================
    (data.index == pd.date_range(start=data.index.min(),
                                end=data.index.max(),
                                freq=data.index.freq)).all()

    # Split data into train-test
    # ==============================================================================
    steps = fix_step
    data_train = data[:-steps]
    data_test  = data[-steps:]

    print(f"Train dates : {data_train.index.min()} --- {data_train.index.max()}  (n={len(data_train)})")
    print(f"Test dates  : {data_test.index.min()} --- {data_test.index.max()}  (n={len(data_test)})")

    fig, ax = plt.subplots(figsize=(6, 2.5))
    data_train['y'].plot(ax=ax, label='train')
    data_test['y'].plot(ax=ax, label='test')
    ax.legend()
    plt.show()




    # Create and train forecaster
    # ==============================================================================
    forecaster = ForecasterAutoreg(
                    regressor = RandomForestRegressor(random_state=123),
                    lags      = 6
                )

    forecaster.fit(y=data_train['y'])
    forecaster

    # Predictions
    # ==============================================================================
    steps = fix_step
    predictions = forecaster.predict(steps=steps)
    predictions.head(5)

    # Plot
    # ==============================================================================
    fig, ax = plt.subplots(figsize=(6, 2.5))
    data_train['y'].plot(ax=ax, label='train')
    data_test['y'].plot(ax=ax, label='test')
    predictions.plot(ax=ax, label='predictions')
    ax.legend()
    plt.show()

    # Test error
    # ==============================================================================
    error_mse = mean_squared_error(
                    y_true = data_test['y'],
                    y_pred = predictions
                )

    print(f"Test error (mse): {error_mse}")
    
if __name__ == "__main__":
    main()