# Dimensionality Reduction & Stock Price Prediction

This project demonstrates two key concepts in machine learning and time series forecasting:
1. **Dimensionality Reduction** – Using Principal Component Analysis (PCA) for visualization.
2. **Stock Price Prediction** – Using Time Series Forecasting with ARIMA.

## Project Overview

The project is divided into two parts:

### Part 1: Dimensionality Reduction
Objective: Reduce high-dimensional data to 2D for visualization.

#### Dataset:
- **Dataset Used**: Iris dataset (or any dataset with high-dimensional features).
- **Task**: 
  - Apply PCA to reduce the dataset to two principal components.
  - Visualize the reduced data using a scatter plot.

#### Deliverables:
- Reduced dataset (2D representation).
- Scatter plot showing the reduced dimensions.

### Part 2: Stock Price Prediction Using Time Series Forecasting
Objective: Predict future stock prices based on historical data using ARIMA.

#### Dataset:
- **Dataset Used**: `stock_prices.csv`
  - **Columns**:
    - Date (Timestamp)
    - Open (Opening price)
    - Close (Closing price)
    - Volume (Trade volume)

#### Project Steps:
1. **Data Preprocessing**:
   - Convert the `Date` column to DateTime format.
   - Handle missing values (if any).
   - Set the `Date` as the index for time-series analysis.
2. **Exploratory Data Analysis (EDA)**:
   - Plot the time series of Close prices to observe trends.
   - Analyze seasonality, trends, and noise in the data.
3. **Feature Engineering**:
   - Create lag features (previous day’s close price as a feature).
   - Perform rolling window calculations (moving averages, etc.).
4. **Model Training**:
   - Train an ARIMA model for forecasting.
   - Tune ARIMA parameters (p, d, q) for better accuracy.
5. **Model Evaluation & Visualization**:
   - Compare actual vs. predicted stock prices.
   - Plot the forecast vs. real stock prices.
   - Analyze forecasting errors (MAE, RMSE, MAPE).

#### Deliverables:
- Trained ARIMA model for stock forecasting.
- Time-series plots comparing predictions vs. actual prices.
- Insights on stock trends, seasonality, and forecast accuracy.

## Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- statsmodels

## Installation

linked in profile link: www.linkedin.com/in/s-berlin-samvel-pandian007
