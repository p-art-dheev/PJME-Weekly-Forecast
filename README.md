# PJME Weekly Power Consumption Forecast

A comprehensive time series forecasting project analyzing and predicting weekly power consumption for the Pennsylvania-Jersey-Maryland Interconnection (PJME) electricity grid.

## Project Overview

This notebook demonstrates multiple approaches to forecasting weekly power consumption using statistical and deep learning techniques. The analysis includes data preprocessing, exploratory visualizations, stationarity tests, and model comparison.

## Dataset

- **Source:** PJME hourly power consumption data
- **Aggregation:** Hourly readings aggregated to weekly totals (168 hours per week)
- **Time Range:** 2002-2015 (training data), 2016 (test data)
- **Train/Test Split:** ~730 weeks training, ~105 weeks testing

## Key Components

### 1. **Data Exploration**
- Time series visualization with trend analysis
- Seasonal patterns (monthly and yearly distributions)
- Rolling mean and variance (52-week windows)
- Classical additive decomposition (trend, seasonality, residual)

### 2. **Stationarity Tests**
- Augmented Dickey-Fuller (ADF) test
- KPSS test
- Autocorrelation (ACF) and Partial Autocorrelation (PACF) plots

### 3. **Forecasting Models**

#### Linear Time Series Models
- **SARIMA(1,1,1)(1,1,1)[52]** — Baseline seasonal ARIMA
- **SARIMA(1,1,2)(2,1,0)[52]** — Auto-tuned via `auto_arima` (best AIC)

#### Non-Linear Model
- **SETAR(2,1)** — Self-exciting threshold autoregressive model with regime switching

#### Deep Learning
- **LSTM(64-32)** — Stacked LSTM with dropout and early stopping

### 4. **Model Evaluation**
Metrics compared across all models:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)

## Files

- `TSA-LoadForecasting.ipynb` — Complete analysis and modeling notebook
- `PJME_hourly.csv` — Source hourly data (requires external download)
- `PJME_weekly.csv` — Aggregated weekly data (generated in notebook)

## Requirements

- pandas
- numpy
- matplotlib
- scikit-learn
- statsmodels
- pmdarima
- tensorflow / keras

## Usage

Run the Jupyter notebook to:
1. Load and preprocess PJME hourly data
2. Perform exploratory data analysis
3. Conduct stationarity and autocorrelation analysis
4. Train and evaluate all four forecasting models
5. Generate performance comparisons

## Key Findings

The analysis reveals:
- Weekly power consumption exhibits strong annual seasonality (52-week cycle)
- Clear trend component with seasonal patterns
- Multiple forecasting approaches each capture different aspects of the time series
- LSTM model captures complex non-linear patterns with competitive performance
