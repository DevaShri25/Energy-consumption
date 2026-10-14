# World Energy Consumption Forecasting using LSTM

## Overview
This project uses historical energy consumption data to forecast future annual primary energy consumption for India using a Long Short-Term Memory (LSTM) neural network.

## Dataset
- Source: [Kaggle – World Energy Consumption](https://www.kaggle.com/datasets/patricklford/world-energy-consumption)
- Country selected: India
- Historical data available in the dataset: 1965–2023

## Workflow
1. Load and inspect the dataset.
2. Convert yearly columns from wide format to long format.
3. Handle missing values and duplicate years.
4. Normalize the data using MinMaxScaler fitted on the training period.
5. Create five-year input sequences to predict the next year.
6. Split the sequences chronologically into training and testing sets.
7. Train an LSTM model using TensorFlow/Keras.
8. Evaluate predictions using MAE, RMSE, R² and MAPE.
9. Generate recursive forecasts for the next ten years.

## Model
- Architecture: LSTM (32 units) followed by Dense (1)
- Optimizer: Adam
- Loss function: Mean Squared Error
- Training epochs: 50
- Batch size: 8

## Current LSTM Evaluation
- MAE: 3.1339
- RMSE: 3.5510
- R²: 0.2269
- MAPE: 9.72%

These metrics were obtained on a test set of 12 sequences and may vary with data processing or training settings.

## Limitations
The ten-year forecasts are model estimates, not actual future measurements. Prediction uncertainty can increase over longer forecast horizons. The model uses historical consumption as its primary feature and does not explicitly model population, economic conditions or energy policies.

## Technologies
Python, Pandas, NumPy, Matplotlib, Scikit-learn, TensorFlow/Keras, Google Colab.
