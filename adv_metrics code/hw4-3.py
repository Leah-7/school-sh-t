import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('vixlarge.csv')

# Convert 'DATE' column to datetime and 'VIX' to numeric (if not already done)
data['DATE'] = pd.to_datetime(data['DATE'])
data['VIX'] = pd.to_numeric(data['VIX'], errors='coerce')

# Create the lag feature
data['VIX_lag'] = data['VIX'].shift(1)

# Drop any rows with missing data in the VIX or VIX_lag columns
data = data.dropna(subset=['VIX', 'VIX_lag'])

# Initialize lists to store MSFE and MAFE values for each model
msfe_ridge_1, msfe_ridge_10 = [], []
msfe_lasso_1, msfe_lasso_10 = [], []

mafe_ridge_1, mafe_ridge_10 = [], []
mafe_lasso_1, mafe_lasso_10 = [], []

# Define the window size (3000)
window_size = 3000

# Loop through the data with a rolling window
for start in range(len(data) - window_size):
    # Define the training and testing data for the window
    train_data = data.iloc[start:start + window_size]
    test_data = data.iloc[start + window_size:start + window_size + 1]
    
    # Define the independent variable (X) and dependent variable (y)
    X_train = train_data[['VIX_lag']]  # Only the lag of VIX
    y_train = train_data['VIX']  # VIX is the dependent variable
    X_test = test_data[['VIX_lag']]  # Predict the next value using the lag
    y_test = test_data['VIX']
    
    # Ridge regression with lambda = 1
    ridge_1 = Ridge(alpha=1)
    ridge_1.fit(X_train, y_train)
    pred_ridge_1 = ridge_1.predict(X_test)
    
    # Ridge regression with lambda = 10
    ridge_10 = Ridge(alpha=10)
    ridge_10.fit(X_train, y_train)
    pred_ridge_10 = ridge_10.predict(X_test)
    
    # Lasso regression with lambda = 1
    lasso_1 = Lasso(alpha=1)
    lasso_1.fit(X_train, y_train)
    pred_lasso_1 = lasso_1.predict(X_test)
    
    # Lasso regression with lambda = 10
    lasso_10 = Lasso(alpha=10)
    lasso_10.fit(X_train, y_train)
    pred_lasso_10 = lasso_10.predict(X_test)
    
    # Calculate MSFE and MAFE for each method
    msfe_ridge_1.append(mean_squared_error(y_test, pred_ridge_1))
    msfe_ridge_10.append(mean_squared_error(y_test, pred_ridge_10))
    msfe_lasso_1.append(mean_squared_error(y_test, pred_lasso_1))
    msfe_lasso_10.append(mean_squared_error(y_test, pred_lasso_10))

    mafe_ridge_1.append(mean_absolute_error(y_test, pred_ridge_1))
    mafe_ridge_10.append(mean_absolute_error(y_test, pred_ridge_10))
    mafe_lasso_1.append(mean_absolute_error(y_test, pred_lasso_1))
    mafe_lasso_10.append(mean_absolute_error(y_test, pred_lasso_10))

# Compute average MSFE and MAFE for each method
msfe_ridge_1_avg = np.mean(msfe_ridge_1)
msfe_ridge_10_avg = np.mean(msfe_ridge_10)
msfe_lasso_1_avg = np.mean(msfe_lasso_1)
msfe_lasso_10_avg = np.mean(msfe_lasso_10)

mafe_ridge_1_avg = np.mean(mafe_ridge_1)
mafe_ridge_10_avg = np.mean(mafe_ridge_10)
mafe_lasso_1_avg = np.mean(mafe_lasso_1)
mafe_lasso_10_avg = np.mean(mafe_lasso_10)

# Print the results
print(f"Ridge (λ=1) MSFE: {msfe_ridge_1_avg}")
print(f"Ridge (λ=10) MSFE: {msfe_ridge_10_avg}")
print(f"Lasso (λ=1) MSFE: {msfe_lasso_1_avg}")
print(f"Lasso (λ=10) MSFE: {msfe_lasso_10_avg}")

print(f"Ridge (λ=1) MAFE: {mafe_ridge_1_avg}")
print(f"Ridge (λ=10) MAFE: {mafe_ridge_10_avg}")
print(f"Lasso (λ=1) MAFE: {mafe_lasso_1_avg}")
print(f"Lasso (λ=10) MAFE: {mafe_lasso_10_avg}")




