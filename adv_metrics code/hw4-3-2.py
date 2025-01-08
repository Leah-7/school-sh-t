import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Load the data
data = pd.read_csv('vixlarge.csv')

# Convert 'DATE' column to datetime and 'VIX' to numeric
data['DATE'] = pd.to_datetime(data['DATE'])
data['VIX'] = pd.to_numeric(data['VIX'], errors='coerce')

# Create the lag feature and drop rows with missing values
data['VIX_lag'] = data['VIX'].shift(1)
data = data.dropna(subset=['VIX', 'VIX_lag'])

# Initialize lists to store MSFE and MAFE values
msfe_rf = []
mafe_rf = []

# Define the window size
window_size = 3000

# Loop through the data with a rolling window
for start in range(len(data) - window_size):
    # Define the training and testing data for the window
    train_data = data.iloc[start:start + window_size]
    test_data = data.iloc[start + window_size:start + window_size + 1]
    
    # Define the independent variable (X) and dependent variable (y)
    X_train = train_data[['VIX_lag']]  # Only the lag of VIX
    y_train = train_data['VIX']
    X_test = test_data[['VIX_lag']]
    y_test = test_data['VIX']
    
    # Scale the data (optional for Random Forest, but good for consistency)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Fit the Random Forest Regressor
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    
    # Predict the next value
    pred_rf = rf_model.predict(X_test_scaled)
    
    # Calculate MSFE and MAFE
    msfe_rf.append(mean_squared_error(y_test, pred_rf))
    mafe_rf.append(mean_absolute_error(y_test, pred_rf))

# Compute average MSFE and MAFE
msfe_rf_avg = np.mean(msfe_rf)
mafe_rf_avg = np.mean(mafe_rf)

# Print the results
print(f"Random Forest MSFE: {msfe_rf_avg}")
print(f"Random Forest MAFE: {mafe_rf_avg}")

from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

# Create the Random Forest model
rf = RandomForestRegressor(random_state=42)

# Perform Grid Search with Cross Validation
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train_scaled, y_train)

# Get the best parameters
print(f"Best Parameters: {grid_search.best_params_}")

# Use the best model for forecasting
best_rf = grid_search.best_estimator_
