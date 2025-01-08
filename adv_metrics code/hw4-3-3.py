from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import numpy as np

# Load the data
data = pd.read_csv('vixlarge.csv')

# Convert 'DATE' column to datetime and 'VIX' to numeric
data['DATE'] = pd.to_datetime(data['DATE'])
data['VIX'] = pd.to_numeric(data['VIX'], errors='coerce')

# Create lagged feature
data['VIX_lag'] = data['VIX'].shift(1)
data = data.dropna()

# Initialize lists to store MSFE and MAFE values
msfe_bagging = []
mafe_bagging = []

# Define rolling window size
window_size = 3000

# Loop through rolling windows
for start in range(len(data) - window_size):
    train_data = data.iloc[start:start + window_size]
    test_data = data.iloc[start + window_size:start + window_size + 1]
    
    X_train = train_data[['VIX_lag']]
    y_train = train_data['VIX']
    X_test = test_data[['VIX_lag']]
    y_test = test_data['VIX']
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Bagging Regressor with estimator compatibility
    bagging_model = BaggingRegressor(
        estimator=DecisionTreeRegressor(max_depth=10),  # For older versions of sklearn
        n_estimators=100,
        max_samples=0.8,
        random_state=42
    )
    
    bagging_model.fit(X_train_scaled, y_train)
    pred_bagging = bagging_model.predict(X_test_scaled)
    
    # Store MSFE and MAFE
    msfe_bagging.append(mean_squared_error(y_test, pred_bagging))
    mafe_bagging.append(mean_absolute_error(y_test, pred_bagging))

# Compute average MSFE and MAFE
msfe_bagging_avg = np.mean(msfe_bagging)
mafe_bagging_avg = np.mean(mafe_bagging)

# Print results
print(f"Bagging Tree MSFE: {msfe_bagging_avg}")
print(f"Bagging Tree MAFE: {mafe_bagging_avg}")
