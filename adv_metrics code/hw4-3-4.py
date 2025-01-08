import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load the data
data = pd.read_csv('vixlarge.csv')
data['DATE'] = pd.to_datetime(data['DATE'])
data['VIX'] = pd.to_numeric(data['VIX'], errors='coerce')
data['VIX_lag'] = data['VIX'].shift(1)
data = data.dropna()

# Forecast horizons
horizons = [1, 5, 10, 22]

# Initialize results storage
results = {h: {'tree': {'MSFE': [], 'MAFE': []}, 'rf': {'MSFE': [], 'MAFE': []}} for h in horizons}

# Rolling window length
window_size = 3000

# Perform forecasting for each horizon
for h in horizons:
    for start in range(len(data) - window_size - h):
        # Define the training and testing windows
        train_data = data.iloc[start:start + window_size]
        test_data = data.iloc[start + window_size + h - 1:start + window_size + h]
        
        # Features and target
        X_train = train_data[['VIX_lag']]
        y_train = train_data['VIX']
        X_test = test_data[['VIX_lag']]
        y_test = test_data['VIX']
        
        # Regression Tree Model
        tree_model = DecisionTreeRegressor(max_depth=10, min_samples_split=5, random_state=42)
        tree_model.fit(X_train, y_train)
        tree_pred = tree_model.predict(X_test)
        
        # Random Forest Model
        rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=5, random_state=42)
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        
        # Calculate errors
        results[h]['tree']['MSFE'].append(mean_squared_error(y_test, tree_pred))
        results[h]['tree']['MAFE'].append(mean_absolute_error(y_test, tree_pred))
        results[h]['rf']['MSFE'].append(mean_squared_error(y_test, rf_pred))
        results[h]['rf']['MAFE'].append(mean_absolute_error(y_test, rf_pred))

# Aggregate results
final_results = {}
for h in horizons:
    final_results[h] = {
        'Tree_MSFE': np.mean(results[h]['tree']['MSFE']),
        'Tree_MAFE': np.mean(results[h]['tree']['MAFE']),
        'RF_MSFE': np.mean(results[h]['rf']['MSFE']),
        'RF_MAFE': np.mean(results[h]['rf']['MAFE']),
    }

# Display results in a table
results_df = pd.DataFrame(final_results).T
print(results_df)