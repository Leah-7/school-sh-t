from xgboost import XGBRegressor
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load the data
data = pd.read_csv('vixlarge.csv')
data['DATE'] = pd.to_datetime(data['DATE'])
data['VIX'] = pd.to_numeric(data['VIX'], errors='coerce')
data['VIX_lag'] = data['VIX'].shift(1)  # Lag feature
data = data.dropna()  # Drop NaN values

# Define forecast horizons and initialize results dictionary
horizons = [1, 5, 10, 22]
results = {h: {'MSFE': [], 'MAFE': []} for h in horizons}
window_size = 3000  # Sliding window size for training data

# Loop over forecast horizons
for h in horizons:
    for start in range(len(data) - window_size - h):
        # Split into training and test sets
        train_data = data.iloc[start:start + window_size]
        test_data = data.iloc[start + window_size + h - 1:start + window_size + h]

        # Define features (X) and target (y)
        X_train = train_data[['VIX_lag']]
        y_train = train_data['VIX']
        X_test = test_data[['VIX_lag']]
        y_test = test_data['VIX']

        # Initialize and fit the XGBoost model
        xgb_model = XGBRegressor(
            max_depth=10,
            n_estimators=100,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        xgb_model.fit(X_train, y_train)

        # Make predictions
        predictions = xgb_model.predict(X_test)

        # Calculate MSFE and MAFE
        msfe = mean_squared_error(y_test, predictions)
        mafe = mean_absolute_error(y_test, predictions)

        # Store results
        results[h]['MSFE'].append(msfe)
        results[h]['MAFE'].append(mafe)

# Compute the average MSFE and MAFE for each horizon
summary = []
for h in horizons:
    avg_msfe = np.mean(results[h]['MSFE'])
    avg_mafe = np.mean(results[h]['MAFE'])
    summary.append({'Horizon': h, 'MSFE': avg_msfe, 'MAFE': avg_mafe})

# Convert results to a DataFrame for easy viewing
results_df = pd.DataFrame(summary)
print(results_df)

