import pandas as pd
import numpy as np
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

# Load the data
data = pd.read_csv('vixlarge.csv')
data['DATE'] = pd.to_datetime(data['DATE'])
data['VIX_lag'] = data['VIX'].shift(1)
data = data.dropna()

# Train-test split
X = data[['VIX_lag']]
y = data['VIX']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# Bagging Tree Regressor
bagging_model = BaggingRegressor(
    n_estimators=100,
    max_samples=0.8,
    random_state=42
)
bagging_model.fit(X_train, y_train)

# Predictions
y_pred = bagging_model.predict(X_test)

# Compute MSFE and MAFE
msfe = mean_squared_error(y_test, y_pred)
mafe = mean_absolute_error(y_test, y_pred)

print(f"MSFE: {msfe:.4f}")
print(f"MAFE: {mafe:.4f}")
