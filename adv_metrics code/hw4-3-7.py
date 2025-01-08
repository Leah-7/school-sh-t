import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# Load the data
data = pd.read_csv('vixlarge.csv')
data['DATE'] = pd.to_datetime(data['DATE'])

# Drop missing values (if any)
data = data.dropna()

# Generate 5 evenly spaced knots (indices)
num_knots = 5
knots = np.linspace(0, len(data) - 1, num_knots).astype(int)

# Convert "DATE" to numerical values for spline fitting
data['DATE_num'] = (data['DATE'] - data['DATE'].min()).dt.days

# Fit a cubic spline
spline = CubicSpline(data['DATE_num'][knots], data['VIX'][knots], bc_type='natural')

# Create a finer grid for smoother plot
date_range = np.linspace(data['DATE_num'].min(), data['DATE_num'].max(), 1000)
spline_values = spline(date_range)
plt.figure(figsize=(12, 6))

# Original data
plt.plot(data['DATE'], data['VIX'], label='Observed VIX', color='blue')

# Spline fit
plt.plot(pd.to_datetime(data['DATE'].min() + pd.to_timedelta(date_range, unit='D')),
         spline_values, label='Cubic Spline Fit', color='red', linestyle='--')

# Mark knots
plt.scatter(data['DATE'][knots], data['VIX'][knots], color='green', label='Knots (Nodes)', zorder=5)

# Labels and legend
plt.title('Cubic Spline Fit with 5 Knots (Nodes)')
plt.xlabel('DATE')
plt.ylabel('VIX')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

