import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
# Load the data
data = pd.read_csv('vixlarge.csv')
data['DATE'] = pd.to_datetime(data['DATE'])

# Example Data (Replace with your actual dataset)
data['DATE'] = pd.to_datetime(data['DATE'])
data['DATE_num'] = (data['DATE'] - data['DATE'].min()).dt.days

# Convert specific dates to numerical values
specific_dates = ['1990-01-02', '2002-08-05','2006-03-14','2008-11-20','2017-03-24']
specific_knots = pd.to_datetime(specific_dates)
specific_knots_num = (specific_knots - data['DATE'].min()).days

# Combine specific knots with additional points
additional_knots = [data['DATE_num'].min(), data['DATE_num'].max()]
all_knots = sorted(list(set(specific_knots_num.tolist() + additional_knots)))

# Ensure there is a value for each knot
knot_values = data.loc[data['DATE_num'].isin(all_knots), 'VIX'].values

# Fit cubic spline
spline = CubicSpline(all_knots, knot_values, bc_type='natural')

# Plot
plt.figure(figsize=(12, 6))
plt.plot(data['DATE'], data['VIX'], label='Observed VIX', color='blue')
date_range = np.linspace(data['DATE_num'].min(), data['DATE_num'].max(), 1000)
spline_values = spline(date_range)
plt.plot(pd.to_datetime(data['DATE'].min() + pd.to_timedelta(date_range, unit='D')), spline_values, 
         label='Cubic Spline Fit', color='red', linestyle='--')
plt.scatter(pd.to_datetime(data['DATE'].min() + pd.to_timedelta(all_knots, unit='D')), 
            knot_values, color='green', label='Knots (Nodes)', zorder=5)
plt.title("Cubic Spline Fit with Specific Knots")
plt.xlabel('DATE')
plt.ylabel('VIX')
plt.legend()
plt.grid()
plt.show()

