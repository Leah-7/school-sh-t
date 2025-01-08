import pandas as pd
import matplotlib.pyplot as plt
import os

# Debug current working directory
print("Current working directory:", os.getcwd())

try:
    # Load the data
    data = pd.read_csv('vixlarge.csv')
    
    # Check and debug column names
    print("Column names in the dataset:", data.columns)
    
    # Rename columns to ensure consistency
    data.columns = data.columns.str.strip().str.upper()  # Convert all column names to uppercase
    
    # Ensure 'DATE' column is parsed as datetime
    if 'DATE' in data.columns:
        data['DATE'] = pd.to_datetime(data['DATE'])
    else:
        print("Error: 'DATE' column not found in the dataset. Check column names.")
        print("Columns found:", data.columns)
        exit()

    # Plot the data
    plt.figure(figsize=(12, 6))
    plt.plot(data['DATE'], data['VIX'], label='VIX', color='blue')  # Use consistent column names
    plt.title('VIX Index Over Time')
    plt.xlabel('DATE')
    plt.ylabel('VIX')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.show()
except FileNotFoundError:
    print("Error: The file 'vixlarge.csv' was not found. Please check the file path.")

