import requests

# Replace this with the correct GDELT dataset URL
url = "http://data.gdeltproject.org/events/20231201.export.CSV.zip"  

try:
    print("Starting download...")
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise error for HTTP issues

    # Save the file
    with open("gdelt_data.zip", "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):  # Download in chunks
            file.write(chunk)

    print("Download completed successfully!")

except requests.exceptions.RequestException as e:
    print(f"Download failed: {e}")

import zipfile

# Path to the ZIP file
zip_path = "gdelt_data.zip"
extract_to = "extracted_data"

# Extract the ZIP file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)
print(f"Files extracted to: {extract_to}")

import pandas as pd

# Load the CSV file
file_path = "C:/econometrics/hw4/myenv/Scripts/20231201.export.CSV"




# Try specifying a tab delimiter
chunk_size = 10000
for chunk in pd.read_csv(file_path, chunksize=chunk_size, delimiter="\t"):
    print(chunk.head())
print(chunk.columns)

import pandas as pd


import time

max_retries = 5
for attempt in range(max_retries):
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            # Process file normally
            break
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}. Retrying {attempt + 1}/{max_retries}...")
        time.sleep(5)  # Wait before retrying
# Load a GKG file
gkg_file = "path_to_your_gkg_file.csv"
gkg_df = pd.read_csv(gkg_file, sep='\t', header=None)

import os
if os.path.exists(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(SAVE_DIR)
else:
    print(f"File {zip_path} not found. Skipping extraction.")


# Inspect the 'SourceCommonName' column for media outlets
gkg_df.columns = [...]  # Assign appropriate column names (check GDELT documentation)
chinese_media = gkg_df[gkg_df['SourceCommonName'].str.contains('.cn', na=False)]
print(chinese_media['SourceCommonName'].value_counts())