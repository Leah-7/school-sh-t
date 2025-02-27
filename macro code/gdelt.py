import os.path
import re
import zipfile
from multiprocessing.pool import ThreadPool
from timeit import default_timer as timer
from urllib.parse import urlparse
from urllib.request import urlretrieve

import pandas as pd

# 01. DEFINE DATA RANGE OF FILES WANTED USING PANDAS
dateRange = pd.date_range(
    start='2020-01-01', end='2020-06-30', freq='1D').tolist()
print("Crawling data from", dateRange[0].strftime(
    '%Y-%m-%d'), "to", dateRange[-1].strftime('%Y-%m-%d'))

# 02. DEFINE GDELT URL AND LOCAL FOLDER URL
gdelturl = "http://data.gdeltproject.org/events/%s.export.CSV.zip"
both_urls = [((gdelturl % ts.strftime('%Y%m%d')),
              "/Users/leah/Desktop/DATASET02/") for ts in dateRange]

# 03. FUNCTION TO MATCH WEB URL WITH LOCAL URL (LOCAL STORAGE)
def crawl_url(urlFolder):
    url = urlFolder[0] 
    folder = urlFolder[1]

    # Use the URL as the name of the file
    filename = folder + os.path.basename(urlparse(url).path)

    if (os.path.isfile(filename)):
        return url, filename, None

    try:
        local_url, http_message = urlretrieve(url, filename)
        return url, local_url, None
    except Exception as e:
        return url, None, e


# 04. FUNCTION TO DOWNLOAD AND EXTRACT FILES
def download_files(both_urls):
    print("Downloading and extracting", len(both_urls), "documents")

    start = timer()
    # Downloading the files (parallel)
    results = ThreadPool(32).imap_unordered(crawl_url, both_urls)

    for url, local_url, error in results:
        if error is None:
            # Extracting the files in a same folder
            z = zipfile.ZipFile(file=local_url, mode='r')
            z.extractall(path='/Users/leah/Desktop/DATASET02/'+'tmp/')
            print("%r ✔️ %.2fs" % (local_url, timer() - start))

        else:
            print("Error fetching %r: %s" % (url, error))


download_files(both_urls)

print("Finished!")





