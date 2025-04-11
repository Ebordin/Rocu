import pandas as pd
import requests
from datetime import datetime
import os

# Example: fetch a public dataset (replace with your real data source)
url = "https://people.sc.fsu.edu/~jburkardt/data/csv/hw_200.csv"
response = requests.get(url)
response.raise_for_status()

# Save fetched data
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs("data", exist_ok=True)
filepath = f"data/collected_{timestamp}.csv"
with open(filepath, "w") as f:
    f.write(response.text)

print(f"✅ Data collected and saved to {filepath}")
