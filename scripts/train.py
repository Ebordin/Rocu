import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import joblib
import os
from datetime import datetime

# Step 1: Download a sample Python code dataset
url = "https://raw.githubusercontent.com/sobolevn/awesome-cryptography/master/README.md"
response = requests.get(url)
response.raise_for_status()

# Step 2: Treat the content as a list of pseudo code snippets split by empty lines
code_snippets = response.text.split('\n\n')

# Step 3: Vectorize the code snippets
vectorizer = TfidfVectorizer(max_features=500, stop_words="english")
X = vectorizer.fit_transform(code_snippets)

# Step 4: Train KMeans model on vectorized code
model = KMeans(n_clusters=5, random_state=42)
model.fit(X)

# Step 5: Save the model
os.makedirs("models", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = f"models/rocu_code_kmeans_{timestamp}.pkl"
joblib.dump(model, model_path)
print(f"Code clustering model saved at {model_path}")

