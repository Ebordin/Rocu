import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import os

# Ensure the data file exists
if not os.path.exists("data/train.csv"):
    print("data/train.csv not found. Creating example training data...")
    example_data = {
        'feature1': [0.1, 0.4, 0.5, 0.8, 0.3],
        'feature2': [1, 0, 1, 0, 1],
        'feature3': [5.2, 3.1, 4.8, 2.7, 3.3],
        'target': [0, 1, 0, 1, 0]
    }
    df = pd.DataFrame(example_data)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/train.csv", index=False)

# Load data
data = pd.read_csv("data/train.csv")
X = data.drop('target', axis=1)
y = data['target']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

# Define and train model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# Save the model
os.makedirs("models", exist_ok=True)
model.save("models/rocu_classifier.h5")
print("✅ Model saved successfully at models/rocu_classifier.h5")