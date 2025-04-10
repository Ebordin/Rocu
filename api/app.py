from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import numpy as np

app = FastAPI()
model = tf.keras.models.load_model("models/rocu_classifier.h5")

class InputData(BaseModel):
    feature1: float
    feature2: float
    feature3: float

@app.post("/predict")
def predict(data: InputData):
    X = np.array([[data.feature1, data.feature2, data.feature3]])
    prediction = model.predict(X)[0][0]
    return {"prediction": float(prediction)}
