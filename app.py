from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import math

app = FastAPI()

# Allow mobile app / frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained model
model = joblib.load("model.pkl")

# ----------------------------
# INPUT FROM MOBILE APP
# ----------------------------
class SensorData(BaseModel):
    latitude: float
    longitude: float
    speed: float

    acc_x: float
    acc_y: float
    acc_z: float

    gyro_x: float
    gyro_y: float
    gyro_z: float


@app.get("/health")
def health():
    return {"status": "running"}

# ----------------------------
# PREDICTION ENDPOINT
# ----------------------------
@app.post("/predict")
def predict(data: SensorData):

    # ----------------------------
    # FEATURE ENGINEERING
    # ----------------------------
    acc_magnitude = math.sqrt(
        data.acc_x**2 + data.acc_y**2 + data.acc_z**2
    )

    gyro_magnitude = math.sqrt(
        data.gyro_x**2 + data.gyro_y**2 + data.gyro_z**2
    )

    # ----------------------------
    # CREATE MODEL INPUT
    # ----------------------------
    df = pd.DataFrame([[
        acc_magnitude,
        gyro_magnitude,
        data.speed,
        data.latitude,
        data.longitude
    ]], columns=[
        "acc_magnitude",
        "gyro_magnitude",
        "speed",
        "latitude",
        "longitude"
    ])

    # ----------------------------
    # PREDICT
    # ----------------------------
    prediction = model.predict(df)[0]

    # ----------------------------
    # OUTPUT LABEL
    # ----------------------------
    result = "pothole" if prediction == 1 else "no pothole"

    return {
        "prediction": result,
        "latitude": data.latitude,
        "longitude": data.longitude
    }