from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import math
from datetime import datetime

import os
import json
import firebase_admin
from firebase_admin import credentials, firestore

# ----------------------------
# FIREBASE INIT
# ----------------------------
firebase_env = os.getenv("FIREBASE_KEY")

if not firebase_env:

    raise Exception("FIREBASE_KEY not set in environment variables")

firebase_key = json.loads(firebase_env)

cred = credentials.Certificate(firebase_key)
firebase_admin.initialize_app(cred)

db = firestore.client()

# ----------------------------
# FASTAPI APP
# ----------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("model.pkl")

# ----------------------------
# INPUT MODEL
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
# PREDICT ENDPOINT
# ----------------------------
@app.post("/predict")
def predict(data: SensorData):

    acc_magnitude = math.sqrt(
        data.acc_x**2 + data.acc_y**2 + data.acc_z**2
    )

    gyro_magnitude = math.sqrt(
        data.gyro_x**2 + data.gyro_y**2 + data.gyro_z**2
    )

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

    prediction = model.predict(df)[0]
    result = "pothole" if prediction == 1 else "no pothole"

    doc = {
        "latitude": data.latitude,
        "longitude": data.longitude,
        "prediction": result,
        "created_at": datetime.utcnow()
    }

    if result == "pothole":
        db.collection("potholes").add(doc)

    return {
        "prediction": result,
        "latitude": data.latitude,
        "longitude": data.longitude
    }