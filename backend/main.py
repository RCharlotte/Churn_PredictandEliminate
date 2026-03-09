from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd

app = FastAPI()

#to allow comm between the frontend and the backend. something about the security too

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_methods=["*"],
    allow_headers=["*"],

)

with open("/Users/charlotteawino/Code/Churnprediction/model/model.pkl","rb") as file:
    model=pickle.load(file) 
with open("/Users/charlotteawino/Code/Churnprediction/model/scaler.pkl","rb") as file:
    scaler=pickle.load(file)


class CustomerData(BaseModel):
    CreditScore: float
    Balance: float
    EstimatedSalary: float
    NumOfProducts: float
    IsActiveMember: float
    BalanceSalaryRatio: float
    BalanceZero: float
    ProductUsage: float
    Male_Germany: float
    Male_Spain: float
    AgeGroup_26_35: float
    AgeGroup_36_45: float
    AgeGroup_46_55: float
    AgeGroup_56_65: float
    AgeGroup_66_75: float
    AgeGroup_76_85: float
    AgeGroup_86_95: float
    TenureGroup_3_4: float
    TenureGroup_5_6: float
    TenureGroup_7_8: float
    TenureGroup_9_10: float 


@app.post("/predict")
def predict(data: CustomerData):
    input_data = pd.DataFrame([{
        'CreditScore': float(data.CreditScore),
        'Balance': float(data.Balance),
        'EstimatedSalary': float(data.EstimatedSalary),
        'NumOfProducts': float(data.NumOfProducts),
        'IsActiveMember': float(data.IsActiveMember),
        'BalanceSalaryRatio': float(data.BalanceSalaryRatio),
        'BalanceZero': float(data.BalanceZero),
        'ProductUsage': float(data.ProductUsage),
        'Male_Germany': float(data.Male_Germany),
        'Male_Spain': float(data.Male_Spain),
        'AgeGroup_26-35': float(data.AgeGroup_26_35),
        'AgeGroup_36-45': float(data.AgeGroup_36_45),
        'AgeGroup_46-55': float(data.AgeGroup_46_55),
        'AgeGroup_56-65': float(data.AgeGroup_56_65),
        'AgeGroup_66-75': float(data.AgeGroup_66_75),
        'AgeGroup_76-85': float(data.AgeGroup_76_85),
        'AgeGroup_86-95': float(data.AgeGroup_86_95),
        'TenureGroup_3-4': float(data.TenureGroup_3_4),
        'TenureGroup_5-6': float(data.TenureGroup_5_6),
        'TenureGroup_7-8': float(data.TenureGroup_7_8),
        'TenureGroup_9-10': float(data.TenureGroup_9_10)
    }])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]
    return {
        "churn": bool(prediction),
        "probability": round(float(probability), 2)
    }

@app.get("/")
def root():
    return {"message": "Churn Prediction API is running"}

    