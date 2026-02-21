from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI()

#to allow comm between the frontend and the backend

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
    input_data = np.array([[
        data.CreditScore, data.Balance, data.EstimatedSalary,
        data.NumOfProducts, data.IsActiveMember, data.BalanceSalaryRatio,
        data.BalanceZero, data.ProductUsage, data.Male_Germany, data.Male_Spain,
        data.AgeGroup_26_35, data.AgeGroup_36_45, data.AgeGroup_46_55,
        data.AgeGroup_56_65, data.AgeGroup_66_75, data.AgeGroup_76_85,
        data.AgeGroup_86_95, data.TenureGroup_3_4, data.TenureGroup_5_6,
        data.TenureGroup_7_8, data.TenureGroup_9_10
    ]])
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

    