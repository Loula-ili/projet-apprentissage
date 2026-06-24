# ---------------------------------------------------------------------------
# Q9 : API avec FastAPI
# ---------------------------------------------------------------------------
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# Charger le pipeline
with open("credit_scoring_pipeline.pkl", "rb") as f:
    pipe = pickle.load(f)

# Créer l'application FastAPI
app = FastAPI(title="API Credit Scoring")

# Définir le format des données en entrée (les 13 features)
class Client(BaseModel):
    Seniority: float
    Home: float
    Time: float
    Age: float
    Marital: float
    Records: float
    Job: float
    Expenses: float
    Income: float
    Assets: float
    Debt: float
    Amount: float
    Price: float

@app.get("/")
def root():
    return {"message": "API Credit Scoring - utilisez /predict/ pour faire une prédiction"}

@app.post("/predict/")
def predict(client: Client):
    # Convertir en array numpy (ordre des colonnes du dataset original)
    X = np.array([[
        client.Seniority,
        client.Home,
        client.Time,
        client.Age,
        client.Marital,
        client.Records,
        client.Job,
        client.Expenses,
        client.Income,
        client.Assets,
        client.Debt,
        client.Amount,
        client.Price
    ]])
    
    # Prédiction
    prediction = pipe.predict(X)[0]
    
    # Probabilités (KNN supporte predict_proba)
    probas = pipe.predict_proba(X)[0]
    
    return {
        "prediction": int(prediction),
        "probabilite_classe_0": round(float(probas[0]), 4),
        "probabilite_classe_1": round(float(probas[1]), 4)
    }