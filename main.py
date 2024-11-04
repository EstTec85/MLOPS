from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
from typing import List

# Cargar el modelo y las características seleccionadas
model_path = '/Users/estebanjimenez/Library/CloudStorage/OneDrive-Personal/Tec_Monterrey/Maestria_Aya/MLOPS/ML_OPS_WINE/wine_model.pkl'
features_path = '/Users/estebanjimenez/Library/CloudStorage/OneDrive-Personal/Tec_Monterrey/Maestria_Aya/MLOPS/ML_OPS_WINE/selected_features.pkl'

with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(features_path, "rb") as f:
    selected_features = pickle.load(f)

class WineData(BaseModel):
    features: List[float]

app = FastAPI()

@app.post("/predict")
def predict(wine_data: WineData):
    try:
        # Verificar si el número de características es correcto
        if len(wine_data.features) != len(selected_features):
            raise HTTPException(
                status_code=400,
                detail=f"Expected {len(selected_features)} features, but got {len(wine_data.features)}"
            )

        # Convertir a DataFrame solo con las características seleccionadas
        input_data = pd.DataFrame([wine_data.features], columns=selected_features)

        # Realizar predicción
        prediction = model.predict(input_data)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Wine classification model API"}
