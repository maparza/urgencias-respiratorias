import pickle
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI(
    title="API Predicción Urgencias Respiratorias",
    version="1.0"
)


# Cargar modelo entrenado
with open("modelo.pkl", "rb") as archivo:
    modelo = pickle.load(archivo)

#datos para la API
class DatosPrediccion(BaseModel):
    SemanaEstadistica: int
    RegionGlosa: str
    lag_1: float
    lag_2: float
    lag_3: float

#endpoint
@app.post("/predict")
def predict(datos: DatosPrediccion):

    promedio_3_semanas = (
        datos.lag_1 +
        datos.lag_2 +
        datos.lag_3
    ) / 3

    entrada = pd.DataFrame([{
        "SemanaEstadistica": datos.SemanaEstadistica,
        "RegionGlosa": datos.RegionGlosa,
        "lag_1": datos.lag_1,
        "lag_2": datos.lag_2,
        "lag_3": datos.lag_3,
        "promedio_3_semanas": promedio_3_semanas
    }])

    prediccion = modelo.predict(entrada)[0]

    probabilidad = modelo.predict_proba(entrada)[0][1]

    return {
        "alta_demanda": bool(prediccion),
        "probabilidad": round(float(probabilidad), 4),
        "probabilidad_porcentaje": round(float(probabilidad) * 100, 2)
    }

@app.get("/")
def home():
    return {
        "mensaje": "API de Predicción de Urgencias Respiratorias",
        "documentacion": "/docs"
    }
