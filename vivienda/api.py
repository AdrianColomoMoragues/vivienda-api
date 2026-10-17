"""API REST para tasar pisos en València.

Arranque:  uvicorn vivienda.api:app --reload
Docs:      http://localhost:8000/docs
"""
from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from enum import Enum
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, model_validator

from . import __version__
from . import config as C
from .geo import cargar_geo

MODELS_DIR = Path(os.getenv("MODELS_DIR", C.MODELS_DIR))
STATIC_DIR = C.ROOT / "static"
MAX_LOTE = 1000

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    ruta = MODELS_DIR / "model.joblib"
    if not ruta.exists():
        raise RuntimeError(f"No existe {ruta}. Ejecuta antes: python -m vivienda.train")
    state["bundle"] = joblib.load(ruta)
    state["info"] = json.loads((MODELS_DIR / "model_info.json").read_text(encoding="utf-8"))
    state["geo"] = cargar_geo()
    state["geojson"] = _geojson_barrios(state["geo"], state["info"].get("barrios", {}))
    yield
    state.clear()


app = FastAPI(
    title="Tasador de pisos en València",
    version=__version__,
    description="Modelo de gradient boosting entrenado con 27 000 anuncios de venta de idealista (2018, "
                "paquete idealista18, ODbL 1.0). Devuelve el precio estimado y un intervalo del 90 % "
                "calibrado.",
    lifespan=lifespan,
)


Estado = Enum("Estado", {k: k for k in C.ESTADOS}, type=str)
Equipamiento = Enum("Equipamiento", {v: v for v in C.EQUIPAMIENTO.values()}, type=str)


class Exterior(str, Enum):
    exterior = "exterior"
    interior = "interior"
    desconocido = "desconocido"


class Piso(BaseModel):
    """Características de un piso. La ubicación basta para calcular barrio y distancias."""
    latitud: float = Field(..., ge=39.40, le=39.55, examples=[39.4699])
    longitud: float = Field(..., ge=-0.45, le=-0.28, examples=[-0.3763])
    metros: float = Field(..., ge=25, le=500, description="Superficie construida (m²)")
    dormitorios: int = Field(..., ge=0, le=10)
    banos: int = Field(..., ge=1, le=6)
    planta: float | None = Field(None, ge=-1, le=30, description="0 = bajo; vacío si se desconoce")
    anio_construccion: float | None = Field(None, ge=1800, le=2018)
    calidad_catastral: float | None = Field(None, ge=0, le=9, description="0 (mejor) … 9 (peor), según catastro")
    plantas_edificio: float | None = Field(None, ge=1, le=40)
    viviendas_edificio: float | None = Field(None, ge=1, le=1000)
    estado: Estado = Estado.buen_estado
    equipamiento: Equipamiento = Equipamiento.cocina_y_muebles
    exterior: Exterior = Exterior.exterior
    ascensor: bool = False
    terraza: bool = False
    aire_acondicionado: bool = False
    parking: bool = False
    piscina: bool = False
    jardin: bool = False
    portero: bool = False
    trastero: bool = False
    armarios_empotrados: bool = False
    atico: bool = False
    duplex: bool = False
    estudio: bool = False
    orientacion_norte: bool = False
    orientacion_sur: bool = False
    orientacion_este: bool = False
    orientacion_oeste: bool = False

    @model_validator(mode="after")
    def coherencia(self):
        if self.dormitorios == 0 and not self.estudio:
            raise ValueError("0 dormitorios solo es posible si es un estudio")
        if self.planta is not None and self.plantas_edificio is not None and self.planta > self.plantas_edificio:
            raise ValueError("la planta no puede ser mayor que el número de plantas del edificio")
        return self

    model_config = {"json_schema_extra": {"examples": [{
        "latitud": 39.4699, "longitud": -0.3763, "metros": 95, "dormitorios": 3, "banos": 2, "planta": 3,
        "anio_construccion": 1975, "estado": "buen_estado", "exterior": "exterior", "ascensor": True,
        "aire_acondicionado": True}]}}


class Prediccion(BaseModel):
    precio_eur: float = Field(description="Precio estimado (euros de finales de 2018)")
    intervalo_90_eur: tuple[float, float] = Field(description="Intervalo del 90 % (calibrado)")
    euros_m2: float
    barrio: str
    euros_m2_barrio: float | None = Field(description="Mediana de €/m² de los anuncios del barrio")
    dist_centro_km: float
    dist_metro_km: float
    modelo: str


def _geojson_barrios(geo, resumen: dict) -> dict:
    feats = []
    for nombre, anillos in geo.barrios:
        r = resumen.get(nombre, {})
        feats.append({"type": "Feature",
                      "properties": {"barrio": nombre, "euros_m2": r.get("euros_m2_mediana"),
                                     "viviendas": r.get("viviendas", 0)},
                      "geometry": {"type": "Polygon", "coordinates": [a.round(5).tolist() for a in anillos]}})
    return {"type": "FeatureCollection", "features": feats}


def _tabla(pisos: list[Piso]) -> pd.DataFrame:
    geo = state["geo"]
    d = pd.DataFrame([p.model_dump(mode="json") for p in pisos])
    barrios = [geo.barrio(la, lo) for la, lo in zip(d["latitud"], d["longitud"])]
    fuera = [i for i, b in enumerate(barrios) if b is None]
    if fuera:
        raise HTTPException(422, f"Ubicación fuera del término municipal de València (pisos {fuera}).")
    d["barrio"] = barrios
    for k, v in geo.distancias(d["latitud"].to_numpy(), d["longitud"].to_numpy()).items():
        d[k] = v
    d["trimestre"] = C.TRIMESTRE_PREDICCION
    for b in C.BINARIAS:
        d[b] = d[b].astype(int)
    for c in ["planta", "anio_construccion", "calidad_catastral", "plantas_edificio", "viviendas_edificio"]:
        d[c] = pd.to_numeric(d[c], errors="coerce").astype(float)
    return d


def _predecir(pisos: list[Piso]) -> list[Prediccion]:
    b, info = state["bundle"], state["info"]
    d = _tabla(pisos)
    X = d[b["columnas"]]
    p = b["modelo"].predict(X)
    lo = b["q_inf"].predict(X) - b["ajuste_log"]
    hi = b["q_sup"].predict(X) + b["ajuste_log"]
    lo, hi = np.minimum(lo, p), np.maximum(hi, p)        # el intervalo siempre contiene la estimación
    out = []
    for i, fila in d.iterrows():
        precio = float(np.exp(p[i]))
        ref = info.get("barrios", {}).get(fila["barrio"], {})
        out.append(Prediccion(
            precio_eur=round(precio, -3),
            intervalo_90_eur=(round(float(np.exp(lo[i])), -3), round(float(np.exp(hi[i])), -3)),
            euros_m2=round(precio / fila["metros"]),
            barrio=fila["barrio"],
            euros_m2_barrio=ref.get("euros_m2_mediana"),
            dist_centro_km=round(float(fila["dist_centro_km"]), 2),
            dist_metro_km=round(float(fila["dist_metro_km"]), 2),
            modelo=info["modelo"],
        ))
    return out


@app.get("/", include_in_schema=False)
def inicio():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "bundle" in state}


@app.get("/model-info")
def model_info():
    """Métricas de validación, cobertura de los intervalos e importancia de variables."""
    return state["info"]


@app.get("/barrio")
def barrio(latitud: float, longitud: float):
    """Barrio y distancias de un punto (lo usa el mapa de la web)."""
    geo = state["geo"]
    nombre = geo.barrio(latitud, longitud)
    if nombre is None:
        raise HTTPException(404, "Ubicación fuera del término municipal de València.")
    dist = {k: round(float(v[0]), 2) for k, v in geo.distancias(latitud, longitud).items()}
    return {"barrio": nombre, **dist, **state["info"].get("barrios", {}).get(nombre, {})}


@app.get("/barrios.geojson")
def barrios_geojson():
    """Polígonos de los barrios con la mediana de €/m² de sus anuncios (mapa de la web)."""
    return state["geojson"]


@app.post("/predict", response_model=Prediccion)
def predict(piso: Piso):
    return _predecir([piso])[0]


@app.post("/predict/batch", response_model=list[Prediccion])
def predict_batch(pisos: list[Piso]):
    if not pisos:
        raise HTTPException(422, "La lista está vacía")
    if len(pisos) > MAX_LOTE:
        raise HTTPException(413, f"Máximo {MAX_LOTE} pisos por petición")
    return _predecir(pisos)
