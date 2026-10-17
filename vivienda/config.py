"""Rutas, fuentes de datos y definición de variables."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATASET_PATH = DATA_DIR / "valencia_2018.parquet"
GEO_PATH = DATA_DIR / "valencia_geo.json"
MODELS_DIR = ROOT / "models"
RAW_DIR = DATA_DIR / "raw"

# Paquete de R idealista18 (licencia ODbL 1.0): anuncios de idealista de 2018.
FUENTE_URL = "https://github.com/paezha/idealista18"
RAW_BASE = "https://raw.githubusercontent.com/paezha/idealista18/master/data/"
RAW_FILES = ["Valencia_Sale.rda", "Valencia_POIS.rda", "Valencia_Polygons.rda"]

SEED = 42
TRIMESTRE_PREDICCION = 4          # la API valora a precios del 4.º trimestre de 2018

# Variables binarias (0/1) del anuncio, con su nombre en la API.
BINARIAS = {
    "ascensor": "HASLIFT",
    "terraza": "HASTERRACE",
    "aire_acondicionado": "HASAIRCONDITIONING",
    "parking": "HASPARKINGSPACE",
    "piscina": "HASSWIMMINGPOOL",
    "jardin": "HASGARDEN",
    "portero": "HASDOORMAN",
    "trastero": "HASBOXROOM",
    "armarios_empotrados": "HASWARDROBE",
    "atico": "ISINTOPFLOOR",
    "duplex": "ISDUPLEX",
    "estudio": "ISSTUDIO",
    "orientacion_norte": "HASNORTHORIENTATION",
    "orientacion_sur": "HASSOUTHORIENTATION",
    "orientacion_este": "HASEASTORIENTATION",
    "orientacion_oeste": "HASWESTORIENTATION",
}

# Columnas finales del dataset limpio (nombres en español).
NUMERICAS = ["metros", "dormitorios", "banos", "planta", "anio_construccion", "calidad_catastral",
             "plantas_edificio", "viviendas_edificio", "trimestre"]
CATEGORICAS = ["estado", "equipamiento", "exterior", "barrio"]
DISTANCIAS = ["dist_centro_km", "dist_metro_km", "dist_blasco_km"]
COORDENADAS = ["latitud", "longitud"]
OBJETIVO = "precio"

ESTADOS = {"nueva": "BUILTTYPEID_1", "a_reformar": "BUILTTYPEID_2", "buen_estado": "BUILTTYPEID_3"}
EQUIPAMIENTO = {1: "sin_cocina_ni_muebles", 2: "cocina_sin_muebles", 3: "cocina_y_muebles"}
