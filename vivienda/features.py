"""Variables del modelo y preprocesado."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config as C

BINARIAS = list(C.BINARIAS)
NUMERICAS = C.NUMERICAS + C.DISTANCIAS + C.COORDENADAS + ["log_metros", "metros_por_dormitorio", "antiguedad"]
CATEGORICAS = C.CATEGORICAS
COLUMNAS_ENTRADA = C.NUMERICAS + C.DISTANCIAS + C.COORDENADAS + BINARIAS + CATEGORICAS


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["log_metros"] = np.log(d["metros"])
    d["metros_por_dormitorio"] = d["metros"] / d["dormitorios"].clip(lower=1)
    d["antiguedad"] = 2018 - d["anio_construccion"]
    return d


def preprocesador(escalar: bool) -> ColumnTransformer:
    """Imputación + (escalado) de numéricas y one-hot de categóricas.

    Los barrios con menos de 20 viviendas se agrupan en una categoría «otros».
    """
    num = make_pipeline(SimpleImputer(strategy="median"), StandardScaler()) if escalar \
        else SimpleImputer(strategy="median")
    return ColumnTransformer(
        [("num", num, NUMERICAS),
         ("bin", "passthrough", BINARIAS),
         ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20, sparse_output=False),
          CATEGORICAS)],
        verbose_feature_names_out=False,
    )
