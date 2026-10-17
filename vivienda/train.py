"""Entrenamiento y evaluación del modelo de precios.

    python -m vivienda.train          # compara modelos, evalúa y guarda models/
"""
from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold, cross_val_predict
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer

from . import config as C
from .datos import cargar
from .features import COLUMNAS_ENTRADA, add_features, preprocesador

ALFA = 0.10                                   # intervalos del 90 %


def _fe():
    return FunctionTransformer(add_features)


def candidate_models() -> dict[str, Pipeline]:
    return {
        "ridge": make_pipeline(_fe(), preprocesador(escalar=True), RidgeCV(alphas=np.logspace(-2, 3, 12))),
        "random_forest": make_pipeline(_fe(), preprocesador(escalar=False), RandomForestRegressor(
            n_estimators=200, min_samples_leaf=2, max_features=0.4, n_jobs=-1, random_state=C.SEED)),
        "hist_gradient_boosting": make_pipeline(_fe(), preprocesador(escalar=False), HistGradientBoostingRegressor(
            max_iter=800, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=1.0,
            early_stopping=False, random_state=C.SEED)),
    }


def modelo_cuantil(q: float) -> Pipeline:
    return make_pipeline(_fe(), preprocesador(escalar=False), HistGradientBoostingRegressor(
        loss="quantile", quantile=q, max_iter=500, learning_rate=0.05, max_leaf_nodes=31,
        min_samples_leaf=40, random_state=C.SEED))


def dividir(X, y, grupos, test_size: float, seed: int = C.SEED):
    """Partición train/test sin repartir un mismo edificio entre las dos partes."""
    tr, te = next(GroupShuffleSplit(1, test_size=test_size, random_state=seed).split(X, y, grupos))
    return X.iloc[tr], X.iloc[te], y[tr], y[te], np.asarray(grupos)[tr], np.asarray(grupos)[te]


def cv_grupos(folds: int) -> GroupKFold:
    return GroupKFold(folds, shuffle=True, random_state=C.SEED)


def metricas(y_log: np.ndarray, pred_log: np.ndarray) -> dict:
    """Métricas en euros (más interpretables) y en escala logarítmica."""
    y, p = np.exp(y_log), np.exp(pred_log)
    err = np.abs(p - y)
    return {
        "mae_eur": float(err.mean()),
        "mediana_error_eur": float(np.median(err)),
        "mape_pct": float(100 * (err / y).mean()),
        "mdape_pct": float(100 * np.median(err / y)),
        "rmse_log": float(np.sqrt(np.mean((pred_log - y_log) ** 2))),
        "r2_log": float(1 - np.sum((pred_log - y_log) ** 2) / np.sum((y_log - y_log.mean()) ** 2)),
    }


def fuga_por_duplicados(modelo: Pipeline, anuncios: pd.DataFrame, folds: int = 5) -> dict:
    """Error de validación cruzada sobre los anuncios con tres particiones cada vez más estrictas.

    - aleatoria: un mismo piso anunciado dos veces puede caer en train y en test;
    - por vivienda: cada vivienda (ASSETID) en un solo lado;
    - por edificio: cada edificio en un solo lado (lo que pasa al tasar un piso nuevo).
    """
    X, y = anuncios[COLUMNAS_ENTRADA], np.log(anuncios[C.OBJETIVO].to_numpy())
    esquemas = {
        "aleatoria": (KFold(folds, shuffle=True, random_state=C.SEED), None),
        "por_vivienda": (cv_grupos(folds), anuncios["id_vivienda"]),
        "por_edificio": (cv_grupos(folds), anuncios["edificio"]),
    }
    out = {n: metricas(y, cross_val_predict(modelo, X, y, cv=cv, groups=g, n_jobs=1))
           for n, (cv, g) in esquemas.items()}
    return {**out, "anuncios": len(anuncios), "viviendas": int(anuncios["id_vivienda"].nunique()),
            "edificios": int(anuncios["edificio"].nunique())}


def train(df: pd.DataFrame, out_dir=C.MODELS_DIR, cv_folds: int = 5, anuncios: pd.DataFrame | None = None,
          importancia: bool = True) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    X, y = df[COLUMNAS_ENTRADA], np.log(df[C.OBJETIVO].to_numpy())
    X_tr, X_te, y_tr, y_te, g_tr, _ = dividir(X, y, df["edificio"], test_size=0.2)

    # 1. Comparación por validación cruzada (agrupada por edificio) en el conjunto de entrenamiento
    cv = cv_grupos(cv_folds)
    resultados = {}
    for nombre, modelo in candidate_models().items():
        t0 = time.time()
        pred = cross_val_predict(modelo, X_tr, y_tr, cv=cv, groups=g_tr, n_jobs=1)
        resultados[nombre] = {**metricas(y_tr, pred), "segundos": round(time.time() - t0, 1)}
        print(f"{nombre:>24}: MdAPE {resultados[nombre]['mdape_pct']:.1f} %  "
              f"R² log {resultados[nombre]['r2_log']:.3f}  ({resultados[nombre]['segundos']} s)")
    mejor = min(resultados, key=lambda k: resultados[k]["rmse_log"])

    # 2. Evaluación del mejor en test
    modelo = candidate_models()[mejor].fit(X_tr, y_tr)
    pred_te = modelo.predict(X_te)
    test = metricas(y_te, pred_te)

    # 3. Intervalos del 90 %: regresión cuantílica + calibración conformal (CQR)
    #    Se ajustan los cuantiles en una parte del entrenamiento y se calibran en la otra.
    X_fit, X_cal, y_fit, y_cal, _, _ = dividir(X_tr, y_tr, g_tr, test_size=0.25)
    q_inf = modelo_cuantil(ALFA / 2).fit(X_fit, y_fit)
    q_sup = modelo_cuantil(1 - ALFA / 2).fit(X_fit, y_fit)
    lo_cal, hi_cal = q_inf.predict(X_cal), q_sup.predict(X_cal)
    cobertura_sin_calibrar = float(np.mean((y_cal >= lo_cal) & (y_cal <= hi_cal)))
    ajuste = _cuantil_conformal(np.maximum(lo_cal - y_cal, y_cal - hi_cal))
    lo_te, hi_te = q_inf.predict(X_te) - ajuste, q_sup.predict(X_te) + ajuste
    dentro = (y_te >= lo_te) & (y_te <= hi_te)
    intervalos = {
        "nivel": 1 - ALFA,
        "metodo": "HGB cuantílico (5 % y 95 %) + calibración conformal (CQR)",
        "ajuste_log": ajuste,
        "cobertura_calibracion_sin_ajuste": cobertura_sin_calibrar,
        "cobertura_test": float(dentro.mean()),
        "anchura_mediana_pct": float(100 * np.median((np.exp(hi_te) - np.exp(lo_te)) / np.exp(pred_te))),
    }
    precio_te = np.exp(y_te)
    tramos = pd.qcut(precio_te, 4, labels=["Q1 (más baratos)", "Q2", "Q3", "Q4 (más caros)"])
    intervalos["cobertura_por_tramo"] = {str(k): float(v) for k, v in
                                         pd.Series(dentro).groupby(np.asarray(tramos)).mean().items()}

    # 4. Importancia por permutación en test (sobre las variables de entrada)
    imp = {}
    if importancia:
        r = permutation_importance(modelo, X_te, y_te, n_repeats=5, random_state=C.SEED,
                                   scoring="neg_root_mean_squared_error", n_jobs=1)
        imp = dict(sorted(zip(COLUMNAS_ENTRADA, r.importances_mean.astype(float)), key=lambda kv: -kv[1]))

    # 5. Fuga de información por duplicados (con los anuncios sin deduplicar)
    fuga = fuga_por_duplicados(candidate_models()[mejor], anuncios) if anuncios is not None else None

    # 6. Modelos finales con todos los datos
    ajuste_final = ajustar_final(df, out_dir, mejor)

    info = {
        "modelo": mejor,
        "objetivo": "log(precio)",
        "n_viviendas": len(df),
        "n_edificios": int(df["edificio"].nunique()),
        "n_train": len(X_tr), "n_test": len(X_te),
        "validacion": "partición por edificio (ningún edificio aparece en train y test a la vez)",
        "cv": resultados,
        "test": test,
        "intervalos": {**intervalos, "ajuste_log_final": ajuste_final},
        "importancia_permutacion": imp,
        "fuga_duplicados": fuga,
        "fuente": "idealista18 (ODbL 1.0), anuncios de venta de València en 2018",
        "barrios": _resumen_barrios(df),
    }
    (out_dir / "model_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return info


def _cuantil_conformal(scores: np.ndarray) -> float:
    n = len(scores)
    return float(np.quantile(scores, min(1.0, np.ceil((n + 1) * (1 - ALFA)) / n)))


def ajustar_final(df: pd.DataFrame, out_dir, nombre: str) -> float:
    """Ajusta con todos los datos el modelo elegido y los de cuantiles (calibrados) y los guarda."""
    X, y = df[COLUMNAS_ENTRADA], np.log(df[C.OBJETIVO].to_numpy())
    final = candidate_models()[nombre].fit(X, y)
    X_fit, X_cal, y_fit, y_cal, _, _ = dividir(X, y, df["edificio"], test_size=0.25)
    qi, qs = modelo_cuantil(ALFA / 2).fit(X_fit, y_fit), modelo_cuantil(1 - ALFA / 2).fit(X_fit, y_fit)
    ajuste = _cuantil_conformal(np.maximum(qi.predict(X_cal) - y_cal, y_cal - qs.predict(X_cal)))
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"modelo": final, "q_inf": qi, "q_sup": qs, "ajuste_log": ajuste,
                 "columnas": COLUMNAS_ENTRADA}, out_dir / "model.joblib", compress=3)
    return ajuste


def _resumen_barrios(df: pd.DataFrame) -> dict:
    """Mediana de €/m² y número de viviendas de cada barrio (para dar contexto a la predicción)."""
    g = (df[C.OBJETIVO] / df["metros"]).groupby(df["barrio"])
    return {b: {"euros_m2_mediana": round(float(m), 0), "viviendas": int(n)}
            for b, m, n in zip(g.median().index, g.median().to_numpy(), g.size().to_numpy())}


def make_fake(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Datos sintéticos con el mismo esquema (para tests rápidos)."""
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({
        "metros": rng.uniform(40, 200, n), "dormitorios": rng.integers(1, 5, n), "banos": rng.integers(1, 3, n),
        "planta": rng.integers(0, 10, n).astype(float), "anio_construccion": rng.integers(1900, 2018, n).astype(float),
        "calidad_catastral": rng.integers(0, 9, n).astype(float),
        "plantas_edificio": rng.integers(2, 12, n).astype(float),
        "viviendas_edificio": rng.integers(4, 80, n).astype(float), "trimestre": rng.integers(1, 5, n),
        "dist_centro_km": rng.uniform(0, 6, n), "dist_metro_km": rng.uniform(0, 2, n),
        "dist_blasco_km": rng.uniform(0, 6, n),
        "latitud": rng.uniform(39.44, 39.50, n), "longitud": rng.uniform(-0.41, -0.33, n),
        "estado": rng.choice(list(C.ESTADOS), n), "equipamiento": rng.choice(list(C.EQUIPAMIENTO.values()), n),
        "exterior": rng.choice(["exterior", "interior"], n), "barrio": rng.choice(["Russafa", "Benimaclet"], n),
    })
    for b in C.BINARIAS:
        d[b] = rng.integers(0, 2, n)
    d["precio"] = np.exp(np.log(1800 * d["metros"]) - 0.08 * d["dist_centro_km"] + rng.normal(0, 0.2, n))
    d["id_vivienda"] = [f"F{i}" for i in range(n)]
    d["edificio"] = np.arange(n) // 2
    return d


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--fake", action="store_true", help="entrena con datos sintéticos (tests/CI)")
    ap.add_argument("--final", action="store_true",
                    help="solo ajusta los modelos finales con el modelo elegido en models/model_info.json")
    a = ap.parse_args()
    if a.final:
        info = json.loads((C.MODELS_DIR / "model_info.json").read_text(encoding="utf-8"))
        ajustar_final(cargar(), C.MODELS_DIR, info["modelo"])
        print(f"Modelo final ({info['modelo']}) guardado en {C.MODELS_DIR / 'model.joblib'}")
        raise SystemExit
    if a.fake:
        info = train(make_fake(), cv_folds=3, importancia=False)
    else:
        info = train(cargar(), anuncios=cargar(anuncios=True))
    print(json.dumps({k: info[k] for k in ["modelo", "test", "intervalos", "fuga_duplicados"]},
                     ensure_ascii=False, indent=2))
