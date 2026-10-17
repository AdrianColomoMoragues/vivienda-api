import json

import numpy as np
import pytest

from vivienda.datos import REGLAS, deduplicar, limpiar
from vivienda.features import COLUMNAS_ENTRADA, add_features
from vivienda.train import candidate_models, fuga_por_duplicados, make_fake, metricas


def test_train_guarda_modelo_y_metricas(models_dir):
    info = json.loads((models_dir / "model_info.json").read_text(encoding="utf-8"))
    assert (models_dir / "model.joblib").exists()
    assert info["modelo"] in {"ridge", "random_forest", "hist_gradient_boosting"}
    assert info["test"]["r2_log"] > 0.5            # el dataset sintético tiene señal clara
    assert 0.75 < info["intervalos"]["cobertura_test"] <= 1


def test_metricas_perfectas():
    y = np.log(np.array([1e5, 2e5, 3e5]))
    m = metricas(y, y)
    assert m["mae_eur"] == 0 and m["r2_log"] == 1


def test_add_features():
    d = add_features(make_fake(5))
    assert np.allclose(d["log_metros"], np.log(d["metros"]))
    assert (d["antiguedad"] == 2018 - d["anio_construccion"]).all()


def test_todas_las_columnas_existen():
    assert set(COLUMNAS_ENTRADA) <= set(make_fake(3).columns)


def _crudo():
    import pandas as pd
    return pd.DataFrame({
        "ASSETID": ["a", "a", "b", "c", "d"], "PERIOD": [201803, 201803, 201806, 201809, 201812],
        "PRICE": [100.0, 120.0, 90.0, 80.0, 70.0], "ROOMNUMBER": [2, 2, 15, 0, 3], "BATHNUMBER": [1, 1, 1, 1, 1],
        "ISSTUDIO": [0, 0, 0, 0, 0], "CADCONSTRUCTIONYEAR": [1990, 1990, 1990, 1990, 1990],
        "CONSTRUCTEDAREA": [80, 80, 80, 80, 80]})


def test_limpieza_quita_imposibles():
    d, inf = limpiar(_crudo())
    assert set(d["ASSETID"]) == {"a", "d"}            # b: 15 dormitorios, c: 0 dormitorios sin ser estudio
    assert inf["filas_tras_limpieza"] == 3
    assert set(REGLAS) <= set(inf)


def test_deduplicar_una_fila_por_vivienda_con_precio_mediano():
    d = deduplicar(limpiar(_crudo())[0]).set_index("ASSETID")
    assert len(d) == 2
    assert d.loc["a", "PRICE"] == pytest.approx(110)
    assert d.loc["a", "n_anuncios"] == 2


def test_fuga_particion_aleatoria_es_optimista():
    """Con anuncios duplicados, la partición aleatoria subestima el error."""
    base = make_fake(300, seed=1)
    dup = base.sample(300, replace=True, random_state=0)        # muchas viviendas repetidas
    modelo = candidate_models()["random_forest"]
    r = fuga_por_duplicados(modelo, dup, folds=3)
    assert r["aleatoria"]["rmse_log"] < r["por_vivienda"]["rmse_log"]
    assert r["edificios"] <= r["viviendas"]
