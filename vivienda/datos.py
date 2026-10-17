"""Descarga, limpieza y preparación de los anuncios de venta de València (idealista, 2018).

Fuente: paquete de R `idealista18` (Rey-Blanco, Arbués, López y Páez), licencia ODbL 1.0.
https://github.com/paezha/idealista18

    python -m vivienda.datos          # descarga los .rda y genera data/valencia_2018.parquet y data/valencia_geo.json
"""
from __future__ import annotations

import json
import urllib.request
import warnings

import numpy as np
import pandas as pd

from . import config as C
from .geo import Geo

# --------------------------------------------------------------------------- #
# Descarga y lectura
# --------------------------------------------------------------------------- #

def descargar(destino=C.RAW_DIR) -> None:
    destino.mkdir(parents=True, exist_ok=True)
    for nombre in C.RAW_FILES:
        ruta = destino / nombre
        if not ruta.exists():
            print(f"Descargando {nombre}…")
            urllib.request.urlretrieve(C.RAW_BASE + nombre, ruta)


def _leer_rda(ruta, objeto: str, **kw):
    import rdata  # solo hace falta para regenerar los datos

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return rdata.conversion.convert(rdata.parser.parse_file(ruta), **kw)[objeto]


def leer_crudos(origen=C.RAW_DIR) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    venta = _leer_rda(origen / "Valencia_Sale.rda", "Valencia_Sale").drop(columns="geometry")
    pois = _leer_rda(origen / "Valencia_POIS.rda", "Valencia_POIS")
    poligonos = _leer_rda(origen / "Valencia_Polygons.rda", "Valencia_Polygons", default_encoding="utf8")
    return pd.DataFrame(venta), pois, poligonos


# --------------------------------------------------------------------------- #
# Geometría
# --------------------------------------------------------------------------- #

def _anillos(geom) -> list[list[list[float]]]:
    """Convierte la geometría (polígono o multipolígono de sf) en una lista de anillos [[lon, lat], ...]."""
    if isinstance(geom, np.ndarray) and geom.ndim == 2:
        return [np.round(geom, 6).tolist()]
    anillos = []
    for parte in geom:
        anillos.extend(_anillos(parte))
    return anillos


def construir_geo(pois: dict, poligonos: pd.DataFrame) -> dict:
    def puntos(df):
        return np.round(df[["Lat", "Lon"]].to_numpy(float), 6).tolist()

    barrios = []
    for _, fila in poligonos.iterrows():
        anillos = _anillos(fila["geometry"])
        # el anillo con más área es el exterior: se pone primero
        anillos.sort(key=lambda r: -abs(_area(np.array(r))))
        barrios.append({"nombre": str(fila["LOCATIONNAME"]), "anillos": anillos})
    return {
        "fuente": f"idealista18 ({C.FUENTE_URL}), ODbL 1.0",
        "centro": puntos(pois["City_Center"])[0],
        "metro": puntos(pois["Metro"]),
        "blasco": puntos(pois["Blasco"]),
        "barrios": barrios,
    }


def _area(anillo: np.ndarray) -> float:
    x, y = anillo[:, 0], anillo[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


# --------------------------------------------------------------------------- #
# Limpieza
# --------------------------------------------------------------------------- #

REGLAS = {
    "más de 10 dormitorios": lambda d: d["ROOMNUMBER"] > 10,
    "más de 6 baños": lambda d: d["BATHNUMBER"] > 6,
    "0 dormitorios sin ser estudio": lambda d: (d["ROOMNUMBER"] == 0) & (d["ISSTUDIO"] == 0),
    "0 baños": lambda d: d["BATHNUMBER"] == 0,
    "año de catastro anterior a 1800": lambda d: d["CADCONSTRUCTIONYEAR"] < 1800,
    "superficie fuera de 25–500 m²": lambda d: ~d["CONSTRUCTEDAREA"].between(25, 500),
}


def limpiar(venta: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Quita registros imposibles. Devuelve los datos y cuántas filas elimina cada regla."""
    d = venta.copy()
    informe = {"filas_iniciales": len(d)}
    quitar = pd.Series(False, index=d.index)
    for nombre, regla in REGLAS.items():
        m = regla(d).fillna(False).astype(bool)
        informe[nombre] = int((m & ~quitar).sum())
        quitar |= m
    d = d[~quitar]
    informe["filas_tras_limpieza"] = len(d)
    return d, informe


def deduplicar(d: pd.DataFrame) -> pd.DataFrame:
    """Una fila por vivienda (ASSETID).

    La misma vivienda aparece en varios anuncios (otra agencia, rebajas…), siempre en el
    mismo trimestre. Se conserva el anuncio más reciente y se le asigna la mediana de los
    precios publicados para esa vivienda.
    """
    d = d.sort_values(["ASSETID", "PERIOD"])
    precio = d.groupby("ASSETID")["PRICE"].median()
    n = d.groupby("ASSETID").size()
    u = d.drop_duplicates("ASSETID", keep="last").set_index("ASSETID")
    u["PRICE"] = precio
    u["n_anuncios"] = n
    return u.reset_index()


def a_espanol(d: pd.DataFrame, geo: Geo) -> pd.DataFrame:
    """Selecciona y renombra variables; asigna barrio y recalcula distancias desde lat/lon."""
    out = pd.DataFrame({
        "id_vivienda": d["ASSETID"].astype(str).to_numpy(),
        "trimestre": (d["PERIOD"].astype(int) % 100 // 3).to_numpy(),
        "precio": d["PRICE"].astype(float).to_numpy(),
        "metros": d["CONSTRUCTEDAREA"].astype(float).to_numpy(),
        "dormitorios": d["ROOMNUMBER"].astype(int).to_numpy(),
        "banos": d["BATHNUMBER"].astype(int).to_numpy(),
        "planta": d["FLOORCLEAN"].astype(float).to_numpy(),
        "anio_construccion": d["CADCONSTRUCTIONYEAR"].astype(float).to_numpy(),
        "calidad_catastral": d["CADASTRALQUALITYID"].astype(float).to_numpy(),
        "plantas_edificio": d["CADMAXBUILDINGFLOOR"].astype(float).to_numpy(),
        "viviendas_edificio": d["CADDWELLINGCOUNT"].astype(float).to_numpy(),
        "latitud": d["LATITUDE"].astype(float).to_numpy(),
        "longitud": d["LONGITUDE"].astype(float).to_numpy(),
    })
    for api, col in C.BINARIAS.items():
        out[api] = d[col].astype(int).to_numpy()
    estado = np.select([d[c].astype(int).to_numpy() == 1 for c in C.ESTADOS.values()], list(C.ESTADOS), "desconocido")
    out["estado"] = estado
    out["equipamiento"] = d["AMENITYID"].astype(int).map(C.EQUIPAMIENTO).to_numpy()
    out["exterior"] = d["FLATLOCATIONID"].map({1: "exterior", 2: "interior"}).fillna("desconocido").to_numpy()
    for k, v in geo.distancias(out["latitud"].to_numpy(), out["longitud"].to_numpy()).items():
        out[k] = np.round(v, 4)
    b = geo.barrios_de(out["latitud"].to_numpy(), out["longitud"].to_numpy())
    out["barrio"] = ["fuera" if x is None else str(x) for x in b]
    out["n_anuncios"] = d["n_anuncios"].to_numpy() if "n_anuncios" in d else 1
    return out


EDIFICIO_ATRIBUTOS = ["anio_construccion", "plantas_edificio", "viviendas_edificio", "calidad_catastral"]
EDIFICIO_RADIO_KM = 0.15


def asignar_edificio(d: pd.DataFrame, radio_km: float = EDIFICIO_RADIO_KM) -> pd.Series:
    """Identificador aproximado del edificio de cada vivienda.

    Las coordenadas de idealista18 llevan ruido de anonimización (decenas de metros), así que
    no sirven para identificar el edificio. Pero los datos de catastro describen el edificio
    (año, plantas, número de viviendas, calidad): dos viviendas con los mismos valores a menos
    de `radio_km` se consideran del mismo edificio (agrupamiento de enlace simple).
    """
    from scipy.cluster.hierarchy import fcluster, linkage

    from .geo import haversine_km

    ids = np.empty(len(d), dtype=np.int64)
    siguiente = 0
    claves = d[EDIFICIO_ATRIBUTOS].astype(float).fillna(-1).round(0)
    for _, idx in claves.groupby(EDIFICIO_ATRIBUTOS, sort=False).indices.items():
        if len(idx) == 1:
            ids[idx] = siguiente
            siguiente += 1
            continue
        la, lo = d["latitud"].to_numpy()[idx], d["longitud"].to_numpy()[idx]
        dist = haversine_km(la[:, None], lo[:, None], la[None, :], lo[None, :])
        cond = dist[np.triu_indices(len(idx), 1)]
        grupos = fcluster(linkage(cond, method="single"), t=radio_km, criterion="distance")
        ids[idx] = siguiente + grupos - 1
        siguiente += grupos.max()
    return pd.Series(ids, index=d.index, name="edificio")


def preparar(guardar: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Devuelve (viviendas únicas, anuncios sin deduplicar, informe)."""
    descargar()
    venta, pois, poligonos = leer_crudos()
    geo_dict = construir_geo(pois, poligonos)
    geo = Geo(geo_dict)
    limpio, informe = limpiar(venta)
    unicas = a_espanol(deduplicar(limpio), geo)
    unicas["edificio"] = asignar_edificio(unicas)
    anuncios = a_espanol(limpio.assign(n_anuncios=1), geo)
    anuncios["edificio"] = anuncios["id_vivienda"].map(unicas.set_index("id_vivienda")["edificio"])
    informe["viviendas_unicas"] = len(unicas)
    informe["edificios"] = int(unicas["edificio"].nunique())
    informe["viviendas_que_comparten_edificio"] = int(unicas["edificio"].duplicated(keep=False).sum())
    informe["fuera_de_barrios"] = int((unicas["barrio"] == "fuera").sum())
    if guardar:
        C.DATA_DIR.mkdir(exist_ok=True)
        unicas.to_parquet(C.DATASET_PATH, index=False)
        anuncios.to_parquet(C.DATA_DIR / "valencia_2018_anuncios.parquet", index=False)
        C.GEO_PATH.write_text(json.dumps(geo_dict, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        (C.DATA_DIR / "limpieza.json").write_text(json.dumps(informe, ensure_ascii=False, indent=2), encoding="utf-8")
    return unicas, anuncios, informe


def cargar(anuncios: bool = False) -> pd.DataFrame:
    ruta = C.DATA_DIR / "valencia_2018_anuncios.parquet" if anuncios else C.DATASET_PATH
    if not ruta.exists():
        raise FileNotFoundError(f"No existe {ruta}. Ejecuta `python -m vivienda.datos`.")
    return pd.read_parquet(ruta)


if __name__ == "__main__":
    _, _, inf = preparar()
    print(json.dumps(inf, ensure_ascii=False, indent=2))
