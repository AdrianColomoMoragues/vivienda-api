"""Geometría de València: distancias a puntos de interés y barrio de un punto.

Se usa tanto al preparar los datos como en la API, para que el modelo reciba
exactamente las mismas variables en los dos casos.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from .config import GEO_PATH

RADIO_TIERRA_KM = 6371.0


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * RADIO_TIERRA_KM * np.arcsin(np.sqrt(a))


def distancia_minima_km(lat, lon, puntos: np.ndarray) -> np.ndarray:
    """Distancia de cada punto (lat, lon) al más cercano de `puntos` [[lat, lon], ...]."""
    lat, lon = np.atleast_1d(lat).astype(float), np.atleast_1d(lon).astype(float)
    d = haversine_km(lat[:, None], lon[:, None], puntos[None, :, 0], puntos[None, :, 1])
    return d.min(axis=1)


def _dentro(lat, lon, anillo: np.ndarray) -> np.ndarray:
    """Ray casting vectorizado: ¿qué puntos (lat, lon) están dentro del anillo [[lon, lat], ...]?"""
    x = np.atleast_1d(lon).astype(float)[:, None]
    y = np.atleast_1d(lat).astype(float)[:, None]
    x1, y1 = anillo[:, 0][None, :], anillo[:, 1][None, :]
    x2, y2 = np.roll(anillo[:, 0], 1)[None, :], np.roll(anillo[:, 1], 1)[None, :]
    cruza = (y1 > y) != (y2 > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        x_corte = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
    return ((cruza & (x < x_corte)).sum(axis=1) % 2) == 1


class Geo:
    """Puntos de interés y barrios cargados de `valencia_geo.json`."""

    def __init__(self, data: dict):
        self.centro = np.array([data["centro"]], float)            # [[lat, lon]]
        self.metro = np.array(data["metro"], float)
        self.blasco = np.array(data["blasco"], float)
        self.barrios = [(b["nombre"], [np.array(r, float) for r in b["anillos"]]) for b in data["barrios"]]
        todos = np.vstack([r for _, rs in self.barrios for r in rs])
        self.lon_min, self.lat_min = todos.min(axis=0)
        self.lon_max, self.lat_max = todos.max(axis=0)

    def distancias(self, lat, lon) -> dict[str, np.ndarray]:
        return {
            "dist_centro_km": distancia_minima_km(lat, lon, self.centro),
            "dist_metro_km": distancia_minima_km(lat, lon, self.metro),
            "dist_blasco_km": distancia_minima_km(lat, lon, self.blasco),
        }

    def barrios_de(self, lat, lon) -> np.ndarray:
        """Barrio de cada punto (None si está fuera de los barrios de la ciudad)."""
        lat, lon = np.atleast_1d(lat).astype(float), np.atleast_1d(lon).astype(float)
        out = np.full(len(lat), None, dtype=object)
        for nombre, anillos in self.barrios:
            idx = np.flatnonzero(np.equal(out, None))
            if len(idx) == 0:
                break
            # el primer anillo es el exterior; los siguientes, huecos
            dentro = _dentro(lat[idx], lon[idx], anillos[0])
            for hueco in anillos[1:]:
                dentro &= ~_dentro(lat[idx], lon[idx], hueco)
            out[idx[dentro]] = nombre
        return out

    def barrio(self, lat: float, lon: float) -> str | None:
        """Nombre del barrio que contiene el punto, o None si está fuera de la ciudad."""
        if not (self.lat_min <= lat <= self.lat_max and self.lon_min <= lon <= self.lon_max):
            return None
        return self.barrios_de(lat, lon)[0]


@lru_cache(maxsize=2)
def cargar_geo(path: Path = GEO_PATH) -> Geo:
    return Geo(json.loads(Path(path).read_text(encoding="utf-8")))
