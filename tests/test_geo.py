import numpy as np
import pytest

from vivienda.geo import cargar_geo, distancia_minima_km, haversine_km

AYUNTAMIENTO = (39.4699, -0.3763)


def test_haversine_conocida():
    # València – Madrid ≈ 303 km en línea recta
    assert haversine_km(39.4699, -0.3763, 40.4168, -3.7038) == pytest.approx(303, abs=3)


def test_distancia_minima_elige_el_punto_mas_cercano():
    puntos = np.array([[39.0, -0.4], [39.47, -0.376]])
    assert distancia_minima_km(39.4699, -0.3763, puntos)[0] < 0.1


def test_geo_tiene_los_puntos_de_interes():
    g = cargar_geo()
    assert len(g.metro) > 50 and len(g.blasco) > 10 and len(g.barrios) >= 70


def test_barrio_de_puntos_conocidos():
    g = cargar_geo()
    assert g.barrio(*AYUNTAMIENTO) is not None
    assert g.barrio(39.4625, -0.3740) == "Russafa"
    assert g.barrio(40.4168, -3.7038) is None          # Madrid
    assert g.barrio(39.46, -0.30) is None              # en el mar


def test_barrios_vectorizado_coincide_con_escalar():
    g = cargar_geo()
    rng = np.random.default_rng(0)
    lat, lon = rng.uniform(39.44, 39.50, 50), rng.uniform(-0.41, -0.33, 50)
    assert list(g.barrios_de(lat, lon)) == [g.barrio(a, b) for a, b in zip(lat, lon)]


def test_distancia_al_centro_desde_el_centro_es_casi_cero():
    g = cargar_geo()
    la, lo = g.centro[0]
    assert g.distancias(la, lo)["dist_centro_km"][0] < 0.01
