PISO = {"latitud": 39.4699, "longitud": -0.3763, "metros": 95, "dormitorios": 3, "banos": 2, "planta": 3,
        "anio_construccion": 1975, "ascensor": True}


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}


def test_home_sirve_la_web(client):
    r = client.get("/")
    assert r.status_code == 200 and "leaflet" in r.text.lower()


def test_model_info(client):
    info = client.get("/model-info").json()
    assert {"modelo", "cv", "test", "intervalos", "barrios"} <= set(info)


def test_predict(client):
    r = client.post("/predict", json=PISO)
    assert r.status_code == 200, r.text
    j = r.json()
    lo, hi = j["intervalo_90_eur"]
    assert 0 < lo <= j["precio_eur"] <= hi
    assert j["barrio"] and j["dist_centro_km"] < 1
    assert abs(j["euros_m2"] - j["precio_eur"] / PISO["metros"]) < 10


def test_campos_opcionales(client):
    minimo = {k: PISO[k] for k in ["latitud", "longitud", "metros", "dormitorios", "banos"]}
    assert client.post("/predict", json=minimo).status_code == 200


def test_mas_metros_mas_caro(client):
    p1 = client.post("/predict", json={**PISO, "metros": 60}).json()["precio_eur"]
    p2 = client.post("/predict", json={**PISO, "metros": 180}).json()["precio_eur"]
    assert p2 > p1


def test_fuera_de_valencia(client):
    r = client.post("/predict", json={**PISO, "latitud": 39.46, "longitud": -0.30})   # mar
    assert r.status_code == 422
    assert "fuera" in r.json()["detail"].lower()


def test_validaciones(client):
    assert client.post("/predict", json={**PISO, "metros": 5}).status_code == 422
    assert client.post("/predict", json={**PISO, "latitud": 40.4}).status_code == 422
    assert client.post("/predict", json={**PISO, "dormitorios": 0}).status_code == 422          # no es estudio
    assert client.post("/predict", json={**PISO, "dormitorios": 0, "estudio": True}).status_code == 200
    assert client.post("/predict", json={**PISO, "planta": 9, "plantas_edificio": 5}).status_code == 422
    assert client.post("/predict", json={**PISO, "estado": "ruina"}).status_code == 422


def test_barrio(client):
    j = client.get("/barrio", params={"latitud": 39.4625, "longitud": -0.3740}).json()
    assert j["barrio"] == "Russafa" and "dist_metro_km" in j
    assert client.get("/barrio", params={"latitud": 39.46, "longitud": -0.30}).status_code == 404


def test_batch(client):
    r = client.post("/predict/batch", json=[PISO, {**PISO, "metros": 150}])
    assert r.status_code == 200 and len(r.json()) == 2
    assert client.post("/predict/batch", json=[]).status_code == 422
    assert client.post("/predict/batch", json=[PISO] * 1001).status_code == 413


def test_barrios_geojson(client):
    gj = client.get("/barrios.geojson").json()
    assert gj["type"] == "FeatureCollection" and len(gj["features"]) >= 70
    assert {"barrio", "euros_m2"} <= set(gj["features"][0]["properties"])
