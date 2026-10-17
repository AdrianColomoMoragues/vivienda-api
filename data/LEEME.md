# Datos

- `valencia_2018.parquet`: una fila por vivienda (27 309), limpia y deduplicada, con barrio y edificio.
- `valencia_2018_anuncios.parquet`: todos los anuncios tras la limpieza (33 523), para medir la fuga por duplicados.
- `valencia_geo.json`: centro, paradas de metro, avenida Blasco Ibáñez y polígonos de los barrios.
- `limpieza.json`: filas eliminadas por cada regla.

Generados con `python -m vivienda.datos` a partir del paquete de R
[idealista18](https://github.com/paezha/idealista18) (Rey-Blanco, Arbués, López y Páez).

Esta base de datos derivada se distribuye con la licencia
[Open Database License (ODbL) v1.0](https://opendatacommons.org/licenses/odbl/1-0/), igual que la original.
