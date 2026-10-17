FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY vivienda/ vivienda/
COPY static/ static/
COPY data/valencia_2018.parquet data/valencia_geo.json data/
COPY models/model_info.json models/

# El modelo se entrena al construir la imagen con los datos incluidos en el repositorio
# (sin descargas): la imagen es autocontenida y reproducible (misma semilla, mismos datos).
# --final reutiliza el informe de evaluación de models/model_info.json y solo ajusta los modelos finales.
RUN python -m vivienda.train --final

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

ENV PORT=8000
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\", 8000)}/health')"
CMD ["sh", "-c", "uvicorn vivienda.api:app --host 0.0.0.0 --port ${PORT}"]
