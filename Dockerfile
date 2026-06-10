# Churn Prediction API — imagem de produção
# Build:  docker build -t churn-api .
# Run:    docker run -p 8000:8000 churn-api
#
# Pré-requisito: os artefatos do modelo precisam existir em models/artifacts/
# (rode `python -m src.models.export_artifacts` antes do build).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_ARTIFACTS_DIR=/app/models/artifacts \
    PORT=8000

WORKDIR /app

# 1. Dependências primeiro (camada cacheável).
#    torch CPU-only do índice oficial — evita baixar o build CUDA (~2GB).
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

# 2. Código-fonte + artefatos do modelo (baked na imagem).
COPY src/ ./src/
COPY models/artifacts/ ./models/artifacts/

# 3. Usuário não-root.
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

# Healthcheck bate no endpoint /health.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os,urllib.request,sys; sys.exit(0 if urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\",\"8000\")}/health').status==200 else 1)"

# Honra $PORT (Cloud Run/Render/Railway injetam essa variável).
CMD ["sh", "-c", "uvicorn src.api.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
