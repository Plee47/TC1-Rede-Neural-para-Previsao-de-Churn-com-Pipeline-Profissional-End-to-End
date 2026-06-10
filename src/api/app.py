"""
src/api/app.py
Endpoint de inferência para o modelo de previsão de churn.
"""

from contextlib import asynccontextmanager
import json
import logging
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
import joblib
import pandas as pd
from pydantic import BaseModel, Field
import torch

from src.models.mlp import ChurnMLP
from src.models.train import predict_proba

logger = logging.getLogger(__name__)

# Mapeamento de campos da API (snake_case) para nomes de colunas do DataFrame de treino
_FIELD_TO_COL: dict[str, str] = {
    "tenure": "tenure",
    "monthly_charges": "MonthlyCharges",
    "total_charges": "TotalCharges",
    "gender": "gender",
    "senior_citizen": "SeniorCitizen",
    "partner": "Partner",
    "dependents": "Dependents",
    "phone_service": "PhoneService",
    "multiple_lines": "MultipleLines",
    "internet_service": "InternetService",
    "online_security": "OnlineSecurity",
    "online_backup": "OnlineBackup",
    "device_protection": "DeviceProtection",
    "tech_support": "TechSupport",
    "streaming_tv": "StreamingTV",
    "streaming_movies": "StreamingMovies",
    "contract": "Contract",
    "paperless_billing": "PaperlessBilling",
    "payment_method": "PaymentMethod",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carrega artefatos do modelo na inicialização; libera recursos no encerramento."""
    artifacts_dir = Path(os.environ.get("MODEL_ARTIFACTS_DIR", "models/artifacts"))
    app.state.model_loaded = False
    try:
        with open(artifacts_dir / "model_config.json", encoding="utf-8") as f:
            config = json.load(f)

        preprocessor = joblib.load(artifacts_dir / "preprocessor.joblib")

        model = ChurnMLP(
            input_dim=config["input_dim"],
            hidden_dims=config["hidden_dims"],
            dropout=config["dropout"],
        )
        model.load_state_dict(
            torch.load(
                artifacts_dir / "mlp_weights.pt",
                map_location="cpu",
                weights_only=True,
            )
        )
        model.eval()

        app.state.preprocessor = preprocessor
        app.state.model = model
        app.state.threshold = float(config.get("threshold", 0.5))
        app.state.config = config
        app.state.model_loaded = True
        logger.info(
            "Modelo carregado de %s (threshold=%.3f)",
            artifacts_dir,
            app.state.threshold,
        )
    except FileNotFoundError as exc:
        logger.warning(
            "Artefatos não encontrados: %s — gere com "
            "`python -m src.models.export_artifacts` (ou rode notebooks/03_mlp.ipynb).",
            exc,
        )
    yield


app = FastAPI(
    title="Churn Prediction API",
    description="Previsão de churn para clientes de telecomunicações — FIAP TC1",
    version="0.1.0",
    lifespan=lifespan,
)


class CustomerFeatures(BaseModel):
    # Numérico
    tenure: int = Field(..., ge=0, description="Meses como cliente")
    monthly_charges: float = Field(..., ge=0, description="Cobrança mensal (USD)")
    total_charges: float = Field(..., ge=0, description="Cobrança total acumulada (USD)")
    # Demográfico
    gender: str = Field(..., description="Male | Female")
    senior_citizen: int = Field(..., ge=0, le=1, description="Cliente idoso (0=Não, 1=Sim)")
    partner: str = Field(..., description="Yes | No")
    dependents: str = Field(..., description="Yes | No")
    # Serviços de telefonia
    phone_service: str = Field(..., description="Yes | No")
    multiple_lines: str = Field(..., description="Yes | No | No phone service")
    # Serviços de internet
    internet_service: str = Field(..., description="DSL | Fiber optic | No")
    online_security: str = Field(..., description="Yes | No | No internet service")
    online_backup: str = Field(..., description="Yes | No | No internet service")
    device_protection: str = Field(..., description="Yes | No | No internet service")
    tech_support: str = Field(..., description="Yes | No | No internet service")
    streaming_tv: str = Field(..., description="Yes | No | No internet service")
    streaming_movies: str = Field(..., description="Yes | No | No internet service")
    # Contrato e pagamento
    contract: str = Field(..., description="Month-to-month | One year | Two year")
    paperless_billing: str = Field(..., description="Yes | No")
    payment_method: str = Field(
        ...,
        description=(
            "Electronic check | Mailed check | "
            "Bank transfer (automatic) | Credit card (automatic)"
        ),
    )


class PredictionResponse(BaseModel):
    churn_probability: float = Field(..., description="Probabilidade de churn [0, 1]")
    churn_prediction: int = Field(..., description="Predição binária (0=não churn, 1=churn)")
    risk_band: str = Field(..., description="Faixa de risco: Alto | Médio | Baixo")
    threshold: float = Field(..., description="Limiar de decisão utilizado")


class BatchPredictRequest(BaseModel):
    customers: list[CustomerFeatures] = Field(
        ..., min_length=1, description="Lista de clientes para scoring em lote"
    )


class BatchPredictionResponse(BaseModel):
    count: int = Field(..., description="Número de clientes avaliados")
    predictions: list[PredictionResponse]


class ModelInfoResponse(BaseModel):
    model_loaded: bool
    input_dim: int | None = None
    hidden_dims: list | None = None
    threshold: float | None = None
    version: str


def _risk_band(proba: float) -> str:
    """Classifica a probabilidade em faixa de risco para priorização (CRM)."""
    if proba >= 0.70:
        return "Alto"
    if proba >= 0.30:
        return "Médio"
    return "Baixo"


def _ensure_model_loaded() -> None:
    """Aborta com 503 se os artefatos do modelo não estiverem carregados."""
    if not app.state.model_loaded:
        raise HTTPException(
            status_code=503,
            detail=(
                "Modelo não carregado. Gere os artefatos com "
                "`python -m src.models.export_artifacts` (ou rode notebooks/03_mlp.ipynb)."
            ),
        )


def _predict_frame(df_input: pd.DataFrame) -> list[PredictionResponse]:
    """Aplica preprocessador + MLP a um DataFrame e devolve uma resposta por linha."""
    x_pp = app.state.preprocessor.transform(df_input)
    device = torch.device("cpu")
    probas = predict_proba(app.state.model, x_pp, device)
    threshold = app.state.threshold
    return [
        PredictionResponse(
            churn_probability=round(float(p), 6),
            churn_prediction=int(p >= threshold),
            risk_band=_risk_band(float(p)),
            threshold=threshold,
        )
        for p in probas
    ]


_DEMO_HTML_PATH = Path(__file__).parent / "static" / "demo.html"


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def demo_page() -> HTMLResponse:
    """Página de demonstração que consome a própria API."""
    if not _DEMO_HTML_PATH.exists():
        return HTMLResponse(
            "<h1>Churn Prediction API</h1><p>Documentação em <a href='/docs'>/docs</a>.</p>"
        )
    return HTMLResponse(_DEMO_HTML_PATH.read_text(encoding="utf-8"))


@app.get("/health", tags=["infra"])
def health_check() -> dict:
    """Verifica se a API está operacional."""
    logger.info("Health check solicitado")
    return {"status": "ok", "version": app.version}


@app.get("/model-info", response_model=ModelInfoResponse, tags=["infra"])
def model_info() -> ModelInfoResponse:
    """Retorna metadados do modelo atualmente carregado."""
    if not app.state.model_loaded:
        return ModelInfoResponse(model_loaded=False, version=app.version)
    cfg = app.state.config
    return ModelInfoResponse(
        model_loaded=True,
        input_dim=cfg.get("input_dim"),
        hidden_dims=cfg.get("hidden_dims"),
        threshold=app.state.threshold,
        version=app.version,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict(customer: CustomerFeatures) -> PredictionResponse:
    """Retorna a probabilidade de churn para um cliente."""
    _ensure_model_loaded()
    logger.info(
        "Predição solicitada | tenure=%d | contract=%s",
        customer.tenure,
        customer.contract,
    )
    row = {_FIELD_TO_COL[field]: value for field, value in customer.model_dump().items()}
    return _predict_frame(pd.DataFrame([row]))[0]


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["inference"])
def predict_batch(request: BatchPredictRequest) -> BatchPredictionResponse:
    """Scoring de churn para uma lista de clientes — caso de uso do time de CRM."""
    _ensure_model_loaded()
    logger.info("Predição em lote solicitada | n=%d", len(request.customers))
    rows = [
        {_FIELD_TO_COL[field]: value for field, value in cust.model_dump().items()}
        for cust in request.customers
    ]
    preds = _predict_frame(pd.DataFrame(rows))
    return BatchPredictionResponse(count=len(preds), predictions=preds)
