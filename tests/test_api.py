"""
tests/test_api.py
Testes da API FastAPI — contrato de endpoints, status codes e schema de resposta.
"""

from fastapi.testclient import TestClient
import numpy as np
import pytest

from src.api.app import app
from src.models.mlp import ChurnMLP

# Payload completo com todos os 19 campos que o modelo espera
VALID_CUSTOMER = {
    "tenure": 12,
    "monthly_charges": 65.0,
    "total_charges": 780.0,
    "gender": "Female",
    "senior_citizen": 0,
    "partner": "Yes",
    "dependents": "No",
    "phone_service": "Yes",
    "multiple_lines": "No",
    "internet_service": "DSL",
    "online_security": "Yes",
    "online_backup": "No",
    "device_protection": "Yes",
    "tech_support": "No",
    "streaming_tv": "No",
    "streaming_movies": "No",
    "contract": "One year",
    "paperless_billing": "Yes",
    "payment_method": "Bank transfer (automatic)",
}


class _MockPreprocessor:
    """Preprocessador sintético que devolve vetor de zeros com input_dim=46."""

    def transform(self, df):
        return np.zeros((len(df), 46), dtype=np.float32)


@pytest.fixture
def client():
    """TestClient com modelo e preprocessador sintéticos injetados no estado da app."""
    mock_model = ChurnMLP(input_dim=46, hidden_dims=[128, 64, 32], dropout=0.3)
    mock_model.eval()

    with TestClient(app) as c:
        c.app.state.preprocessor = _MockPreprocessor()
        c.app.state.model = mock_model
        c.app.state.threshold = 0.5
        c.app.state.config = {
            "input_dim": 46,
            "hidden_dims": [128, 64, 32],
            "dropout": 0.3,
            "threshold": 0.5,
            "model_version": "1.0.0",
        }
        c.app.state.model_loaded = True
        yield c


@pytest.mark.unit
def test_health_returns_200(client):
    """GET /health deve retornar HTTP 200."""
    response = client.get("/health")
    assert response.status_code == 200


@pytest.mark.unit
def test_health_response_schema(client):
    """GET /health deve retornar JSON com campos 'status' e 'version'."""
    response = client.get("/health")
    body = response.json()
    assert body["status"] == "ok"
    assert "version" in body


@pytest.mark.unit
def test_predict_returns_200(client):
    """POST /predict com payload válido deve retornar HTTP 200."""
    response = client.post("/predict", json=VALID_CUSTOMER)
    assert response.status_code == 200


@pytest.mark.unit
def test_predict_response_schema(client):
    """POST /predict deve retornar churn_probability, churn_prediction e threshold."""
    response = client.post("/predict", json=VALID_CUSTOMER)
    body = response.json()
    assert "churn_probability" in body
    assert "churn_prediction" in body
    assert "threshold" in body


@pytest.mark.unit
def test_predict_probability_range(client):
    """churn_probability deve estar no intervalo [0, 1]."""
    response = client.post("/predict", json=VALID_CUSTOMER)
    prob = response.json()["churn_probability"]
    assert 0.0 <= prob <= 1.0


@pytest.mark.unit
def test_predict_binary_prediction(client):
    """churn_prediction deve ser 0 ou 1."""
    response = client.post("/predict", json=VALID_CUSTOMER)
    pred = response.json()["churn_prediction"]
    assert pred in (0, 1)


@pytest.mark.unit
def test_predict_missing_field_returns_422(client):
    """POST /predict sem campo obrigatório deve retornar HTTP 422."""
    payload = {k: v for k, v in VALID_CUSTOMER.items() if k != "tenure"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


@pytest.mark.unit
def test_predict_negative_tenure_returns_422(client):
    """tenure negativo deve ser rejeitado com HTTP 422."""
    payload = {**VALID_CUSTOMER, "tenure": -1}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


@pytest.mark.unit
def test_model_info_when_loaded(client):
    """GET /model-info deve retornar model_loaded=True quando modelo está carregado."""
    response = client.get("/model-info")
    assert response.status_code == 200
    body = response.json()
    assert body["model_loaded"] is True
    assert body["input_dim"] == 46
    assert body["threshold"] == 0.5
    assert body["model_version"] == "1.0.0"
