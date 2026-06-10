"""
tests/test_artifacts_integration.py
Testes de integração — exigem os artefatos REAIS em models/artifacts/.

Diferente de test_api.py (que injeta um modelo sintético), aqui o lifespan
da aplicação carrega preprocessor.joblib + mlp_weights.pt de verdade.
São pulados automaticamente se os artefatos não existirem; no pipeline de
retreino rodam logo após o export, como smoke test do modelo novo.
"""

import os
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from src.api.app import app

_ARTIFACTS_DIR = Path(os.environ.get("MODEL_ARTIFACTS_DIR", "models/artifacts"))

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (_ARTIFACTS_DIR / "mlp_weights.pt").exists(),
        reason="artefatos ausentes — rode `python -m src.models.export_artifacts`",
    ),
]

# Perfis de referência (mesmos da página de demo)
CLIENTE_ALTO_RISCO = {
    "tenure": 2,
    "monthly_charges": 85.0,
    "total_charges": 170.0,
    "gender": "Male",
    "senior_citizen": 1,
    "partner": "No",
    "dependents": "No",
    "phone_service": "Yes",
    "multiple_lines": "No",
    "internet_service": "Fiber optic",
    "online_security": "No",
    "online_backup": "No",
    "device_protection": "No",
    "tech_support": "No",
    "streaming_tv": "Yes",
    "streaming_movies": "Yes",
    "contract": "Month-to-month",
    "paperless_billing": "Yes",
    "payment_method": "Electronic check",
}
CLIENTE_BAIXO_RISCO = {
    "tenure": 65,
    "monthly_charges": 20.0,
    "total_charges": 1300.0,
    "gender": "Female",
    "senior_citizen": 0,
    "partner": "Yes",
    "dependents": "Yes",
    "phone_service": "Yes",
    "multiple_lines": "No",
    "internet_service": "DSL",
    "online_security": "Yes",
    "online_backup": "Yes",
    "device_protection": "Yes",
    "tech_support": "Yes",
    "streaming_tv": "No",
    "streaming_movies": "No",
    "contract": "Two year",
    "paperless_billing": "No",
    "payment_method": "Mailed check",
}


def test_lifespan_carrega_artefatos_reais():
    """A aplicação deve carregar os artefatos do disco no startup."""
    with TestClient(app) as client:
        body = client.get("/model-info").json()
        assert body["model_loaded"] is True
        assert body["input_dim"] == 46


def test_predict_com_modelo_real():
    """POST /predict com o modelo real deve devolver probabilidade válida."""
    with TestClient(app) as client:
        response = client.post("/predict", json=CLIENTE_ALTO_RISCO)
        assert response.status_code == 200
        body = response.json()
        assert 0.0 <= body["churn_probability"] <= 1.0
        assert body["churn_prediction"] in (0, 1)


def test_modelo_discrimina_alto_de_baixo_risco():
    """Sanidade do modelo: perfil de alto risco deve pontuar acima do fiel."""
    with TestClient(app) as client:
        prob_alto = client.post("/predict", json=CLIENTE_ALTO_RISCO).json()[
            "churn_probability"
        ]
        prob_baixo = client.post("/predict", json=CLIENTE_BAIXO_RISCO).json()[
            "churn_probability"
        ]
    assert prob_alto > prob_baixo, (
        f"modelo não discrimina: alto={prob_alto} <= baixo={prob_baixo}"
    )
