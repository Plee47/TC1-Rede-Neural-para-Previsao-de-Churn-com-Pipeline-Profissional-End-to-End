"""Testes para src/pipeline/retrain.py."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.pipeline.retrain import _load_current_config, run_retrain


# ---------------------------------------------------------------------------
# Fixtures compartilhadas
# ---------------------------------------------------------------------------

def _make_df(n: int = 200) -> pd.DataFrame:
    """DataFrame sintético com schema Telco completo."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "tenure": rng.integers(1, 72, n),
        "SeniorCitizen": rng.integers(0, 2, n),
        "MonthlyCharges": rng.uniform(20, 120, n).round(2),
        "TotalCharges": rng.uniform(50, 8000, n).round(2),
        "gender": rng.choice(["Male", "Female"], n),
        "Partner": rng.choice(["Yes", "No"], n),
        "Dependents": rng.choice(["Yes", "No"], n),
        "PhoneService": rng.choice(["Yes", "No"], n),
        "MultipleLines": rng.choice(["Yes", "No", "No phone service"], n),
        "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n),
        "OnlineSecurity": rng.choice(["Yes", "No", "No internet service"], n),
        "OnlineBackup": rng.choice(["Yes", "No", "No internet service"], n),
        "DeviceProtection": rng.choice(["Yes", "No", "No internet service"], n),
        "TechSupport": rng.choice(["Yes", "No", "No internet service"], n),
        "StreamingTV": rng.choice(["Yes", "No", "No internet service"], n),
        "StreamingMovies": rng.choice(["Yes", "No", "No internet service"], n),
        "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n),
        "PaperlessBilling": rng.choice(["Yes", "No"], n),
        "PaymentMethod": rng.choice(
            ["Electronic check", "Mailed check",
             "Bank transfer (automatic)", "Credit card (automatic)"], n
        ),
        "Churn_bin": rng.integers(0, 2, n),
    })


@pytest.fixture
def artifacts_dir(tmp_path: Path) -> Path:
    """Diretório temporário com artefatos mínimos para testes."""
    config = {
        "auc_roc": 0.80,
        "pr_auc": 0.60,
        "f1": 0.55,
        "precision": 0.50,
        "recall": 0.70,
        "hidden_dims": [128, 64, 32],
        "dropout": 0.3,
        "threshold": 0.35,
        "input_dim": 46,
    }
    (tmp_path / "model_config.json").write_text(json.dumps(config))
    (tmp_path / "mlp_weights.pt").write_bytes(b"placeholder")
    (tmp_path / "preprocessor.joblib").write_bytes(b"placeholder")
    return tmp_path
