from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import sys

import joblib
import mlflow
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import torch

from src.data.loader import load_raw
from src.evaluation.metrics import compute_metrics
from src.models.mlp import ChurnMLP
from src.models.train import fit, predict_proba

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

NUMERIC_COLS = ["tenure", "SeniorCitizen", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_COLS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]


def _build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLS),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_COLS,
            ),
        ]
    )


def _load_current_config(artifacts_path: Path) -> dict:
    with open(artifacts_path / "model_config.json") as f:
        return json.load(f)


def _bump_version(version: str) -> str:
    """Incrementa o patch de um SemVer (X.Y.Z -> X.Y.Z+1); fallback para '1.0.0'."""
    try:
        major, minor, patch = (int(p) for p in str(version).split("."))
        return f"{major}.{minor}.{patch + 1}"
    except (ValueError, TypeError):
        return "1.0.0"


def _save_artifacts(
    artifacts_path: Path,
    model: ChurnMLP,
    preprocessor: ColumnTransformer,
    config: dict,
) -> None:
    torch.save(model.state_dict(), artifacts_path / "mlp_weights.pt")
    joblib.dump(preprocessor, artifacts_path / "preprocessor.joblib")
    with open(artifacts_path / "model_config.json", "w") as f:
        json.dump(config, f, indent=2)


def run_retrain(
    data_path: str | Path,
    artifacts_path: str | Path,
    experiment_name: str = "tc1-churn-retrain",
) -> int:
    data_path = Path(data_path)
    artifacts_path = Path(artifacts_path)

    current_config = _load_current_config(artifacts_path)
    current_auc = float(current_config["auc_roc"])
    logger.info("AUC-ROC atual: %.4f", current_auc)

    df = load_raw(data_path)
    X = df[NUMERIC_COLS + CATEGORICAL_COLS]
    y = df["Churn_bin"].values

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=round(0.15 / 0.85, 4),
        random_state=42,
        stratify=y_trainval,
    )

    preprocessor = _build_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train).astype(np.float32)
    X_val_t = preprocessor.transform(X_val).astype(np.float32)
    X_test_t = preprocessor.transform(X_test).astype(np.float32)

    hidden_dims: list[int] = current_config.get("hidden_dims", [128, 64, 32])
    dropout = float(current_config.get("dropout", 0.3))
    threshold = float(current_config.get("threshold", 0.35))
    input_dim = X_train_t.shape[1]

    pos_weight_val = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    model = ChurnMLP(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout)

    fit(
        model,
        X_train_t,
        y_train,
        X_val_t,
        y_val,
        epochs=100,
        batch_size=256,
        lr=1e-3,
        patience=10,
        pos_weight=torch.tensor([pos_weight_val]),
    )

    y_prob = predict_proba(model, X_test_t, device="cpu")
    new_metrics = compute_metrics(y_test, y_prob, threshold=threshold)
    new_auc = float(new_metrics["auc_roc"])
    logger.info("AUC-ROC novo: %.4f", new_auc)

    promoted = new_auc > current_auc

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        mlflow.log_params(
            {
                "input_dim": input_dim,
                "hidden_dims": str(hidden_dims),
                "dropout": dropout,
                "threshold": threshold,
            }
        )
        mlflow.log_metrics({k: float(v) for k, v in new_metrics.items()})
        mlflow.set_tag("promoted", str(promoted).lower())

    if promoted:
        new_config = {
            **current_config,
            **{k: float(v) for k, v in new_metrics.items()},
            "input_dim": input_dim,
            "model_version": _bump_version(current_config.get("model_version", "1.0.0")),
        }
        _save_artifacts(artifacts_path, model, preprocessor, new_config)
        logger.info("Modelo promovido (AUC %.4f > %.4f)", new_auc, current_auc)
        return 0

    logger.info("Modelo rejeitado (AUC %.4f <= %.4f)", new_auc, current_auc)
    return 2


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-treino mensal do modelo de churn")
    parser.add_argument(
        "--data-path",
        default=os.getenv(
            "RETRAIN_DATA_PATH",
            "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv",
        ),
    )
    parser.add_argument(
        "--artifacts-path",
        default=os.getenv("RETRAIN_ARTIFACTS_PATH", "models/artifacts"),
    )
    parser.add_argument(
        "--experiment-name",
        default=os.getenv("MLFLOW_EXPERIMENT_NAME", "tc1-churn-retrain"),
    )
    args = parser.parse_args()
    sys.exit(run_retrain(args.data_path, args.artifacts_path, args.experiment_name))


if __name__ == "__main__":
    main()
