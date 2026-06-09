from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import joblib
import mlflow
import numpy as np
import torch
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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
    raise NotImplementedError


def _load_current_config(artifacts_path: Path) -> dict:
    with open(artifacts_path / "model_config.json") as f:
        return json.load(f)


def _save_artifacts(
    artifacts_path: Path,
    model: ChurnMLP,
    preprocessor: ColumnTransformer,
    config: dict,
) -> None:
    raise NotImplementedError


def run_retrain(
    data_path: str | Path,
    artifacts_path: str | Path,
    experiment_name: str = "tc1-churn-retrain",
) -> int:
    raise NotImplementedError


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
