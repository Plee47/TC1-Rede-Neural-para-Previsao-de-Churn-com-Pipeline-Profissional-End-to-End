"""
src/models/export_artifacts.py

Treina a MLP de churn e exporta os três artefatos que a API
(`src/api/app.py`) carrega na inicialização:

  - preprocessor.joblib : ColumnTransformer (StandardScaler + OneHotEncoder) fitado no treino
  - mlp_weights.pt      : state_dict do melhor checkpoint PyTorch
  - model_config.json   : input_dim, hidden_dims, dropout, threshold ótimo e métricas

Reproduz a lógica de ``notebooks/03_mlp.ipynb`` num script executável, porque
Docker/CI não rodam notebooks. Reaproveita os módulos já existentes em ``src/``.

Uso:
    python -m src.models.export_artifacts
    python -m src.models.export_artifacts --data data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv --out models/artifacts
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import torch

from src.data.loader import load_raw
from src.evaluation.metrics import compute_metrics
from src.models.mlp import ChurnMLP
from src.models.train import fit, predict_proba

logger = logging.getLogger(__name__)

SEED = 42

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]
TARGET = "Churn_bin"

# Hiperparâmetros — idênticos a notebooks/03_mlp.ipynb
HIDDEN_DIMS = [128, 64, 32]
DROPOUT = 0.3
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 256
MAX_EPOCHS = 150
PATIENCE = 15

# Custos de negócio para escolher o threshold ótimo (idênticos ao notebook)
COST_FP = 10
COST_FN = 300


def build_preprocessor() -> ColumnTransformer:
    """ColumnTransformer com StandardScaler (numéricas) + OneHotEncoder (categóricas)."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_fp: int = COST_FP,
    cost_fn: int = COST_FN,
) -> tuple[float, int]:
    """
    Threshold que minimiza o custo total de negócio (FP*cost_fp + FN*cost_fn).

    Reproduz a análise de custo de notebooks/03_mlp.ipynb: como o FN é ~30x mais
    caro que o FP, o ponto ótimo favorece Recall (threshold < 0.5).
    """
    best_t, best_cost = 0.5, float("inf")
    for t in np.linspace(0.05, 0.95, 91):
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        cost = fp * cost_fp + fn * cost_fn
        if cost < best_cost:
            best_cost, best_t = cost, float(round(t, 3))
    return best_t, int(best_cost)


def main(data_path: str, out_dir: str) -> None:
    # Console Windows usa cp1252 e quebra em emoji/acentos — força UTF-8.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # 1. Dados ------------------------------------------------------------------
    df = load_raw(data_path)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET].values

    # 70 / 15 / 15 estratificado — idêntico ao notebook
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.15, random_state=SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=round(0.15 / 0.85, 4),
        random_state=SEED,
        stratify=y_trainval,
    )

    # 2. Pré-processamento ------------------------------------------------------
    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)
    X_train_pp = preprocessor.transform(X_train)
    X_val_pp = preprocessor.transform(X_val)
    X_test_pp = preprocessor.transform(X_test)

    input_dim = X_train_pp.shape[1]
    pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    logger.info(
        "Treino=%d Val=%d Teste=%d | input_dim=%d | pos_weight=%.2f",
        len(X_train), len(X_val), len(X_test), input_dim, pos_weight,
    )

    # 3. Treino da MLP ----------------------------------------------------------
    model = ChurnMLP(input_dim=input_dim, hidden_dims=HIDDEN_DIMS, dropout=DROPOUT)
    history = fit(
        model,
        X_train_pp, y_train,
        X_val_pp, y_val,
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        patience=PATIENCE,
        pos_weight=pos_weight,
        device=device,
    )

    # 4. Avaliação + threshold ótimo -------------------------------------------
    y_prob_test = predict_proba(model, X_test_pp, device)
    test_m = compute_metrics(y_test, y_prob_test)
    best_thresh, best_cost = optimal_threshold(y_test, y_prob_test)
    logger.info("Teste: %s", test_m)
    logger.info("Threshold ótimo=%.3f (custo=$%d)", best_thresh, best_cost)

    # 5. Exportar artefatos -----------------------------------------------------
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    joblib.dump(preprocessor, out / "preprocessor.joblib")
    torch.save(model.state_dict(), out / "mlp_weights.pt")

    config = {
        "input_dim": input_dim,
        "hidden_dims": HIDDEN_DIMS,
        "dropout": DROPOUT,
        "threshold": best_thresh,
        "pos_weight": round(pos_weight, 3),
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "test_auc_roc": test_m["auc_roc"],
        "test_pr_auc": test_m["pr_auc"],
        "test_recall": test_m["recall"],
        "test_f1": test_m["f1"],
        "best_val_auc": round(float(history["best_val_auc"]), 4),
    }
    with open(out / "model_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Artefatos exportados em {out.resolve()}")
    print(
        f"   input_dim={input_dim} | threshold={best_thresh} | "
        f"test AUC={test_m['auc_roc']} recall={test_m['recall']}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Treina e exporta os artefatos da MLP de churn para models/artifacts/."
    )
    parser.add_argument(
        "--data",
        default="data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv",
        help="Caminho para o CSV bruto do Telco.",
    )
    parser.add_argument(
        "--out",
        default="models/artifacts",
        help="Diretório de saída dos artefatos.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.data, args.out)
