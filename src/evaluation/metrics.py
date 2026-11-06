"""
src/evaluation/metrics.py
Métricas de avaliação reutilizáveis para todos os modelos do projeto.
"""

import logging

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    Calcula o conjunto completo de métricas de avaliação.

    Args:
        y_true:    labels reais (0 ou 1)
        y_prob:    probabilidades preditas para a classe positiva
        threshold: limiar de classificação (default 0.5)

    Returns:
        Dicionário com auc_roc, pr_auc, f1, precision e recall
    """
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "auc_roc": round(roc_auc_score(y_true, y_prob), 4),
        "pr_auc": round(average_precision_score(y_true, y_prob), 4),
        "f1": round(f1_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
    }

    logger.info(
        "Métricas (threshold=%.2f): AUC-ROC=%.4f | PR-AUC=%.4f | F1=%.4f | Recall=%.4f",
        threshold,
        metrics["auc_roc"],
        metrics["pr_auc"],
        metrics["f1"],
        metrics["recall"],
    )

    return metrics
