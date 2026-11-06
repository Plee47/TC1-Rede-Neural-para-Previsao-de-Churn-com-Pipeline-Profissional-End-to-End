"""
src/models/train.py
Loop de treinamento com early stopping e mini-batching para o ChurnMLP.
"""

from __future__ import annotations

import copy
import logging

import numpy as np
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Para o treinamento quando o AUC de validação para de melhorar."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score: float | None = None
        self.best_weights: dict | None = None
        self.should_stop = False

    def __call__(self, score: float, model: nn.Module) -> None:
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.best_weights = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(
                    "Early stopping ativado após %d épocas sem melhora (melhor AUC=%.4f)",
                    self.patience,
                    self.best_score,
                )


def _to_loader(
    X: np.ndarray,  # noqa: N803
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    x_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    return DataLoader(TensorDataset(x_t, y_t), batch_size=batch_size, shuffle=shuffle)


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for x_b, y_b in loader:
        x_b, y_b = x_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x_b), y_b)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_b)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def predict_proba(
    model: nn.Module,
    X: np.ndarray,  # noqa: N803
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    """Retorna probabilidades da classe positiva (churn=1)."""
    model.eval()
    loader = _to_loader(X, np.zeros(len(X)), batch_size=batch_size, shuffle=False)
    probs: list[np.ndarray] = []
    for x_b, _ in loader:
        logits = model(x_b.to(device))
        probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs)


def fit(
    model: nn.Module,
    X_train: np.ndarray,  # noqa: N803
    y_train: np.ndarray,
    X_val: np.ndarray,  # noqa: N803
    y_val: np.ndarray,
    *,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 10,
    pos_weight: float | None = None,
    device: torch.device | None = None,
) -> dict:
    """
    Treina o modelo com mini-batches e early stopping monitorando AUC-ROC de validação.
    Restaura automaticamente os melhores pesos ao final.

    Returns:
        Dicionário com histórico de loss/AUC e metadados do treinamento.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    pw = torch.tensor([pos_weight], device=device) if pos_weight is not None else None
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loader = _to_loader(X_train, y_train, batch_size=batch_size, shuffle=True)
    stopper = EarlyStopping(patience=patience)

    history: dict = {"train_loss": [], "val_auc": []}
    epoch = 0

    for epoch in range(1, epochs + 1):
        train_loss = _train_epoch(model, train_loader, criterion, optimizer, device)
        val_probs = predict_proba(model, X_val, device)
        val_auc = float(roc_auc_score(y_val, val_probs))

        history["train_loss"].append(train_loss)
        history["val_auc"].append(val_auc)

        stopper(val_auc, model)

        if epoch % 10 == 0 or stopper.should_stop:
            logger.info(
                "Época %3d | loss=%.4f | val_auc=%.4f | melhor=%.4f",
                epoch,
                train_loss,
                val_auc,
                stopper.best_score,
            )

        if stopper.should_stop:
            break

    if stopper.best_weights is not None:
        model.load_state_dict(stopper.best_weights)

    history["best_val_auc"] = stopper.best_score
    history["epochs_trained"] = epoch

    return history
