"""
src/models/mlp.py
Arquitetura MLP em PyTorch para classificação binária de churn.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ChurnMLP(nn.Module):
    """
    MLP com BatchNorm + Dropout para previsão de churn.

    Retorna logits (sem sigmoid) — use BCEWithLogitsLoss no treino.
    Para inferência use torch.sigmoid() ou src.models.train.predict_proba().
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers += [
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(1)
