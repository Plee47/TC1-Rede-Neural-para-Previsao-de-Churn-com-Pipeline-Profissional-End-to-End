"""
tests/test_models.py
Testes unitários para ChurnMLP e utilitários de treinamento.
"""

import numpy as np
import pytest
import torch


@pytest.mark.unit
def test_churn_mlp_forward_shape():
    """ChurnMLP deve retornar tensor 1-D com tamanho igual ao batch."""
    from src.models.mlp import ChurnMLP

    model = ChurnMLP(input_dim=20, hidden_dims=[32, 16], dropout=0.0)
    x = torch.randn(8, 20)
    out = model(x)
    assert out.shape == (8,), f"Esperado (8,), obtido {out.shape}"


@pytest.mark.unit
def test_churn_mlp_default_hidden_dims():
    """ChurnMLP sem hidden_dims deve usar [128, 64, 32]."""
    from src.models.mlp import ChurnMLP

    model = ChurnMLP(input_dim=10)
    layers = [m for m in model.network if isinstance(m, torch.nn.Linear)]
    hidden_sizes = [layer.out_features for layer in layers[:-1]]
    assert hidden_sizes == [128, 64, 32]


@pytest.mark.unit
def test_churn_mlp_output_is_logit():
    """Saída deve ser logit: pode assumir qualquer valor real (não restrito a [0,1])."""
    from src.models.mlp import ChurnMLP

    model = ChurnMLP(input_dim=10, hidden_dims=[16], dropout=0.0)
    x = torch.randn(50, 10) * 10
    out = model(x)
    assert (out < 0).any() or (out > 1).any(), "Saída parece sigmoid, não logit"


@pytest.mark.unit
def test_early_stopping_triggers():
    """EarlyStopping deve disparar após `patience` épocas sem melhora."""
    from src.models.mlp import ChurnMLP
    from src.models.train import EarlyStopping

    stopper = EarlyStopping(patience=3, min_delta=0.0)
    model = ChurnMLP(input_dim=4, hidden_dims=[8])

    stopper(0.80, model)  # melhora
    stopper(0.79, model)  # sem melhora 1
    stopper(0.79, model)  # sem melhora 2
    assert not stopper.should_stop
    stopper(0.79, model)  # sem melhora 3 → dispara
    assert stopper.should_stop


@pytest.mark.unit
def test_early_stopping_restores_best_weights():
    """EarlyStopping deve preservar os pesos do melhor checkpoint."""
    from src.models.mlp import ChurnMLP
    from src.models.train import EarlyStopping

    model = ChurnMLP(input_dim=4, hidden_dims=[8])
    stopper = EarlyStopping(patience=5)

    stopper(0.70, model)
    best_state = {k: v.clone() for k, v in stopper.best_weights.items()}

    stopper(0.65, model)  # piora — não deve sobrescrever best_weights

    for key in best_state:
        assert torch.equal(stopper.best_weights[key], best_state[key])


@pytest.mark.unit
def test_predict_proba_range():
    """predict_proba deve retornar valores em [0, 1]."""
    from src.models.mlp import ChurnMLP
    from src.models.train import predict_proba

    model = ChurnMLP(input_dim=10, hidden_dims=[16], dropout=0.0)
    x_data = np.random.randn(30, 10).astype(np.float32)
    device = torch.device("cpu")
    probs = predict_proba(model, x_data, device)

    assert probs.shape == (30,)
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0


@pytest.mark.slow
def test_fit_reduces_loss():
    """fit() deve produzir melhor AUC após treinamento vs aleatório."""
    from sklearn.metrics import roc_auc_score

    from src.models.mlp import ChurnMLP
    from src.models.train import fit, predict_proba

    rng = np.random.default_rng(42)
    n, d = 200, 15
    x_all = rng.standard_normal((n, d)).astype(np.float32)
    # Target correlacionado com primeira feature para aprendizado ser possível
    y = (x_all[:, 0] + rng.standard_normal(n) * 0.5 > 0).astype(np.float32)

    x_train, x_val = x_all[:160], x_all[160:]
    y_train, y_val = y[:160], y[160:]

    model = ChurnMLP(input_dim=d, hidden_dims=[32, 16], dropout=0.0)
    device = torch.device("cpu")

    auc_before = roc_auc_score(y_val, predict_proba(model, x_val, device))

    fit(
        model,
        x_train,
        y_train,
        x_val,
        y_val,
        epochs=30,
        batch_size=32,
        patience=30,
        device=device,
    )

    auc_after = roc_auc_score(y_val, predict_proba(model, x_val, device))

    assert auc_after > auc_before, (
        f"AUC não melhorou: antes={auc_before:.4f}, depois={auc_after:.4f}"
    )
