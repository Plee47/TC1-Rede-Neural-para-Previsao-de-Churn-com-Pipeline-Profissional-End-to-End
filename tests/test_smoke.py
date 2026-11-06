"""
tests/test_smoke.py
Smoke tests — verifica que os módulos importam e executam sem erros críticos.
"""

import pytest


@pytest.mark.unit
def test_loader_imports():
    """src.data.loader deve ser importável."""
    from src.data.loader import load_raw  # noqa: F401


@pytest.mark.unit
def test_metrics_imports():
    """src.evaluation.metrics deve ser importável."""
    from src.evaluation.metrics import compute_metrics  # noqa: F401


@pytest.mark.unit
def test_api_imports():
    """src.api.app deve ser importável e expor o objeto FastAPI."""
    from src.api.app import app

    assert app is not None


@pytest.mark.unit
def test_load_raw_runs(raw_csv_path):
    """load_raw deve processar o CSV sintético sem lançar exceção."""
    from src.data.loader import load_raw

    df = load_raw(raw_csv_path)
    assert df is not None
    assert len(df) > 0


@pytest.mark.unit
def test_compute_metrics_runs(binary_arrays):
    """compute_metrics deve retornar um dicionário com todas as chaves esperadas."""
    from src.evaluation.metrics import compute_metrics

    y_true, y_prob = binary_arrays
    result = compute_metrics(y_true, y_prob)

    expected_keys = {"auc_roc", "pr_auc", "f1", "precision", "recall"}
    assert expected_keys == set(result.keys())
    assert all(0.0 <= v <= 1.0 for v in result.values())
