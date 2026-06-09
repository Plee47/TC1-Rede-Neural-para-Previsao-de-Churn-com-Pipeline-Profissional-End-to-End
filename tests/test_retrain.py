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


@pytest.mark.unit
def test_load_current_config_returns_dict_with_auc(artifacts_dir: Path) -> None:
    config = _load_current_config(artifacts_dir)
    assert config["auc_roc"] == 0.80
    assert config["threshold"] == 0.35
    assert config["hidden_dims"] == [128, 64, 32]


@pytest.mark.unit
def test_build_preprocessor_transforms_to_correct_shape() -> None:
    from src.pipeline.retrain import _build_preprocessor, NUMERIC_COLS, CATEGORICAL_COLS
    df = _make_df(50)
    preprocessor = _build_preprocessor()
    X = df[NUMERIC_COLS + CATEGORICAL_COLS]
    X_t = preprocessor.fit_transform(X)
    # 4 numeric + OHE categorical — deve ser >= 19 features
    assert X_t.shape[0] == 50
    assert X_t.shape[1] >= 19


@pytest.mark.unit
def test_save_artifacts_writes_all_three_files(artifacts_dir: Path) -> None:
    from src.pipeline.retrain import _build_preprocessor, _save_artifacts, NUMERIC_COLS, CATEGORICAL_COLS
    df = _make_df(50)
    preprocessor = _build_preprocessor()
    X = df[NUMERIC_COLS + CATEGORICAL_COLS]
    preprocessor.fit(X)

    model = MagicMock()
    model.state_dict.return_value = {}

    new_config = {"auc_roc": 0.85, "threshold": 0.35, "hidden_dims": [128, 64, 32],
                  "dropout": 0.3, "input_dim": 46}

    with patch("src.pipeline.retrain.torch.save") as mock_torch_save:
        _save_artifacts(artifacts_dir, model, preprocessor, new_config)

    mock_torch_save.assert_called_once()
    assert (artifacts_dir / "preprocessor.joblib").exists()
    saved_config = json.loads((artifacts_dir / "model_config.json").read_text())
    assert saved_config["auc_roc"] == 0.85


@pytest.mark.unit
def test_run_retrain_promotes_when_new_model_is_better(artifacts_dir: Path) -> None:
    """Exit 0 e artefatos atualizados quando AUC novo (0.90) > atual (0.80)."""
    better_metrics = {
        "auc_roc": 0.90, "pr_auc": 0.65, "f1": 0.60,
        "precision": 0.55, "recall": 0.82,
    }
    mock_mlflow = MagicMock()
    mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=MagicMock())
    mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

    with (
        patch("src.pipeline.retrain.load_raw", return_value=_make_df(200)),
        patch("src.pipeline.retrain.fit"),
        patch("src.pipeline.retrain.predict_proba", return_value=np.random.default_rng(0).random(30)),
        patch("src.pipeline.retrain.compute_metrics", return_value=better_metrics),
        patch("src.pipeline.retrain.mlflow", mock_mlflow),
        patch("src.pipeline.retrain.torch.save"),
        patch("src.pipeline.retrain.joblib.dump"),
    ):
        exit_code = run_retrain("fake/data.csv", artifacts_dir)

    assert exit_code == 0
    saved_config = json.loads((artifacts_dir / "model_config.json").read_text())
    assert saved_config["auc_roc"] == 0.90
    mock_mlflow.set_tag.assert_called_with("promoted", "true")


@pytest.mark.unit
def test_run_retrain_rejects_when_new_model_is_worse(artifacts_dir: Path) -> None:
    """Exit 2 e artefatos inalterados quando AUC novo (0.70) < atual (0.80)."""
    worse_metrics = {
        "auc_roc": 0.70, "pr_auc": 0.50, "f1": 0.48,
        "precision": 0.45, "recall": 0.60,
    }
    original_config_bytes = (artifacts_dir / "model_config.json").read_bytes()

    mock_mlflow = MagicMock()
    mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=MagicMock())
    mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

    with (
        patch("src.pipeline.retrain.load_raw", return_value=_make_df(200)),
        patch("src.pipeline.retrain.fit"),
        patch("src.pipeline.retrain.predict_proba", return_value=np.random.default_rng(1).random(30)),
        patch("src.pipeline.retrain.compute_metrics", return_value=worse_metrics),
        patch("src.pipeline.retrain.mlflow", mock_mlflow),
        patch("src.pipeline.retrain.torch.save") as mock_torch_save,
        patch("src.pipeline.retrain.joblib.dump") as mock_joblib_dump,
    ):
        exit_code = run_retrain("fake/data.csv", artifacts_dir)

    assert exit_code == 2
    mock_torch_save.assert_not_called()
    mock_joblib_dump.assert_not_called()
    assert (artifacts_dir / "model_config.json").read_bytes() == original_config_bytes
    mock_mlflow.set_tag.assert_called_with("promoted", "false")


@pytest.mark.unit
def test_run_retrain_preserves_threshold_when_rejected(artifacts_dir: Path) -> None:
    """Threshold do config atual é mantido quando modelo é rejeitado."""
    worse_metrics = {
        "auc_roc": 0.70, "pr_auc": 0.50, "f1": 0.48,
        "precision": 0.45, "recall": 0.60,
    }
    mock_mlflow = MagicMock()
    mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=MagicMock())
    mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

    with (
        patch("src.pipeline.retrain.load_raw", return_value=_make_df(200)),
        patch("src.pipeline.retrain.fit"),
        patch("src.pipeline.retrain.predict_proba", return_value=np.random.default_rng(2).random(30)),
        patch("src.pipeline.retrain.compute_metrics", return_value=worse_metrics),
        patch("src.pipeline.retrain.mlflow", mock_mlflow),
        patch("src.pipeline.retrain.torch.save"),
        patch("src.pipeline.retrain.joblib.dump"),
    ):
        run_retrain("fake/data.csv", artifacts_dir)

    remaining_config = json.loads((artifacts_dir / "model_config.json").read_text())
    assert remaining_config["threshold"] == 0.35


@pytest.mark.unit
def test_run_retrain_logs_mlflow_run_in_both_outcomes(artifacts_dir: Path) -> None:
    """MLflow deve ser chamado independentemente de promoção ou rejeição."""
    for metrics, expected_tag in [
        ({"auc_roc": 0.90, "pr_auc": 0.65, "f1": 0.60, "precision": 0.55, "recall": 0.82}, "true"),
        ({"auc_roc": 0.70, "pr_auc": 0.50, "f1": 0.48, "precision": 0.45, "recall": 0.60}, "false"),
    ]:
        mock_mlflow = MagicMock()
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        with (
            patch("src.pipeline.retrain.load_raw", return_value=_make_df(200)),
            patch("src.pipeline.retrain.fit"),
            patch("src.pipeline.retrain.predict_proba", return_value=np.random.default_rng(3).random(30)),
            patch("src.pipeline.retrain.compute_metrics", return_value=metrics),
            patch("src.pipeline.retrain.mlflow", mock_mlflow),
            patch("src.pipeline.retrain._save_artifacts"),
        ):
            run_retrain("fake/data.csv", artifacts_dir)

        mock_mlflow.set_experiment.assert_called_once()
        mock_mlflow.start_run.assert_called_once()
        mock_mlflow.set_tag.assert_called_with("promoted", expected_tag)


@pytest.mark.slow
def test_run_retrain_end_to_end_with_real_data(artifacts_dir: Path, raw_csv_path: Path) -> None:
    """Pipeline completo sem mocks — usa CSV sintético do conftest.

    Verifica apenas que a execução termina sem erro e que model_config.json
    é um JSON válido ao final (independentemente de promoção ou rejeição).
    """
    with patch("src.pipeline.retrain.mlflow") as mock_mlflow:
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)
        exit_code = run_retrain(raw_csv_path, artifacts_dir, experiment_name="test-retrain")

    assert exit_code in (0, 2)
    final_config = json.loads((artifacts_dir / "model_config.json").read_text())
    assert "auc_roc" in final_config
    assert "threshold" in final_config
    assert "hidden_dims" in final_config
