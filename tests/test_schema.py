"""
tests/test_schema.py
Schema tests — valida contratos de dados (colunas, tipos, invariantes).
"""

import pytest

EXPECTED_COLUMNS = {
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
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
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
    "Churn_bin",
}


@pytest.mark.unit
def test_output_has_expected_columns(loaded_df):
    """load_raw deve preservar todas as colunas originais e adicionar Churn_bin."""
    assert EXPECTED_COLUMNS.issubset(set(loaded_df.columns))


@pytest.mark.unit
def test_churn_bin_is_binary(loaded_df):
    """Churn_bin deve conter apenas 0 e 1."""
    unique_values = set(loaded_df["Churn_bin"].unique())
    assert unique_values.issubset({0, 1})


@pytest.mark.unit
def test_no_nulls_after_load(loaded_df):
    """Não deve haver valores nulos após load_raw (NaN de TotalCharges imputados)."""
    assert loaded_df.isnull().sum().sum() == 0


@pytest.mark.unit
def test_total_charges_is_float(loaded_df):
    """TotalCharges deve ser numérico (float64) após conversão."""
    import pandas as pd

    assert pd.api.types.is_float_dtype(loaded_df["TotalCharges"])


@pytest.mark.unit
def test_tenure_non_negative(loaded_df):
    """tenure não pode ser negativo."""
    assert (loaded_df["tenure"] >= 0).all()


@pytest.mark.unit
def test_monthly_charges_positive(loaded_df):
    """MonthlyCharges deve ser positivo."""
    assert (loaded_df["MonthlyCharges"] > 0).all()


@pytest.mark.unit
def test_churn_rate_in_valid_range(loaded_df):
    """Taxa de churn deve estar entre 0% e 100%."""
    churn_rate = loaded_df["Churn_bin"].mean()
    assert 0.0 <= churn_rate <= 1.0


@pytest.mark.unit
def test_compute_metrics_threshold_sensitivity(binary_arrays):
    """Threshold menor deve produzir recall maior ou igual."""
    from src.evaluation.metrics import compute_metrics

    y_true, y_prob = binary_arrays
    metrics_05 = compute_metrics(y_true, y_prob, threshold=0.5)
    metrics_03 = compute_metrics(y_true, y_prob, threshold=0.3)

    assert metrics_03["recall"] >= metrics_05["recall"]
