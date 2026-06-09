"""
tests/conftest.py
Fixtures compartilhadas entre todos os testes.
"""

import textwrap

import numpy as np
import pandas as pd
import pytest

TELCO_CSV = textwrap.dedent("""\
    customerID,gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies,Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges,Churn
    A001,Female,0,Yes,No,12,Yes,No,DSL,Yes,No,Yes,No,No,No,One year,Yes,Bank transfer (automatic),50.0,600.0,No
    A002,Male,1,No,No,1,Yes,No,Fiber optic,No,No,No,No,Yes,Yes,Month-to-month,Yes,Electronic check,85.0, ,Yes
    A003,Female,0,Yes,Yes,60,Yes,Yes,DSL,Yes,Yes,Yes,Yes,Yes,Yes,Two year,No,Credit card (automatic),75.0,4500.0,No
    A004,Male,0,No,No,24,No,No phone service,No,No internet service,No internet service,No internet service,No internet service,No internet service,No internet service,One year,No,Mailed check,20.0,480.0,No
    A005,Male,1,Yes,No,3,Yes,Yes,Fiber optic,No,Yes,No,No,Yes,No,Month-to-month,Yes,Electronic check,90.0,270.0,Yes
    A006,Female,0,No,No,8,Yes,No,Fiber optic,No,No,No,No,No,Yes,Month-to-month,Yes,Electronic check,70.0,560.0,Yes
    A007,Male,0,Yes,Yes,45,Yes,Yes,DSL,Yes,Yes,Yes,Yes,No,No,One year,No,Bank transfer (automatic),55.0,2475.0,No
    A008,Female,1,No,No,2,Yes,No,Fiber optic,No,No,No,No,Yes,Yes,Month-to-month,Yes,Electronic check,95.0,190.0,Yes
    A009,Male,0,Yes,No,36,No,No phone service,DSL,Yes,No,Yes,No,No,No,Two year,No,Credit card (automatic),30.0,1080.0,No
    A010,Female,0,No,Yes,18,Yes,No,DSL,No,Yes,No,No,No,No,One year,Yes,Mailed check,45.0,810.0,No
    A011,Male,1,No,No,4,Yes,Yes,Fiber optic,No,No,No,No,Yes,Yes,Month-to-month,Yes,Electronic check,100.0,400.0,Yes
    A012,Female,0,Yes,No,72,Yes,Yes,DSL,Yes,Yes,Yes,Yes,Yes,Yes,Two year,No,Bank transfer (automatic),80.0,5760.0,No
    A013,Male,0,No,No,6,Yes,No,Fiber optic,No,No,No,No,No,No,Month-to-month,Yes,Electronic check,65.0,390.0,Yes
    A014,Female,1,Yes,Yes,48,Yes,No,DSL,Yes,Yes,Yes,Yes,No,Yes,Two year,No,Credit card (automatic),60.0,2880.0,No
    A015,Male,0,No,No,2,No,No phone service,No,No internet service,No internet service,No internet service,No internet service,No internet service,No internet service,Month-to-month,No,Mailed check,20.0,40.0,Yes
    A016,Female,0,Yes,No,30,Yes,Yes,Fiber optic,No,Yes,No,Yes,Yes,Yes,One year,Yes,Electronic check,95.0,2850.0,No
    A017,Male,1,No,No,1,Yes,No,Fiber optic,No,No,No,No,Yes,No,Month-to-month,Yes,Electronic check,70.0,70.0,Yes
    A018,Female,0,No,Yes,55,Yes,No,DSL,Yes,No,Yes,No,No,No,Two year,No,Bank transfer (automatic),50.0,2750.0,No
    A019,Male,0,Yes,No,15,Yes,Yes,Fiber optic,No,Yes,No,No,Yes,Yes,Month-to-month,Yes,Electronic check,88.0,1320.0,Yes
    A020,Female,0,No,No,40,Yes,No,DSL,Yes,Yes,Yes,Yes,Yes,No,One year,Yes,Credit card (automatic),65.0,2600.0,No
    A021,Male,1,No,No,3,Yes,Yes,Fiber optic,No,No,No,No,No,Yes,Month-to-month,Yes,Electronic check,85.0,255.0,Yes
    A022,Female,0,Yes,Yes,60,No,No phone service,No,No internet service,No internet service,No internet service,No internet service,No internet service,No internet service,Two year,No,Mailed check,20.0,1200.0,No
    A023,Male,0,No,No,5,Yes,No,Fiber optic,No,No,No,No,Yes,Yes,Month-to-month,Yes,Electronic check,75.0,375.0,Yes
    A024,Female,0,Yes,No,25,Yes,No,DSL,Yes,Yes,No,Yes,No,No,One year,No,Bank transfer (automatic),52.0,1300.0,No
    A025,Male,1,No,Yes,2,Yes,Yes,Fiber optic,No,No,No,No,Yes,No,Month-to-month,Yes,Electronic check,90.0,180.0,Yes
    A026,Female,0,No,No,50,Yes,Yes,DSL,Yes,Yes,Yes,Yes,Yes,Yes,Two year,No,Credit card (automatic),78.0,3900.0,No
    A027,Male,0,Yes,No,9,Yes,No,Fiber optic,No,Yes,No,No,No,Yes,Month-to-month,Yes,Electronic check,68.0,612.0,Yes
    A028,Female,1,No,Yes,66,Yes,No,DSL,Yes,No,Yes,Yes,No,No,Two year,No,Bank transfer (automatic),55.0,3630.0,No
    A029,Male,0,No,No,1,No,No phone service,No,No internet service,No internet service,No internet service,No internet service,No internet service,No internet service,Month-to-month,Yes,Mailed check,20.0,20.0,Yes
    A030,Female,0,Yes,Yes,42,Yes,Yes,DSL,Yes,Yes,Yes,No,Yes,No,One year,Yes,Credit card (automatic),70.0,2940.0,No
    A031,Male,1,No,No,4,Yes,No,Fiber optic,No,No,No,No,Yes,Yes,Month-to-month,Yes,Electronic check,92.0,368.0,Yes
    A032,Female,0,Yes,No,58,Yes,No,DSL,Yes,Yes,Yes,Yes,No,Yes,Two year,No,Bank transfer (automatic),62.0,3596.0,No
    A033,Male,0,No,No,7,Yes,Yes,Fiber optic,No,No,No,No,Yes,No,Month-to-month,Yes,Electronic check,82.0,574.0,Yes
    A034,Female,0,No,Yes,35,Yes,No,DSL,Yes,No,No,Yes,No,No,One year,No,Mailed check,47.0,1645.0,No
    A035,Male,1,Yes,No,2,Yes,No,Fiber optic,No,Yes,No,No,Yes,Yes,Month-to-month,Yes,Electronic check,89.0,178.0,Yes
    A036,Female,0,Yes,Yes,70,No,No phone service,DSL,Yes,Yes,Yes,Yes,Yes,Yes,Two year,No,Credit card (automatic),72.0,5040.0,No
    A037,Male,0,No,No,6,Yes,No,Fiber optic,No,No,No,No,No,Yes,Month-to-month,Yes,Electronic check,66.0,396.0,Yes
    A038,Female,1,Yes,No,44,Yes,Yes,DSL,Yes,Yes,Yes,Yes,Yes,No,One year,Yes,Bank transfer (automatic),74.0,3256.0,No
    A039,Male,0,No,Yes,11,Yes,No,Fiber optic,No,No,No,No,Yes,Yes,Month-to-month,Yes,Electronic check,77.0,847.0,Yes
    A040,Female,0,Yes,No,52,Yes,Yes,DSL,Yes,No,Yes,No,No,No,Two year,No,Credit card (automatic),58.0,3016.0,No
    A041,Male,1,No,No,3,Yes,No,Fiber optic,No,No,No,No,Yes,No,Month-to-month,Yes,Electronic check,84.0,252.0,Yes
    A042,Female,0,No,Yes,65,No,No phone service,No,No internet service,No internet service,No internet service,No internet service,No internet service,No internet service,Two year,No,Mailed check,20.0,1300.0,No
    A043,Male,0,Yes,No,14,Yes,Yes,Fiber optic,No,Yes,No,Yes,Yes,Yes,Month-to-month,Yes,Electronic check,96.0,1344.0,Yes
    A044,Female,0,No,No,38,Yes,No,DSL,Yes,Yes,No,No,No,Yes,One year,No,Bank transfer (automatic),48.0,1824.0,No
    A045,Male,1,Yes,No,1,Yes,No,Fiber optic,No,No,No,No,No,Yes,Month-to-month,Yes,Electronic check,73.0,73.0,Yes
    A046,Female,0,Yes,Yes,56,Yes,No,DSL,Yes,Yes,Yes,Yes,Yes,No,Two year,No,Credit card (automatic),67.0,3752.0,No
    A047,Male,0,No,No,5,Yes,Yes,Fiber optic,No,No,No,No,Yes,Yes,Month-to-month,Yes,Electronic check,91.0,455.0,Yes
    A048,Female,0,Yes,No,48,Yes,No,DSL,Yes,No,Yes,No,No,No,One year,Yes,Bank transfer (automatic),53.0,2544.0,No
    A049,Male,1,No,No,2,Yes,No,Fiber optic,No,Yes,No,No,Yes,No,Month-to-month,Yes,Electronic check,87.0,174.0,Yes
    A050,Female,0,No,Yes,62,Yes,Yes,DSL,Yes,Yes,Yes,Yes,No,Yes,Two year,No,Credit card (automatic),76.0,4712.0,No
""")


@pytest.fixture
def raw_csv_path(tmp_path) -> str:
    """Cria um CSV sintético com estrutura idêntica ao dataset Telco."""
    csv_file = tmp_path / "telco.csv"
    csv_file.write_text(TELCO_CSV, encoding="utf-8")
    return str(csv_file)


@pytest.fixture
def loaded_df(raw_csv_path) -> pd.DataFrame:
    """DataFrame já processado por load_raw."""
    from src.data.loader import load_raw
    return load_raw(raw_csv_path)


@pytest.fixture
def binary_arrays() -> tuple[np.ndarray, np.ndarray]:
    """Par (y_true, y_prob) para testar compute_metrics."""
    rng = np.random.default_rng(42)
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0])
    y_prob = rng.uniform(0, 1, size=len(y_true))
    # Garante discriminação suficiente para roc_auc_score não lançar exceção
    y_prob[y_true == 1] = np.clip(y_prob[y_true == 1] + 0.3, 0, 1)
    return y_true, y_prob
