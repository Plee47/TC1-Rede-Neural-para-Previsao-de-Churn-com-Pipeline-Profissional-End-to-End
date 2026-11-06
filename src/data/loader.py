"""
src/data/loader.py
Carregamento e limpeza inicial do dataset Telco Customer Churn.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

TARGET_COL = "Churn"
DROP_COLS = ["customerID"]


def load_raw(path: str) -> pd.DataFrame:
    """
    Carrega o CSV bruto e aplica correções básicas de tipo.

    Tratamentos aplicados:
    - TotalCharges convertido para float (vem como string com espaços)
    - 11 NaN preenchidos com mediana (clientes com tenure=0)
    - Target binarizado: Yes -> 1, No -> 0

    Args:
        path: caminho para o arquivo CSV

    Returns:
        DataFrame limpo e pronto para EDA / pré-processamento
    """
    logger.info("Carregando dataset de %s", path)
    df = pd.read_csv(path)
    logger.info("Shape bruto: %s", df.shape)

    # TotalCharges: string com espaços em branco em 11 clientes (tenure=0)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    n_nan = df["TotalCharges"].isnull().sum()
    if n_nan > 0:
        mediana = df["TotalCharges"].median()
        df["TotalCharges"] = df["TotalCharges"].fillna(mediana)
        logger.info(
            "TotalCharges: %d NaN preenchidos com mediana (%.2f)", n_nan, mediana
        )

    # Binarizar target
    df["Churn_bin"] = (df[TARGET_COL] == "Yes").astype(int)

    # Remover duplicatas
    n_dup = df.duplicated().sum()
    if n_dup > 0:
        df = df.drop_duplicates()
        logger.warning("Removidas %d linhas duplicadas", n_dup)

    logger.info(
        "Dataset carregado: %d registros | churn rate: %.1f%%",
        len(df),
        df["Churn_bin"].mean() * 100,
    )
    return df
