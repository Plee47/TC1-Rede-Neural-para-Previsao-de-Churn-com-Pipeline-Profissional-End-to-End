"""
scripts/quality_gate.py

Portão de qualidade do retreino: valida as métricas dos artefatos
recém-exportados (models/artifacts/model_config.json) contra os SLOs
definidos em docs/ml_canvas.md. Sai com código 1 se qualquer métrica
ficar abaixo da meta — o que bloqueia o pipeline antes do deploy.

Uso:
    python scripts/quality_gate.py            # valida (CI / local)
    python scripts/quality_gate.py --pr-body  # imprime markdown para o corpo do PR
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Metas mínimas — fonte: docs/ml_canvas.md, seção SLOs.
GATES = {
    "test_auc_roc": 0.78,
    "test_pr_auc": 0.60,
    "test_recall": 0.70,
}

LABELS = {
    "test_auc_roc": "AUC-ROC",
    "test_pr_auc": "PR-AUC",
    "test_recall": "Recall",
}


def load_config() -> dict:
    artifacts_dir = Path(os.environ.get("MODEL_ARTIFACTS_DIR", "models/artifacts"))
    config_path = artifacts_dir / "model_config.json"
    if not config_path.exists():
        print(f"ERRO: {config_path} não encontrado — rode o export primeiro.")
        sys.exit(1)
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


def evaluate(config: dict) -> tuple[list[tuple[str, float, float, bool]], bool]:
    rows = []
    all_ok = True
    for key, minimum in GATES.items():
        value = config.get(key)
        ok = value is not None and value >= minimum
        all_ok = all_ok and ok
        rows.append((LABELS[key], value, minimum, ok))
    return rows, all_ok


def print_report(rows: list, all_ok: bool) -> None:
    print("Portão de qualidade — SLOs do ML Canvas")
    print("-" * 46)
    for label, value, minimum, ok in rows:
        status = "OK  " if ok else "FALHA"
        shown = f"{value:.4f}" if value is not None else "ausente"
        print(f"  {status}  {label:<8} {shown}  (meta >= {minimum})")
    print("-" * 46)
    print("RESULTADO:", "aprovado" if all_ok else "REPROVADO")


def print_pr_body(config: dict, rows: list, all_ok: bool) -> None:
    print("## Retreino automático do modelo de churn")
    print()
    print("Artefatos regenerados por `src/models/export_artifacts.py`.")
    print()
    print("| Métrica (teste) | Valor | Meta (SLO) | Status |")
    print("|---|---|---|---|")
    for label, value, minimum, ok in rows:
        shown = f"{value:.4f}" if value is not None else "ausente"
        print(f"| {label} | {shown} | ≥ {minimum} | {'✅' if ok else '❌'} |")
    print()
    print(f"- Threshold exportado: `{config.get('threshold')}`")
    print(f"- Melhor AUC de validação: `{config.get('best_val_auc')}`")
    print(f"- Dimensão de entrada: `{config.get('input_dim')}`")
    print()
    print(
        "Smoke test de integração executado no pipeline "
        "(API carrega os artefatos e discrimina alto × baixo risco)."
    )


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    config = load_config()
    rows, all_ok = evaluate(config)

    if "--pr-body" in sys.argv:
        print_pr_body(config, rows, all_ok)
    else:
        print_report(rows, all_ok)

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
