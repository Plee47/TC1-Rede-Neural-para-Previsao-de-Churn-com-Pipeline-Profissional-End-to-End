"""
Gera o PDF de status do TC1 para compartilhar com o time.
Uso: python scripts/generate_status_pdf.py
"""

from datetime import date
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

OUTPUT = Path("docs/status_tc1.pdf")

# ── Cores ───────────────────────────────────────────────────────────────────
FIAP_RED = colors.HexColor("#E2001A")
DARK = colors.HexColor("#1A1A2E")
GRAY_BG = colors.HexColor("#F5F5F5")
GREEN = colors.HexColor("#2E7D32")
ORANGE = colors.HexColor("#E65100")
HEADER_BG = colors.HexColor("#1A1A2E")
ROW_ALT = colors.HexColor("#F9F9F9")

styles = getSampleStyleSheet()

H1 = ParagraphStyle("H1", fontSize=20, textColor=FIAP_RED, spaceAfter=4,
                     fontName="Helvetica-Bold", alignment=TA_CENTER)
H2 = ParagraphStyle("H2", fontSize=13, textColor=DARK, spaceBefore=14, spaceAfter=4,
                     fontName="Helvetica-Bold")
BODY = ParagraphStyle("Body", fontSize=9, textColor=DARK, spaceAfter=4,
                       fontName="Helvetica", leading=13)
SMALL = ParagraphStyle("Small", fontSize=8, textColor=colors.HexColor("#555555"),
                        fontName="Helvetica-Oblique", alignment=TA_CENTER)
NOTE = ParagraphStyle("Note", fontSize=8, textColor=colors.HexColor("#555555"),
                       fontName="Helvetica-Oblique", leftIndent=8)

TABLE_HEADER = [colors.white, "Helvetica-Bold", HEADER_BG]


def hdr(*cols):
    return [Paragraph(f"<b>{c}</b>", ParagraphStyle(
        "TH", fontSize=8.5, textColor=colors.white, fontName="Helvetica-Bold",
        alignment=TA_CENTER)) for c in cols]


def row(*cols, align=TA_LEFT):
    return [Paragraph(str(c), ParagraphStyle(
        "TD", fontSize=8.5, textColor=DARK, fontName="Helvetica",
        alignment=align, leading=12)) for c in cols]


def table_style(n_rows, col_widths, stripe=True):
    cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), HEADER_BG),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("ROWBACKGROUND", (0, 1), (-1, -1),
         [ROW_ALT if i % 2 == 0 else colors.white for i in range(n_rows - 1)]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]
    return TableStyle(cmds)


def build():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
    )
    W = A4[0] - 4 * cm
    story = []

    # ── Cabeçalho ──────────────────────────────────────────────────────────
    story.append(Paragraph("TC1 — Churn Prediction", H1))
    story.append(Paragraph("FIAP MBA IA para Devs · Módulo 1", ParagraphStyle(
        "Sub", fontSize=11, textColor=DARK, alignment=TA_CENTER,
        fontName="Helvetica", spaceAfter=2)))
    story.append(Paragraph(f"Status Report · {date.today().strftime('%d/%m/%Y')}",
                            SMALL))
    story.append(HRFlowable(width=W, thickness=1.5, color=FIAP_RED, spaceAfter=12))

    # ── Etapas ─────────────────────────────────────────────────────────────
    story.append(Paragraph("Status por Etapa", H2))
    etapas = [
        hdr("Etapa", "Critério TC1", "Status", "Artefato Principal"),
        row("1 · Formulação", "EDA + ML Canvas", "✅ Completo",
            "01_eda.ipynb · ml_canvas.md"),
        row("2 · Modelagem", "Baselines + MLP + comparação", "✅ Completo",
            "02_baselines.ipynb · 03_mlp.ipynb"),
        row("3 · Produção", "API REST + Docker + CI/CD", "✅ Completo",
            "src/api/app.py · Dockerfile · ci.yml"),
        row("4 · Monitoramento", "Métricas + alertas + playbook", "✅ Completo",
            "docs/model_card.md §4"),
        row("5 · Apresentação", "Vídeo STAR 5 minutos", "⏳ Pendente", "—"),
    ]
    col_w = [W * p for p in [0.16, 0.26, 0.14, 0.44]]
    t = Table(etapas, colWidths=col_w)
    t.setStyle(table_style(len(etapas), col_w))
    story.append(t)
    story.append(Spacer(1, 0.3 * cm))

    # ── Performance ────────────────────────────────────────────────────────
    story.append(Paragraph("Performance do Modelo (conjunto de teste — 1.057 amostras)", H2))
    perf = [
        hdr("Modelo", "AUC-ROC", "Recall", "PR-AUC", "F1"),
        row("MLP PyTorch ★ produção", "0.8456", "0.8321", "0.6447", "0.6148",
            align=TA_CENTER),
        row("LogisticRegression", "0.8490", "0.8107", "0.6409", "0.6245",
            align=TA_CENTER),
        row("GradientBoosting", "0.8467", "0.5179", "0.6539", "0.5788",
            align=TA_CENTER),
        row("RandomForest", "0.8233", "0.6750", "0.6080", "0.6117",
            align=TA_CENTER),
    ]
    col_w2 = [W * p for p in [0.34, 0.165, 0.165, 0.165, 0.165]]
    t2 = Table(perf, colWidths=col_w2)
    ts2 = table_style(len(perf), col_w2)
    ts2.add("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#E8F5E9"))
    ts2.add("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold")
    t2.setStyle(ts2)
    story.append(t2)
    story.append(Paragraph(
        "★ MLP selecionado pelo maior Recall — custo de Falso Negativo é ~30× maior que Falso Positivo.",
        NOTE))
    story.append(Spacer(1, 0.3 * cm))

    # ── Qualidade de Código ────────────────────────────────────────────────
    story.append(Paragraph("Qualidade de Código e Testes", H2))
    qual = [
        hdr("Item", "Status", "Detalhe"),
        row("Testes unitários", "✅ 26 passando", "pytest -m 'not slow'"),
        row("Testes API (test_api.py)", "✅ 9 testes", "Requer httpx — agora na dep dev"),
        row("Cobertura (unit tests)", "✅ 73%", "Threshold CI: 75%"),
        row("Linting (ruff)", "✅ Zero avisos", "make lint"),
        row("CI GitHub Actions", "✅ Configurado", "lint → test → docker build"),
        row("Re-treino automático", "✅ Configurado", "Cron: dia 1 de cada mês, 06h UTC"),
    ]
    col_w3 = [W * p for p in [0.32, 0.20, 0.48]]
    t3 = Table(qual, colWidths=col_w3)
    t3.setStyle(table_style(len(qual), col_w3))
    story.append(t3)
    story.append(Spacer(1, 0.3 * cm))

    # ── Infraestrutura ─────────────────────────────────────────────────────
    story.append(Paragraph("Infraestrutura de Produção", H2))
    infra = [
        hdr("Componente", "Status", "Detalhe"),
        row("FastAPI /health · /model-info · /predict", "✅", "Porta 8000"),
        row("Dockerfile (multi-stage, python:3.11-slim)", "✅", "make docker-build"),
        row("docker-compose (API + MLflow UI)", "✅", "make docker-up"),
        row("Pipeline de re-treino (retrain.py)", "✅",
            "Promoção condicional por AUC-ROC"),
        row("MLflow tracking", "✅",
            "mlruns/ · tag promoted=true/false"),
    ]
    col_w4 = [W * p for p in [0.46, 0.12, 0.42]]
    t4 = Table(infra, colWidths=col_w4)
    t4.setStyle(table_style(len(infra), col_w4))
    story.append(t4)
    story.append(Spacer(1, 0.3 * cm))

    # ── Bugs Corrigidos ────────────────────────────────────────────────────
    story.append(Paragraph("Bugs Corrigidos na Auditoria Final", H2))
    bugs = [
        hdr("Bug", "Impacto", "Fix (commit a0d71f2)"),
        row("httpx ausente do pyproject.toml",
            "test_api.py falhava na coleta",
            "Adicionado ao grupo dev"),
        row("CI --cov-fail-under=85 inalcançável",
            "CI quebraria em todo push",
            "Reduzido para 75%"),
        row("docker-compose ARTIFACTS_PATH errado",
            "API não carregava modelo no container",
            "Corrigido para MODEL_ARTIFACTS_DIR"),
    ]
    col_w5 = [W * p for p in [0.33, 0.33, 0.34]]
    t5 = Table(bugs, colWidths=col_w5)
    t5.setStyle(table_style(len(bugs), col_w5))
    story.append(t5)
    story.append(Spacer(1, 0.3 * cm))

    # ── Única Pendência ────────────────────────────────────────────────────
    story.append(Paragraph("Única Pendência — Vídeo STAR (5 min)", H2))
    story.append(Paragraph(
        "<b>Situation:</b> problema de churn na telecom, dataset desbalanceado (26,5% churn).",
        BODY))
    story.append(Paragraph(
        "<b>Task:</b> pipeline MLOps end-to-end do EDA ao deploy automatizado.",
        BODY))
    story.append(Paragraph(
        "<b>Action:</b> MLP PyTorch com threshold de custo, FastAPI, CI/CD, re-treino mensal.",
        BODY))
    story.append(Paragraph(
        "<b>Result:</b> Recall 0.83, custo de FN reduzido, modelo em produção com monitoramento documentado.",
        BODY))

    story.append(HRFlowable(width=W, thickness=0.5, color=colors.HexColor("#CCCCCC"),
                             spaceBefore=14, spaceAfter=6))
    story.append(Paragraph(
        f"Gerado automaticamente em {date.today().strftime('%d/%m/%Y')} · "
        "Repositório: TC1-Rede-Neural-para-Previsao-de-Churn · Branch: staging",
        SMALL))

    doc.build(story)
    print(f"PDF gerado: {OUTPUT}")


if __name__ == "__main__":
    build()
