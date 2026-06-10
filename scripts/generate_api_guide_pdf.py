"""
Gera o PDF de guia de testes da API para compartilhar com o time.
Uso: python scripts/generate_api_guide_pdf.py
"""

import json
from datetime import date
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

OUTPUT = Path("docs/guia_testes_api.pdf")

# ── Cores ────────────────────────────────────────────────────────────────────
FIAP_RED   = colors.HexColor("#E2001A")
DARK       = colors.HexColor("#1A1A2E")
CODE_BG    = colors.HexColor("#F0F0F0")
CODE_DARK  = colors.HexColor("#1E1E2E")
GREEN_BG   = colors.HexColor("#E8F5E9")
GREEN_TXT  = colors.HexColor("#2E7D32")
RED_BG     = colors.HexColor("#FFEBEE")
RED_TXT    = colors.HexColor("#C62828")
BLUE_BG    = colors.HexColor("#E3F2FD")
BLUE_TXT   = colors.HexColor("#1565C0")
HEADER_BG  = colors.HexColor("#1A1A2E")
STEP_BG    = colors.HexColor("#FFF8E1")
STEP_BORD  = colors.HexColor("#F9A825")

# ── Estilos ──────────────────────────────────────────────────────────────────
H1 = ParagraphStyle("H1", fontSize=20, textColor=FIAP_RED, spaceAfter=4,
                     fontName="Helvetica-Bold", alignment=TA_CENTER)
H2 = ParagraphStyle("H2", fontSize=13, textColor=DARK, spaceBefore=16, spaceAfter=6,
                     fontName="Helvetica-Bold")
H3 = ParagraphStyle("H3", fontSize=10, textColor=DARK, spaceBefore=10, spaceAfter=4,
                     fontName="Helvetica-Bold")
BODY = ParagraphStyle("Body", fontSize=9, textColor=DARK, spaceAfter=4,
                       fontName="Helvetica", leading=14)
SMALL = ParagraphStyle("Small", fontSize=8, textColor=colors.HexColor("#555555"),
                        fontName="Helvetica-Oblique", alignment=TA_CENTER)
CODE = ParagraphStyle("Code", fontSize=8, textColor=CODE_DARK, fontName="Courier",
                       leading=12, leftIndent=0, spaceAfter=0)
NOTE_OK  = ParagraphStyle("NoteOK",  fontSize=8.5, textColor=GREEN_TXT,
                           fontName="Helvetica-Bold", leading=13)
NOTE_ERR = ParagraphStyle("NoteErr", fontSize=8.5, textColor=RED_TXT,
                           fontName="Helvetica-Bold", leading=13)
NOTE_INFO = ParagraphStyle("NoteInfo", fontSize=8.5, textColor=BLUE_TXT,
                            fontName="Helvetica", leading=13)
BADGE = ParagraphStyle("Badge", fontSize=8, textColor=colors.white,
                        fontName="Helvetica-Bold", alignment=TA_CENTER)
URL_STYLE = ParagraphStyle("URL", fontSize=10, textColor=FIAP_RED,
                            fontName="Helvetica-Bold", alignment=TA_CENTER,
                            spaceBefore=4, spaceAfter=4)


def hr(spacebefore=8, spaceafter=8, color=colors.HexColor("#DDDDDD"), thick=0.5):
    return HRFlowable(width="100%", thickness=thick, color=color,
                      spaceBefore=spacebefore, spaceAfter=spaceafter)


def code_block(json_obj_or_str, W):
    """Renderiza um bloco de código JSON com fundo cinza."""
    if isinstance(json_obj_or_str, dict):
        text = json.dumps(json_obj_or_str, indent=2, ensure_ascii=False)
    else:
        text = json_obj_or_str
    lines = [Paragraph(line.replace(" ", "&nbsp;").replace("<", "&lt;"),
                        CODE) for line in text.split("\n")]
    inner = Table([[l] for l in lines], colWidths=[W - 1.2 * cm])
    inner.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), CODE_BG),
        ("LEFTPADDING",  (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING",   (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 2),
    ]))
    return inner


def badge_table(method, path, W, method_color):
    method_cell = Paragraph(method, BADGE)
    path_cell   = Paragraph(f"<b>{path}</b>", ParagraphStyle(
        "Path", fontSize=10, textColor=DARK, fontName="Courier-Bold", leading=13))
    t = Table([[method_cell, path_cell]], colWidths=[1.5 * cm, W - 1.5 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (0, 0), method_color),
        ("BACKGROUND",   (1, 0), (1, 0), colors.HexColor("#F7F7F7")),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",   (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 6),
        ("LEFTPADDING",  (0, 0), (-1, -1), 8),
        ("BOX",          (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
    ]))
    return t


def result_box(text, style, bg):
    t = Table([[Paragraph(text, style)]], colWidths=["100%"])
    t.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, -1), bg),
        ("LEFTPADDING",  (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING",   (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 6),
        ("BOX",          (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
    ]))
    return t


def step_box(number, title, W):
    inner = Table([[
        Paragraph(str(number), ParagraphStyle("SN", fontSize=14, textColor=FIAP_RED,
                                               fontName="Helvetica-Bold",
                                               alignment=TA_CENTER)),
        Paragraph(title, ParagraphStyle("ST", fontSize=11, textColor=DARK,
                                         fontName="Helvetica-Bold", leading=14)),
    ]], colWidths=[0.8 * cm, W - 0.8 * cm])
    inner.setStyle(TableStyle([
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("BACKGROUND",   (0, 0), (-1, -1), STEP_BG),
        ("BOX",          (0, 0), (-1, -1), 1.5, STEP_BORD),
        ("LEFTPADDING",  (0, 0), (-1, -1), 8),
        ("TOPPADDING",   (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 6),
    ]))
    return inner


# ── Payloads ─────────────────────────────────────────────────────────────────
PAYLOAD_HIGH_RISK = {
    "tenure": 2,
    "monthly_charges": 85.50,
    "total_charges": 171.00,
    "gender": "Male",
    "senior_citizen": 0,
    "partner": "No",
    "dependents": "No",
    "phone_service": "Yes",
    "multiple_lines": "No",
    "internet_service": "Fiber optic",
    "online_security": "No",
    "online_backup": "No",
    "device_protection": "No",
    "tech_support": "No",
    "streaming_tv": "Yes",
    "streaming_movies": "Yes",
    "contract": "Month-to-month",
    "paperless_billing": "Yes",
    "payment_method": "Electronic check",
}

PAYLOAD_LOW_RISK = {
    "tenure": 60,
    "monthly_charges": 55.00,
    "total_charges": 3300.00,
    "gender": "Female",
    "senior_citizen": 0,
    "partner": "Yes",
    "dependents": "Yes",
    "phone_service": "Yes",
    "multiple_lines": "No",
    "internet_service": "DSL",
    "online_security": "Yes",
    "online_backup": "Yes",
    "device_protection": "Yes",
    "tech_support": "Yes",
    "streaming_tv": "No",
    "streaming_movies": "No",
    "contract": "Two year",
    "paperless_billing": "No",
    "payment_method": "Bank transfer (automatic)",
}

RESPONSE_OK = '{\n  "status": "ok",\n  "version": "0.1.0"\n}'
RESPONSE_MODEL_INFO = (
    '{\n  "model_loaded": true,\n  "input_dim": 46,\n'
    '  "hidden_dims": [128, 64, 32],\n  "threshold": 0.35,\n'
    '  "version": "0.1.0"\n}'
)
RESPONSE_HIGH = (
    '{\n  "churn_probability": 0.823456,\n'
    '  "churn_prediction": 1,\n'
    '  "threshold": 0.35\n}'
)
RESPONSE_LOW = (
    '{\n  "churn_probability": 0.071234,\n'
    '  "churn_prediction": 0,\n'
    '  "threshold": 0.35\n}'
)
RESPONSE_422 = (
    '{\n  "detail": [\n    {\n'
    '      "type": "missing",\n'
    '      "loc": ["body", "tenure"],\n'
    '      "msg": "Field required"\n'
    '    }\n  ]\n}'
)


# ── Build ─────────────────────────────────────────────────────────────────────
def build():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
    )
    W = A4[0] - 4 * cm
    s = []

    # ── Capa ─────────────────────────────────────────────────────────────────
    s.append(Paragraph("Guia de Testes — Churn Prediction API", H1))
    s.append(Paragraph("FIAP MBA IA para Devs · TC1 · Módulo 1", ParagraphStyle(
        "Sub", fontSize=11, textColor=DARK, alignment=TA_CENTER,
        fontName="Helvetica", spaceAfter=2)))
    s.append(Paragraph(f"Versão 0.1.0 · {date.today().strftime('%d/%m/%Y')}",
                        SMALL))
    s.append(hr(spacebefore=10, spaceafter=14, color=FIAP_RED, thick=1.5))

    # ── Pré-requisito ─────────────────────────────────────────────────────────
    s.append(Paragraph("Pré-requisito: subir a API", H2))
    s.append(Paragraph(
        "Antes de testar, garanta que o servidor está rodando localmente. "
        "No terminal, dentro da pasta do projeto:", BODY))
    s.append(code_block("python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload", W))
    s.append(Spacer(1, 0.3 * cm))
    s.append(Paragraph("Acesse o Swagger UI no browser:", BODY))
    s.append(Paragraph("http://localhost:8000/docs", URL_STYLE))
    s.append(Paragraph("Documentação alternativa (ReDoc):", BODY))
    s.append(Paragraph("http://localhost:8000/redoc", URL_STYLE))
    s.append(Spacer(1, 0.2 * cm))
    s.append(result_box(
        "ℹ️  No Swagger: clique no endpoint → clique em \"Try it out\" → "
        "preencha o body → clique em \"Execute\".",
        NOTE_INFO, BLUE_BG))
    s.append(hr())

    # ── Teste 1 ───────────────────────────────────────────────────────────────
    s.append(step_box(1, "Health Check  ·  GET /health", W))
    s.append(Spacer(1, 0.2 * cm))
    s.append(Paragraph("Confirma que a API está no ar e com a versão correta.", BODY))
    s.append(badge_table("GET", "/health", W, GREEN_TXT))
    s.append(Spacer(1, 0.2 * cm))
    s.append(Paragraph("<b>Nenhum parâmetro necessário.</b> Clique em Execute.", BODY))
    s.append(Spacer(1, 0.15 * cm))
    s.append(Paragraph("<b>Resposta esperada  (HTTP 200):</b>", BODY))
    s.append(code_block(RESPONSE_OK, W))
    s.append(Spacer(1, 0.2 * cm))
    s.append(result_box("✅  Critério de aceite: status = \"ok\" e HTTP 200.", NOTE_OK, GREEN_BG))
    s.append(hr())

    # ── Teste 2 ───────────────────────────────────────────────────────────────
    s.append(step_box(2, "Info do Modelo  ·  GET /model-info", W))
    s.append(Spacer(1, 0.2 * cm))
    s.append(Paragraph(
        "Confirma que os artefatos (pesos, preprocessador, config) carregaram corretamente.", BODY))
    s.append(badge_table("GET", "/model-info", W, GREEN_TXT))
    s.append(Spacer(1, 0.2 * cm))
    s.append(Paragraph("<b>Nenhum parâmetro necessário.</b> Clique em Execute.", BODY))
    s.append(Spacer(1, 0.15 * cm))
    s.append(Paragraph("<b>Resposta esperada  (HTTP 200):</b>", BODY))
    s.append(code_block(RESPONSE_MODEL_INFO, W))
    s.append(Spacer(1, 0.2 * cm))
    s.append(result_box(
        "✅  Critério de aceite: model_loaded = true, input_dim = 46, threshold = 0.35.",
        NOTE_OK, GREEN_BG))
    s.append(result_box(
        "⚠️  Se model_loaded = false: artefatos não encontrados em models/artifacts/ "
        "— execute o notebook 03_mlp.ipynb primeiro.",
        NOTE_ERR, RED_BG))
    s.append(hr())

    # ── Teste 3A ──────────────────────────────────────────────────────────────
    s.append(step_box(3, "Predição — Cliente de ALTO Risco  ·  POST /predict", W))
    s.append(Spacer(1, 0.2 * cm))
    s.append(Paragraph(
        "Perfil clássico de churn: contrato mensal, 2 meses de casa, fibra ótica, "
        "sem serviços de suporte, boleto eletrônico.", BODY))
    s.append(badge_table("POST", "/predict", W, colors.HexColor("#E65100")))
    s.append(Spacer(1, 0.2 * cm))
    s.append(Paragraph("<b>Body (JSON) — copie e cole no campo Request body:</b>", BODY))
    s.append(code_block(PAYLOAD_HIGH_RISK, W))
    s.append(Spacer(1, 0.2 * cm))
    s.append(Paragraph("<b>Resposta esperada  (HTTP 200):</b>", BODY))
    s.append(code_block(RESPONSE_HIGH, W))
    s.append(Spacer(1, 0.2 * cm))
    s.append(result_box(
        "✅  Critério de aceite: churn_prediction = 1  e  churn_probability > 0.60.",
        NOTE_OK, GREEN_BG))
    s.append(hr())

    # ── Teste 3B ──────────────────────────────────────────────────────────────
    s.append(step_box(4, "Predição — Cliente de BAIXO Risco  ·  POST /predict", W))
    s.append(Spacer(1, 0.2 * cm))
    s.append(Paragraph(
        "Perfil fiel: contrato bianual, 60 meses de casa, DSL, todos os serviços de suporte "
        "ativos, débito automático.", BODY))
    s.append(badge_table("POST", "/predict", W, colors.HexColor("#E65100")))
    s.append(Spacer(1, 0.2 * cm))
    s.append(Paragraph("<b>Body (JSON):</b>", BODY))
    s.append(code_block(PAYLOAD_LOW_RISK, W))
    s.append(Spacer(1, 0.2 * cm))
    s.append(Paragraph("<b>Resposta esperada  (HTTP 200):</b>", BODY))
    s.append(code_block(RESPONSE_LOW, W))
    s.append(Spacer(1, 0.2 * cm))
    s.append(result_box(
        "✅  Critério de aceite: churn_prediction = 0  e  churn_probability < 0.20.",
        NOTE_OK, GREEN_BG))
    s.append(hr())

    # ── Teste 4 — Campo Faltando ──────────────────────────────────────────────
    s.append(step_box(5, "Validação — Campo Obrigatório Faltando  ·  POST /predict", W))
    s.append(Spacer(1, 0.2 * cm))
    s.append(Paragraph(
        "Verifica que a API rejeita payloads incompletos com a mensagem correta.", BODY))
    s.append(badge_table("POST", "/predict", W, colors.HexColor("#E65100")))
    s.append(Spacer(1, 0.2 * cm))
    s.append(Paragraph(
        "<b>Use o mesmo payload do Teste 3 (alto risco) e remova o campo <font name='Courier'>tenure</font>.</b>",
        BODY))
    s.append(Spacer(1, 0.15 * cm))
    s.append(Paragraph("<b>Resposta esperada  (HTTP 422):</b>", BODY))
    s.append(code_block(RESPONSE_422, W))
    s.append(Spacer(1, 0.2 * cm))
    s.append(result_box(
        "✅  Critério de aceite: HTTP 422 com \"msg\": \"Field required\" para tenure.",
        NOTE_OK, GREEN_BG))
    s.append(hr())

    # ── Teste 5 — Valor Inválido ──────────────────────────────────────────────
    s.append(step_box(6, "Validação — Valor Inválido  ·  POST /predict", W))
    s.append(Spacer(1, 0.2 * cm))
    s.append(Paragraph(
        "Verifica que a API rejeita valores fora do range permitido.", BODY))
    s.append(badge_table("POST", "/predict", W, colors.HexColor("#E65100")))
    s.append(Spacer(1, 0.2 * cm))
    s.append(Paragraph(
        "<b>Use o payload do Teste 3 (alto risco) e altere:</b>", BODY))
    s.append(code_block('  "tenure": -5,   ← valor negativo, mínimo é 0', W))
    s.append(Spacer(1, 0.15 * cm))
    s.append(result_box(
        "✅  Critério de aceite: HTTP 422 com erro de validação em tenure.",
        NOTE_OK, GREEN_BG))
    s.append(hr())

    # ── Tabela Resumo ─────────────────────────────────────────────────────────
    s.append(Paragraph("Resumo dos Testes", H2))

    def th(*cols):
        return [Paragraph(f"<b>{c}</b>", ParagraphStyle(
            "TH", fontSize=8, textColor=colors.white,
            fontName="Helvetica-Bold", alignment=TA_CENTER)) for c in cols]

    def td(*cols):
        return [Paragraph(str(c), ParagraphStyle(
            "TD", fontSize=8, textColor=DARK,
            fontName="Helvetica", alignment=TA_CENTER, leading=12)) for c in cols]

    rows = [
        th("#", "Método", "Endpoint", "Cenário", "HTTP Esperado", "Critério"),
        td("1", "GET",  "/health",     "API no ar",             "200", "status = ok"),
        td("2", "GET",  "/model-info", "Modelo carregado",      "200", "model_loaded = true"),
        td("3", "POST", "/predict",    "Alto risco de churn",   "200", "prediction = 1, prob > 0.60"),
        td("4", "POST", "/predict",    "Baixo risco de churn",  "200", "prediction = 0, prob < 0.20"),
        td("5", "POST", "/predict",    "Campo faltando",        "422", "msg: Field required"),
        td("6", "POST", "/predict",    "Valor negativo",        "422", "erro em tenure"),
    ]
    col_w = [W * p for p in [0.05, 0.08, 0.17, 0.27, 0.17, 0.26]]
    t = Table(rows, colWidths=col_w)
    t.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0), HEADER_BG),
        ("ROWBACKGROUND",(0, 1), (-1, -1),
         [colors.HexColor("#F9F9F9") if i % 2 == 0 else colors.white
          for i in range(len(rows) - 1)]),
        ("GRID",         (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
        ("ALIGN",        (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",   (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
    ]))
    s.append(t)

    # ── Rodapé ────────────────────────────────────────────────────────────────
    s.append(hr(spacebefore=16, spaceafter=6, color=colors.HexColor("#CCCCCC")))
    s.append(Paragraph(
        f"TC1 Churn Prediction API · FIAP MBA IA para Devs · "
        f"Gerado em {date.today().strftime('%d/%m/%Y')}",
        SMALL))

    doc.build(s)
    print(f"PDF gerado: {OUTPUT}")


if __name__ == "__main__":
    build()
