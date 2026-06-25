# TC1 — Rede Neural para Previsão de Churn (End-to-End)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.12-orange)
![MLflow](https://img.shields.io/badge/MLflow-3.13-green)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.9-blue)
![Tests](https://img.shields.io/badge/testes-41%20passed-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-92%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

Solução para o **Tech Challenge 1 — FIAP Pós-Tech MLOps (Fase 1)**. Pipeline profissional end-to-end para previsão de churn em telecomunicações: da ingestão de dados ao endpoint de inferência com FastAPI, com rastreamento completo de experimentos via MLflow.

---

## Objetivo

Identificar clientes propensos ao cancelamento (churn) antes que ele ocorra, permitindo que equipes de CRM atuem de forma proativa e reduzam perda de receita.

**Abordagem técnica:**
- EDA e limpeza de dados com Pandas / Seaborn
- Baselines com Scikit-Learn (DummyClassifier, LogisticRegression, RandomForest, GradientBoosting)
- Rede neural MLP em PyTorch com early stopping e batching
- Rastreamento de experimentos com MLflow
- Análise de custo FP vs FN com otimização de threshold
- Endpoint de inferência com FastAPI (modelo real carregado de artefatos)
- Testes automatizados com 92% de cobertura

**Metas de performance:**
- AUC-ROC ≥ 0.78
- Recall ≥ 0.70
- Latência de API ≤ 500 ms (p95)

---

## Equipe e Responsabilidades

| Membro | Responsabilidade | Etapa |
|---|---|---|
| Tathiana Araujo Rodnarchuki | — | — |
| Giselly Kathellyn Domingos da Silva | — | — |
| Pedro Henrique Ostroski | Etapa 1 | Etapa 1 |
| Alisson Henrique Lepesqueur Borges Fabiano | Etapa 3 | Etapa 3 |
| Rafael Fernando Gimenes | Etapa 2 | Etapa 2 |

---

## Estrutura do Repositório

```
.
├── data/
│   ├── raw/                        # Dataset original (não versionado)
│   └── processed/
│       └── telco_treated.csv       # Gerado pelo 01_eda.ipynb
├── notebooks/
│   ├── 01_eda.ipynb                # EDA, limpeza e geração do dataset tratado
│   ├── 02_baselines.ipynb          # DummyClassifier + LogisticRegression + MLflow
│   └── 03_mlp.ipynb                # MLP PyTorch + árvores + comparação + custo FP/FN + exportação de artefatos
├── src/
│   ├── data/
│   │   └── loader.py               # load_raw(): carrega e limpa o CSV bruto
│   ├── models/
│   │   ├── mlp.py                  # ChurnMLP (PyTorch)
│   │   └── train.py                # fit(), predict_proba(), EarlyStopping
│   ├── evaluation/
│   │   └── metrics.py              # compute_metrics(): AUC-ROC, PR-AUC, F1, Recall
│   ├── api/
│   │   └── app.py                  # FastAPI: GET /health, GET /model-info, POST /predict
│   └── pipeline/
│       └── retrain.py              # Re-treino agendado com substituição condicional por AUC-ROC
├── tests/
│   ├── conftest.py                 # Fixtures compartilhadas (CSV sintético)
│   ├── test_smoke.py               # Smoke tests — imports e execução básica
│   ├── test_schema.py              # Schema tests — contratos de dados
│   ├── test_models.py              # Unit tests — ChurnMLP, EarlyStopping, fit()
│   ├── test_api.py                 # API tests — endpoints, validação Pydantic, mock de modelo
│   └── test_retrain.py             # Unit + slow tests — pipeline de re-treino
├── docs/
│   ├── ml_canvas.md                # ML Canvas com SLOs e análise de risco
│   ├── model_card.md               # Model Card completo (arquitetura, métricas, limitações, monitoramento)
│   ├── comparacao_modelos.png      # Gráfico comparativo (gerado pelo 03_mlp.ipynb)
│   └── analise_custo.png           # Análise de custo por threshold
├── models/
│   └── artifacts/                  # Artefatos exportados pelo 03_mlp.ipynb
│       ├── preprocessor.joblib     # Pipeline sklearn (StandardScaler + OHE)
│       ├── mlp_weights.pt          # Pesos do melhor checkpoint PyTorch
│       └── model_config.json       # Metadados: arquitetura, threshold, métricas
├── .github/
│   └── workflows/
│       ├── ci.yml                  # CI: lint → testes → docker build (push para main/staging)
│       └── retrain.yml             # Re-treino mensal agendado (cron dia 1, 06h UTC)
├── mlruns/                         # Experimentos MLflow (local)
├── Dockerfile                      # Build multi-stage para a API de inferência
├── docker-compose.yml              # Stack local: API (:8000) + MLflow UI (:5000)
├── Makefile                        # Targets: install, lint, format, test, api, docker-build, docker-up
├── .env.example                    # Variáveis de ambiente documentadas
├── pyproject.toml                  # Dependências, pytest, ruff, mypy
├── .gitignore                      # Ignora dados, modelos, venv, cache
└── .pre-commit-config.yaml         # Hooks: ruff lint+format, trailing whitespace, etc.
```

---

## Dataset

**Telco Customer Churn** — IBM / Kaggle

| Atributo | Valor |
|---|---|
| Registros | 7.043 clientes |
| Colunas | 21 (20 features + 1 target) |
| Target | `Churn` (Sim/Não — classificação binária) |
| Desbalanceamento | ~26% churn / ~74% não-churn |

**Principais colunas:**

| Coluna | Tipo | Descrição |
|---|---|---|
| `tenure` | numérico | Meses como cliente |
| `MonthlyCharges` | numérico | Cobrança mensal |
| `TotalCharges` | numérico | Cobrança total acumulada |
| `Contract` | categórico | Month-to-month / 1 year / 2 year |
| `InternetService` | categórico | DSL / Fiber optic / No |
| `Churn` | target | Cancelou o serviço? (Yes/No) |

**Principais insights (EDA):**
- Contratos mensais: ~42% churn vs. ~3% em contratos de 2 anos
- Clientes com tenure < 6 meses: ~48% churn
- Correlação `tenure` × `Churn`: −0.35

**Como obter o dataset:**

```bash
# Opção 1 — Kaggle CLI
pip install kaggle
kaggle datasets download -d blastchar/telco-customer-churn -p data/raw/ --unzip

# Opção 2 — Download manual
# https://www.kaggle.com/datasets/blastchar/telco-customer-churn
# Salvar em: data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

---

## Instalação e Setup

### Pré-requisitos
- Python 3.9 ou superior
- Git

### 1. Clone o repositório

```bash
git clone <url-do-repositorio>
cd TC1-Rede-Neural-para-Previsao-de-Churn-com-Pipeline-Profissional-End-to-End
```

### 2. Crie e ative o ambiente virtual

```bash
# Criar
python -m venv .venv

# Ativar — Windows
.venv\Scripts\activate

# Ativar — Linux / macOS
source .venv/bin/activate
```

### 3. Instale as dependências

```bash
# Instalação completa (recomendado para desenvolvimento)
pip install -e ".[all]"

# Apenas runtime (treino + inferência)
pip install -e .

# Apenas API de deploy
pip install -e ".[api]"
```

> **PyTorch no Windows/macOS sem GPU:** caso a instalação do `torch` falhe, instale separadamente via canal oficial antes de rodar o comando acima:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> pip install -e ".[all]"
> ```

### 4. (Opcional) Ative os hooks de qualidade de código

```bash
pip install pre-commit
pre-commit install
```

---

## Como Executar o Pipeline

### Ordem obrigatória de execução

```bash
# Inicie o JupyterLab
jupyter lab
```

| Ordem | Notebook | O que faz | Saída |
|---|---|---|---|
| **1º** | `notebooks/01_eda.ipynb` | EDA, limpeza, análise de correlações | `data/processed/telco_treated.csv` |
| **2º** | `notebooks/02_baselines.ipynb` | DummyClassifier + LogisticRegression + MLflow | Runs no MLflow |
| **3º** | `notebooks/03_mlp.ipynb` | MLP PyTorch + RandomForest + GradientBoosting + análise de custo + **exportação de artefatos** | Modelo + artefatos em `models/artifacts/` + runs no MLflow |

> O notebook `03_mlp.ipynb` exporta automaticamente os artefatos necessários para a API (`preprocessor.joblib`, `mlp_weights.pt`, `model_config.json`). A API só funciona com o modelo treinado.

### Executar via linha de comando (sem abrir o Jupyter)

```bash
# Notebook 1 — EDA
jupyter nbconvert --execute --to notebook --inplace notebooks/01_eda.ipynb

# Notebook 2 — Baselines
jupyter nbconvert --execute --to notebook --inplace notebooks/02_baselines.ipynb

# Notebook 3 — MLP (pode levar alguns minutos)
jupyter nbconvert --execute --to notebook --inplace notebooks/03_mlp.ipynb
```

---

## Visualizar Experimentos no MLflow

Em um terminal separado (com o venv ativado):

```bash
mlflow ui --backend-store-uri mlruns
```

Acesse: **http://localhost:5000**

O que está registrado em cada experimento:

| Experimento | Modelos | O que está logado |
|---|---|---|
| `churn-etapa1-baselines` | DummyClassifier, LogisticRegression | Params, métricas CV, curvas ROC/PR, modelo |
| `tc1-churn-etapa2` | RandomForest, GradientBoosting, MLP | Params, métricas val+test, curvas de treino, modelo |

---

## Rodar a API de Inferência

> **Pré-requisito:** os artefatos devem existir em `models/artifacts/`. Execute o `03_mlp.ipynb` primeiro.

```bash
uvicorn src.api.app:app --reload
```

Acesse: **http://localhost:8000/docs** (Swagger UI automático)

**Endpoints disponíveis:**

| Método | Rota | Descrição |
|---|---|---|
| `GET` | `/health` | Status da API |
| `GET` | `/model-info` | Metadados do modelo carregado (arquitetura, threshold) |
| `POST` | `/predict` | Retorna `churn_probability` e `churn_prediction` |

**Exemplo de chamada (cliente de alto risco):**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tenure": 3,
    "monthly_charges": 85.0,
    "total_charges": 255.0,
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
    "streaming_movies": "No",
    "contract": "Month-to-month",
    "paperless_billing": "Yes",
    "payment_method": "Electronic check"
  }'
```

**Resposta esperada:**

```json
{
  "churn_probability": 0.847312,
  "churn_prediction": 1,
  "threshold": 0.05
}
```

**Variável de ambiente para artefatos customizados:**

```bash
MODEL_ARTIFACTS_DIR=/caminho/para/artefatos uvicorn src.api.app:app --reload
```

---

## Resultados — Comparação de Modelos

Avaliação no conjunto de **teste** (1.057 amostras, nunca vistas durante treino ou validação):

| Modelo | AUC-ROC | PR-AUC | Recall | F1 | Precisão |
|---|---|---|---|---|---|
| LogisticRegression | 0.8490 | 0.6409 | 0.8107 | 0.6245 | 0.5078 |
| GradientBoosting | 0.8467 | 0.6539 | 0.5179 | 0.5788 | 0.6561 |
| **MLP PyTorch** ⭐ | **0.8456** | **0.6447** | **0.8321** | **0.6148** | **0.4874** |
| RandomForest | 0.8233 | 0.6080 | 0.6750 | 0.6117 | 0.5592 |
| DummyClassifier | 0.4759 | 0.2568 | 0.2286 | 0.2290 | 0.2294 |

> ⭐ MLP selecionado como modelo de produção pelo **maior Recall (0.8321)** — métrica prioritária dado que o custo de um FN ($300) é 30× maior que um FP ($10).

---

## Arquitetura dos Modelos

### Baselines (Scikit-Learn)

| Modelo | Configuração principal |
|---|---|
| DummyClassifier | `strategy='stratified'` |
| LogisticRegression | `C=1.0`, `class_weight='balanced'` |
| RandomForest | `n_estimators=200`, `class_weight='balanced'` |
| GradientBoosting | `n_estimators=200`, `learning_rate=0.05`, `max_depth=4` |

**Pré-processamento (compartilhado):**
- Numéricas (`tenure`, `MonthlyCharges`, `TotalCharges`): `StandardScaler`
- Categóricas (16 features): `OneHotEncoder(handle_unknown='ignore')`

### MLP (PyTorch)

```
Input (46 features após OHE)
  → Linear(128) → BatchNorm1d → ReLU → Dropout(0.3)
  → Linear(64)  → BatchNorm1d → ReLU → Dropout(0.3)
  → Linear(32)  → BatchNorm1d → ReLU → Dropout(0.3)
  → Linear(1)   → [logits]
```

| Configuração | Valor |
|---|---|
| Loss | `BCEWithLogitsLoss` com `pos_weight ≈ 2.77` |
| Otimizador | Adam, `lr=1e-3`, `weight_decay=1e-4` |
| Batch size | 256 |
| Early stopping | Paciência 15 épocas, monitora AUC-ROC de validação |
| Split | 70% treino / 15% validação / 15% teste (estratificado) |
| Parâmetros treináveis | 16.833 |

---

## Métricas de Avaliação

| Métrica | Descrição | Meta |
|---|---|---|
| **AUC-ROC** | Capacidade discriminatória geral | ≥ 0.78 ✅ |
| **PR-AUC** | Precision-Recall (robusta ao desbalanceamento) | ≥ 0.60 ✅ |
| **Recall** | Taxa de detecção de churners | ≥ 0.70 ✅ |
| **F1-Score** | Equilíbrio entre precisão e recall | — |
| **Precisão** | Proporção de alertas corretos | — |

**Análise de custo FP vs FN:**

| Erro | Situação | Custo estimado |
|---|---|---|
| Falso Positivo (FP) | Cliente contatado que não ia churnar | $10 |
| Falso Negativo (FN) | Cliente churnou sem intervenção | $300 |

O threshold ótimo é calculado automaticamente no `03_mlp.ipynb` para minimizar o custo operacional total (tipicamente < 0.5, favorecendo Recall).

---

## Re-treino Automatizado

O pipeline de re-treino roda mensalmente via GitHub Actions e substitui os artefatos em produção **apenas se o novo modelo superar o atual em AUC-ROC**.

### Execução manual

```bash
# uso padrão (lê variáveis de ambiente ou defaults)
python -m src.pipeline.retrain

# com caminhos explícitos
python -m src.pipeline.retrain \
  --data-path data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv \
  --artifacts-path models/artifacts \
  --experiment-name tc1-churn-retrain
```

**Exit codes:**
- `0` — modelo **promovido** (artefatos atualizados, run logada no MLflow com `promoted=true`)
- `2` — modelo **rejeitado** (artefatos mantidos, run logada com `promoted=false`)
- `1` — erro inesperado

### Fluxo interno

```
load_raw() → train/test split (80/20 estratificado)
    ↓
ColumnTransformer fit_transform → 46 features
    ↓
ChurnMLP.fit() com EarlyStopping → avalia AUC-ROC no test set
    ↓
AUC-ROC novo > AUC-ROC atual (de model_config.json)?
    ├── SIM (exit 0) → salva artefatos + loga MLflow run (promoted=true)
    └── NÃO (exit 2) → mantém artefatos + loga MLflow run (promoted=false)
```

### Agendamento (GitHub Actions)

O workflow `.github/workflows/retrain.yml` roda todo dia 1 do mês às 06h UTC. Se o modelo for promovido (exit 0), os novos artefatos são commitados automaticamente no repositório. O disparo manual também está disponível via `workflow_dispatch` na aba Actions do GitHub.

### Variáveis de ambiente

Copie `.env.example` para `.env` e ajuste conforme necessário:

```
RETRAIN_DATA_PATH=data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
RETRAIN_ARTIFACTS_PATH=models/artifacts
MLFLOW_TRACKING_URI=mlruns
```

---

## Testes Automatizados

```bash
# Rodar testes rápidos (sem treino real)
make test

# Rodar todos os testes incluindo @slow (treino real)
make test-all
```

**Suite atual: 41 testes | Cobertura: ≥ 85%**

| Arquivo | Testes | O que cobre |
|---|---|---|
| `test_smoke.py` | 5 | Imports e execução básica dos módulos |
| `test_schema.py` | 8 | Colunas, tipos e invariantes de dados |
| `test_models.py` | 7 | ChurnMLP forward, EarlyStopping, fit(), predict_proba() |
| `test_api.py` | 9 | Endpoints /health, /model-info e /predict; validação Pydantic; mock de modelo |
| `test_retrain.py` | 4+1 | Promoção/rejeição condicional, logging MLflow, preservação de threshold; end-to-end @slow |

Os testes da API e de re-treino (@unit) usam mocks — não é necessário ter os artefatos treinados para rodá-los.

---

## Qualidade de Código

```bash
# Via Makefile (recomendado)
make lint      # ruff check src/ tests/
make format    # ruff format src/ tests/

# Ou diretamente
ruff check src/ tests/
ruff format src/ tests/
mypy src/
```

Configuração completa em [pyproject.toml](pyproject.toml).

---

## Documentação

| Documento | Descrição |
|---|---|
| [docs/ml_canvas.md](docs/ml_canvas.md) | ML Canvas: formulação do problema, SLOs, análise de custo, stakeholders |
| [docs/model_card.md](docs/model_card.md) | Model Card: arquitetura, performance, limitações, plano de monitoramento, ética |
| [docs/comparacao_modelos.png](docs/comparacao_modelos.png) | Gráfico comparativo de modelos (gerado automaticamente) |
| [docs/analise_custo.png](docs/analise_custo.png) | Curva de custo vs. threshold (gerado automaticamente) |

---

## Versões das Dependências

| Pacote | Versão |
|---|---|
| Python | 3.9+ (testado em 3.14.5) |
| torch | 2.12.0 |
| mlflow | 3.13.0 |
| scikit-learn | 1.9.0 |
| pandas | 2.3.3 |
| numpy | 2.4.6 |
| fastapi | 0.136.3 |
| pandera | 0.31.1 |

---

## Deploy

### Status atual

A API de inferência está **totalmente funcional** localmente. Execute o pipeline completo (notebooks 01→03) para gerar os artefatos e então suba a API:

```bash
# via Make
make api

# ou diretamente
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

Para subir a stack completa (API + MLflow UI) com Docker:

```bash
make docker-up
# API:    http://localhost:8000/docs
# MLflow: http://localhost:5000
```

**SLOs de produção:**
- Latência ≤ 500 ms (p95)
- Disponibilidade ≥ 99.5%
- Retreinamento mensal automático (substituição condicional por AUC-ROC)

### Fase 3 — Roadmap

| Item | Status |
|------|--------|
| Dockerfile (API multi-stage) | ✅ Concluído |
| docker-compose (API + MLflow) | ✅ Concluído |
| GitHub Actions CI (lint + test + build) | ✅ Concluído |
| Re-treino automatizado mensal | ✅ Concluído |
| Monitoramento de data drift | 🔜 A implementar |
| Dashboard de métricas em produção | 🔜 A implementar |
| Deploy em cloud (AWS/GCP/Azure) | 🔜 A implementar |

---

## Como Contribuir

1. Abra uma issue descrevendo a melhoria ou o bug
2. Crie um branch: `git checkout -b feature/descricao-curta`
3. Implemente com testes que validem a mudança
4. Submeta um pull request com descrição clara

---

## Contato

| Membro | E-mail |
|---|---|
| Rafael Fernando Gimenes | rafael.gimenes17@gmail.com |

---

## Licença

Este projeto está licenciado sob a **MIT License**.
