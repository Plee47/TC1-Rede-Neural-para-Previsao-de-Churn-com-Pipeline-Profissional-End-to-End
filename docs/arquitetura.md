# Arquitetura do Sistema — Churn Prediction API

> Documento de visão geral. Se você só vai ler **um** arquivo para entender este
> projeto, leia este. Os notebooks são o *relatório dos experimentos*; este
> documento explica o *sistema* que resultou deles.

---

## 1. O que este sistema faz

Recebe os dados cadastrais/contratuais de um cliente de telecom (19 campos) e
responde a probabilidade de ele cancelar o serviço (churn), junto com uma
classificação binária e uma faixa de risco para priorização do time de CRM.

```
JSON do cliente  ──►  API  ──►  {"churn_probability": 0.918, "churn_prediction": 1, "risk_band": "Alto"}
```

---

## 2. O conceito mais importante: treino ≠ execução

A confusão mais comum em ML serving é achar que o modelo "treina quando roda".
**Não treina.** São duas fases completamente separadas:

| | **Fase 1 — Treino** | **Fase 2 — Execução (serving)** |
|---|---|---|
| Quando | Uma vez, deliberadamente | Toda vez que a API sobe / recebe requisição |
| Onde | Máquina de desenvolvimento | Container (local ou nuvem) |
| Precisa do dataset? | Sim (7.043 clientes históricos) | **Não** — o container nem tem o CSV |
| Duração | ~10 segundos | Subida ~2 s; cada predição em milissegundos |
| Comando | `python -m src.models.export_artifacts` | `uvicorn src.api.app:app` (ou `docker run`) |
| Produz | 3 arquivos em `models/artifacts/` | Respostas JSON |

O elo entre as fases são **3 arquivos** (~84 KB no total):

| Arquivo | O que é |
|---|---|
| `mlp_weights.pt` | O "modelo": 16.833 números aprendidos no treino (pesos da rede neural) |
| `preprocessor.joblib` | O "tradutor": converte os 19 campos crus nos 46 números que a rede entende — **idêntico** ao usado no treino |
| `model_config.json` | A "etiqueta": arquitetura da rede, threshold de decisão e métricas de teste |

**Retreino** acontece só quando alguém decide (SLO do canvas: a cada 90 dias ou
se AUC-ROC cair 3 p.p.): roda o script de novo → novos artefatos → rebuild da
imagem → redeploy.

---

## 3. Jornada de uma requisição `POST /predict`

```
Cliente (Swagger, CRM, curl)
   │  JSON com 19 campos
   ▼
src/api/app.py (FastAPI)
   │  valida campos (Pydantic) ──► campo inválido? HTTP 422
   │                            ──► artefatos ausentes? HTTP 503
   ▼
preprocessor.joblib
   │  19 campos ──► vetor de 46 números (StandardScaler + OneHotEncoder)
   ▼
mlp_weights.pt (rede MLP)
   │  46 números ──► probabilidade entre 0 e 1
   ▼
model_config.json
   │  probabilidade ≥ threshold (0.05)? ──► churn_prediction = 1
   ▼
Resposta JSON: probabilidade + predição + faixa de risco + threshold
```

---

## 4. Mapa do repositório: produção × laboratório

A divisão que organiza tudo:

> **Notebook = laboratório/relatório** (explora, analisa, justifica)
> **`src/` + script = fábrica** (reproduzível, testado, sem humano no loop)
> **Container = produto** (só carrega o resultado)

| Caminho | Papel | A API precisa? |
|---|---|---|
| `src/api/app.py` | A API: endpoints, validação, carga do modelo | ✅ |
| `src/api/static/demo.html` | Página de demonstração servida em `GET /` | ✅ |
| `src/models/mlp.py` | Arquitetura da rede (classe `ChurnMLP`) | ✅ (para reconstituir a rede) |
| `src/models/train.py` | Loop de treino + `predict_proba` | ✅ (só o `predict_proba`) |
| `src/models/export_artifacts.py` | **Fábrica do modelo**: treina e exporta os 3 artefatos | Só para gerar modelo novo |
| `src/data/loader.py` | Carga/limpeza do CSV bruto | Só no treino |
| `src/evaluation/metrics.py` | Métricas (AUC, recall…) | Só no treino |
| `models/artifacts/*` | O modelo pronto (3 arquivos) | ✅ |
| `notebooks/01_eda.ipynb` | Relatório: análise exploratória, insights de churn | ❌ |
| `notebooks/02_baselines.ipynb` | Relatório: baselines + MLflow (piso de performance) | ❌ |
| `notebooks/03_mlp.ipynb` | Relatório: treino da MLP, comparação, análise de custo | ❌ |
| `tests/` | 37 testes automatizados (unitários + integração com artefatos reais) | — |
| `scripts/quality_gate.py` | Portão de qualidade: valida métricas contra os SLOs do canvas | Só no retreino |
| `Dockerfile` + `.dockerignore` | Receita da imagem de produção | — |
| `render.yaml` | Deploy na nuvem (Render Blueprint) | — |
| `.github/workflows/retrain.yml` | Pipeline de retreino (agendado/manual) | — |
| `.github/workflows/ci.yml` | CI: roda a suíte em todo push/PR | — |

Os notebooks ficam no repositório como **registro avaliável e justificativa das
decisões** — nenhum deles está no caminho de execução da API.

---

## 5. Como rodar

### Local (desenvolvimento)

```bash
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -r requirements.txt

# gerar o modelo (uma vez) — precisa do CSV em data/raw/
python -m src.models.export_artifacts

# subir a API
uvicorn src.api.app:app --reload
```

- Demo visual: <http://localhost:8000/>
- Swagger (testar endpoints): <http://localhost:8000/docs>
- ReDoc (documentação navegável): <http://localhost:8000/redoc>

### Testes

```bash
python -m pytest
```

Os testes da API usam um modelo sintético injetado — **não** exigem artefatos
treinados.

### Container

```bash
docker build -t churn-api .        # ou podman build
docker run -p 8000:8000 churn-api
```

A imagem embarca código + dependências + artefatos. Não baixa nada em runtime.

### Nuvem

`render.yaml` na raiz: no Render, **New → Blueprint** apontando para o repo.
O Dockerfile honra `$PORT`, então Railway/Fly/Cloud Run também funcionam.

### Retreino automatizado (GitHub Actions)

O workflow [`retrain.yml`](../.github/workflows/retrain.yml) implementa a
política de retreino do canvas (90 dias) sem treino no boot nem endpoint
`/train` — decisão justificada na seção 7:

```
disparo (botão "Run workflow" ou cron trimestral)
  → baixa dataset (com sanity check)
  → python -m src.models.export_artifacts          (treina)
  → python scripts/quality_gate.py                 (SLOs: AUC ≥ 0.78, PR-AUC ≥ 0.60, recall ≥ 0.70)
  → pytest -m integration                          (API carrega e discrimina?)
  → abre PR com os 3 artefatos + métricas no corpo (revisão humana)
  → merge → Render redeploya
```

Se qualquer gate falhar, o pipeline para e **o modelo antigo continua em
produção** — modelo novo nunca entra sem validação + revisão.

> Configuração única no GitHub: *Settings → Actions → General → "Allow GitHub
> Actions to create and approve pull requests"*. O agendamento (cron) só roda
> na branch padrão do repositório.

---

## 6. Endpoints

| Método | Rota | Para quê |
|---|---|---|
| `GET` | `/` | Página de demonstração (formulário visual) |
| `GET` | `/health` | Liveness — usado pelo healthcheck do container |
| `GET` | `/model-info` | Metadados do modelo carregado |
| `POST` | `/predict` | Score de 1 cliente |
| `POST` | `/predict/batch` | Score de uma lista de clientes (CRM) |
| `GET` | `/docs`, `/redoc` | Documentação interativa gerada pelo FastAPI |

---

## 7. Decisões registradas (e por quê)

| Decisão | Motivo |
|---|---|
| **Threshold 0.05** (não 0.5) | Análise de custo do `03_mlp.ipynb`: um churner perdido (FN ≈ $300) custa 30× um alarme falso (FP ≈ $10); o ponto de custo mínimo favorece recall. ⚠️ *Pendente de validação final pelo time — é agressivo.* |
| **MLP escolhida** (e não LogisticRegression, que tem AUC ligeiramente maior) | Maior recall (0.832), a métrica prioritária do problema |
| **Artefatos versionados no Git** (exceção no `.gitignore`) | 84 KB; permite build reproduzível direto do GitHub sem artifact store. Num MLOps estrito, troque por registry |
| **Script de export além do notebook** | Docker/CI não executam `.ipynb`; produção não pode depender de humano rodando célula |
| **torch CPU-only na imagem** | Evita ~2 GB do build CUDA; inferência de MLP pequena não precisa de GPU |
| **Modelo carregado no startup** (lifespan), não por requisição | Latência: carregar uma vez (~2 s) e servir da memória (ms) |
| **Usuário não-root no container** | Boa prática de segurança |
| **Retreino via pipeline, não via `/train` nem treino no boot** | Treino no serving bloqueia atendimento, morre com o container e pula validação. O pipeline traz dados novos, gates de qualidade e revisão humana — container burro e rápido; inteligência no pipeline |

### Dívida técnica conhecida

- A lista das 19 features existe em 4 lugares (`02_baselines`, `03_mlp`,
  `export_artifacts.py`, `app.py`). Proposta: centralizar em `src/features.py`
  e importar nos quatro. Requer combinação com o time (toca código de todos).
- Custos de negócio divergem entre documentos (FN $388 no canvas/01 vs $300 no
  03). Inofensivo hoje, mas vale alinhar.

---

## 8. Perguntas frequentes

**O modelo já vem treinado ou treina quando executa?**
Já vem treinado. O treino aconteceu uma vez (seção 2) e o resultado são os 3
arquivos. O container não tem o dataset — é incapaz de treinar.

**Preciso dos notebooks para subir a API?**
Não. O caminho de execução é: artefatos + `src/` → container. Os notebooks são
relatório/justificativa (seção 4).

**Por que a API às vezes responde 503?**
Os artefatos não estão em `models/artifacts/`. Gere com
`python -m src.models.export_artifacts`.

**Como troco o modelo em produção?**
Pelo pipeline: GitHub → aba Actions → `retreinar-modelo` → *Run workflow* (ou
espere o agendamento trimestral). Ele treina, valida os SLOs e abre um PR com
os artefatos novos; o merge dispara o redeploy. A API não muda: ela é
agnóstica ao conteúdo dos artefatos, desde que o contrato (19 campos →
probabilidade) se mantenha.

**Swagger é o único jeito de usar?**
Não. `GET /` (demo visual), `/redoc`, Postman/Insomnia (importando
`/openapi.json`), `curl`, ou qualquer linguagem com HTTP.
