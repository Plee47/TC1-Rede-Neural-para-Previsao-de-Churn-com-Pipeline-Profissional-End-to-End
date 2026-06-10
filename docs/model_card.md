# Model Card — Churn MLP PyTorch

## Visão Geral do Modelo

| Campo | Valor |
|---|---|
| **Nome** | ChurnMLP |
| **Versão** | 0.1.0 |
| **Tipo** | Classificação binária supervisionada |
| **Framework** | PyTorch 2.x |
| **Data de treinamento** | Junho/2025 |
| **Responsáveis** | Equipe TC1 — FIAP MBA IA para Devs |

### Arquitetura

```
Input (46 features após OHE)
  → Linear(128) → BatchNorm1d → ReLU → Dropout(0.3)
  → Linear(64)  → BatchNorm1d → ReLU → Dropout(0.3)
  → Linear(32)  → BatchNorm1d → ReLU → Dropout(0.3)
  → Linear(1)   → [logits]
```

- **Loss:** `BCEWithLogitsLoss` com `pos_weight ≈ 2.77` (compensa o desbalanceamento de classes)
- **Otimizador:** Adam (`lr=1e-3`, `weight_decay=1e-4`)
- **Early stopping:** paciência de 15 épocas monitorando AUC-ROC de validação; restaura melhores pesos
- **Parâmetros treináveis:** 16.833

---

## Uso Pretendido

### Uso Primário

Prever a probabilidade de churn (cancelamento) de clientes de uma operadora de telecomunicações com base em características contratuais e de consumo. A saída é consumida pela equipe de CRM para priorizar ações de retenção proativa.

### Usuários Pretendidos

- Equipes de CRM e retenção de clientes
- Analistas de negócio que interpretam scores de risco
- Plataforma interna de decisão via API REST

### Uso Fora do Escopo

- Não deve ser usado como único critério para cancelamento de serviços
- Não adequado para segmentos fora do contexto de telecomunicações norte-americanas
- Não deve ser usado para decisões que impactem privacidade ou crédito sem revisão humana

---

## Dados de Treinamento

| Atributo | Valor |
|---|---|
| **Dataset** | Telco Customer Churn (IBM / Kaggle) |
| **Registros** | 7.043 clientes |
| **Features originais** | 20 (3 numéricas + 16 categóricas + 1 target) |
| **Features após pré-processamento** | 46 (StandardScaler + OneHotEncoder) |
| **Taxa de churn** | 26,5% (desbalanceado) |
| **Split** | 70% treino / 15% validação / 15% teste (StratifiedShuffleSplit) |
| **Seed** | 42 |

### Principais Features

| Feature | Tipo | Importância (EDA) |
|---|---|---|
| `Contract` | Categórica | Alta — Month-to-month: ~42% churn |
| `tenure` | Numérica | Alta — 0–6 meses: ~48% churn |
| `InternetService` | Categórica | Alta — Fiber optic: ~42% churn |
| `TechSupport` | Categórica | Média |
| `OnlineSecurity` | Categórica | Média |
| `MonthlyCharges` | Numérica | Média |

### Pré-processamento

```python
ColumnTransformer([
    ('num', StandardScaler(), ['tenure', 'MonthlyCharges', 'TotalCharges']),
    ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_FEATURES),
])
```

---

## Performance

Avaliação no conjunto de **teste** (1.057 amostras, nunca vistas durante treino ou validação).

| Modelo | AUC-ROC | PR-AUC | Recall | F1 | Precisão |
|---|---|---|---|---|---|
| **MLP PyTorch** | **0.8456** | **0.6447** | **0.8321** | **0.6148** | **0.4874** |
| LogisticRegression | 0.8490 | 0.6409 | 0.8107 | 0.6245 | 0.5078 |
| GradientBoosting | 0.8467 | 0.6539 | 0.5179 | 0.5788 | 0.6561 |
| RandomForest | 0.8233 | 0.6080 | 0.6750 | 0.6117 | 0.5592 |
| DummyClassifier | 0.4759 | 0.2568 | 0.2286 | 0.2290 | 0.2294 |

> O MLP foi selecionado como modelo de produção pelo **maior Recall (0.8321)**, métrica prioritária dado que o custo de um Falso Negativo (cliente que churnará sem intervenção) é ~30× maior que um Falso Positivo (campanha desnecessária).

### Threshold de Decisão

O limiar de 0.5 foi substituído pelo **threshold ótimo de custo**, calculado varrendo 91 limiares (0.05–0.95) e minimizando:

```
custo_total = FP × $10 + FN × $300
```

O threshold ótimo é menor que 0.5, priorizando Recall em detrimento de Precisão.

### Metas de SLO

| Métrica | Meta | Status |
|---|---|---|
| AUC-ROC (produção) | ≥ 0,78 | ✅ |
| Recall (classe positiva) | ≥ 0,70 | ✅ |
| PR-AUC | ≥ 0,60 | ✅ |
| Latência API (p95) | ≤ 500 ms | A verificar em produção |
| Disponibilidade | ≥ 99,5% | A verificar em produção |

---

## Análise de Custo — FP vs FN

| Erro | Situação | Custo estimado |
|---|---|---|
| **Falso Positivo (FP)** | Cliente contatado que não ia churnar | $10 (campanha desnecessária) |
| **Falso Negativo (FN)** | Cliente churnou sem intervenção | $300 (receita mensal perdida) |

**Premissas do cálculo:**
- Ticket médio mensal: $64,76
- Duração de retenção bem-sucedida: 6 meses
- Taxa de sucesso da campanha: 30%
- Custo da campanha por cliente: $15,00

Cálculo completo disponível em `notebooks/03_mlp.ipynb` (Seção 5) e `docs/analise_custo.png`.

---

## Limitações e Vieses

- **Escopo geográfico:** Dataset de operadora norte-americana; performance pode ser inferior em outros mercados
- **Dados estáticos:** Não captura comportamento temporal do cliente (séries temporais)
- **Features ausentes:** Sem dados de satisfação (NPS, tickets de suporte), que são preditores fortes de churn
- **Desbalanceamento:** Apenas 26,5% de positivos; o modelo usa `pos_weight` para compensar, mas pode haver viés em segmentos sub-representados
- **Drift:** O modelo foi treinado em dados de um período específico; performance pode degradar com mudanças no comportamento dos clientes

---

## Plano de Monitoramento

Este plano descreve como detectar degradação do modelo ChurnMLP em produção, quais alertas devem ser disparados e o passo-a-passo de resposta para cada cenário. O ciclo de monitoramento é **mensal**, alinhado com o re-treino automatizado (`.github/workflows/retrain.yml`).

---

### 4.1 Métricas de Monitoramento

#### 4.1.1 Métricas de Performance do Modelo

Avaliadas mensalmente em um conjunto de avaliação rotulado ("golden set") coletado nos 30 dias anteriores.

| Métrica | Baseline (treino) | Alerta Amarelo | Alerta Vermelho |
|---|---|---|---|
| **AUC-ROC** | 0.8456 | < 0.82 (−2,5 pp) | < 0.79 (−5,5 pp) |
| **Recall** | 0.8321 | < 0.75 (−8 pp) | < 0.65 (−18 pp) |
| **PR-AUC** | 0.6447 | < 0.60 (−4,5 pp) | < 0.55 (−9,5 pp) |
| **F1-Score** | 0.6148 | < 0.57 (−4,5 pp) | < 0.52 (−9,5 pp) |

> **Justificativa dos limiares:** degradação de 2–5 pp indica drift incipiente e requer observação; degradação ≥ 5 pp indica falha sistêmica e exige intervenção imediata. Recall recebe limiar mais conservador pois o custo de Falso Negativo é ~30× maior que o de Falso Positivo (FN = $300 vs. FP = $10).

#### 4.1.2 Métricas de Data Drift (Drift de Covariáveis)

Monitoram mudanças na distribuição das features de entrada, comparando a distribuição mensal atual contra a distribuição de treinamento (referência).

**Features numéricas** — Population Stability Index (PSI):

| Feature | Alerta | Interpretação |
|---|---|---|
| `tenure` | PSI > 0.10 (moderado) | Mudança no tempo médio de contrato da base |
| `MonthlyCharges` | PSI > 0.10 | Possível reajuste de preços ou mudança de plano |
| `TotalCharges` | PSI > 0.25 (severo) | Correlaciona com tenure — avaliar em conjunto |
| `SeniorCitizen` | Δ proporção > 5 pp | Mudança demográfica na amostra |

> **Interpretação PSI:** 0–0.10 = estável; 0.10–0.25 = mudança moderada (monitorar); > 0.25 = mudança severa (acionar retreino).

**Features categóricas** — diferença de frequência por categoria (ou teste qui-quadrado):

| Feature | Alerta | Interpretação |
|---|---|---|
| `Contract` | Δ > 10 pp em qualquer categoria | Mudança no perfil contratual da base |
| `InternetService` | Δ > 10 pp | Mudança no mix de produtos ofertados |
| `PaymentMethod` | Δ > 10 pp | Mudança no comportamento de pagamento |
| `Churn` (target real) | Δ taxa real vs. prevista > 5 pp | Calibração do modelo degradou |

#### 4.1.3 Métricas de Output Drift (Drift de Predições)

Monitoram mudanças na distribuição das probabilidades de saída, independente do label real.

| Métrica | Referência | Alerta |
|---|---|---|
| Média de `churn_probability` | ~0.27 (taxa de churn do treino) | Δ > 5 pp |
| Percentil 90 de `churn_probability` | Estável | Δ > 8 pp vs. referência |
| Taxa de predições positivas (score > threshold) | ~26,5% | Δ > 8 pp vs. taxa histórica |
| Kolmogorov-Smirnov (mensal vs. referência) | p-value > 0.05 (estável) | p-value < 0.05 |

#### 4.1.4 Métricas Operacionais (SLO da API)

| Métrica | Meta (SLO) | Alerta |
|---|---|---|
| Latência p50 | ≤ 100 ms | > 200 ms |
| Latência p95 | ≤ 500 ms | > 800 ms |
| Latência p99 | ≤ 1.000 ms | > 2.000 ms |
| Taxa de erro HTTP 5xx | ≤ 0,1% | > 1% |
| Disponibilidade | ≥ 99,5% | < 99% |

---

### 4.2 Alertas e Severidade

| Severidade | Critério | Prazo de Resposta |
|---|---|---|
| **P0 — Crítico** | AUC-ROC < 0.79 **ou** Recall < 0.65 **ou** API indisponível (disponibilidade < 99%) | 4 horas |
| **P1 — Alto** | AUC-ROC < 0.82 **ou** Recall < 0.75 **ou** PSI > 0.25 em feature numérica principal | 24 horas |
| **P2 — Médio** | PR-AUC < 0.60 **ou** PSI 0.10–0.25 **ou** taxa de churn real vs. prevista Δ > 5 pp | 72 horas |
| **P3 — Baixo** | Latência p95 > 500 ms **ou** KS p-value < 0.05 (output drift moderado) | Próximo ciclo mensal |

**Canais de alerta sugeridos:**
- P0/P1: notificação imediata via e-mail + Slack para a equipe de ML
- P2: ticket criado no sistema de gestão de incidentes
- P3: registrado no relatório mensal de monitoramento

---

### 4.3 Playbook de Resposta a Incidentes

#### Cenário A — Degradação de Performance sem Data Drift

**Sintomas:** AUC-ROC ou Recall caiu além do limiar, mas distribuições de features estão estáveis (PSI < 0.10).

**Causa provável:** Concept drift — o relacionamento entre features e target mudou (ex.: novos fatores de churn não capturados pelo modelo).

**Passos:**

1. Confirmar que os dados de avaliação mensal são representativos (verificar ausência de bug de coleta — contagem de registros, proporção de classes)
2. Verificar no MLflow o histórico de runs de `tc1-churn-retrain` para identificar tendência de queda nas últimas execuções
3. Executar re-treino imediato: `python -m src.pipeline.retrain --experiment-name tc1-churn-emergency`
4. Se o novo modelo não melhorar (exit code 2), escalar para revisão manual de features — podem ser necessárias novas variáveis preditoras (ex.: dados de suporte, NPS)
5. Documentar a ocorrência no MLflow (campo `notes` da run) com data, métricas observadas e ação tomada

#### Cenário B — Data Drift com Degradação de Performance

**Sintomas:** PSI > 0.25 ou qui-quadrado significativo **E** queda de performance simultânea.

**Causa provável:** A distribuição da população de clientes mudou de forma que o modelo não generaliza bem para o novo perfil (ex.: campanha de captação, sazonalidade, mudança de produto).

**Passos:**

1. Identificar quais features sofreram drift pelo relatório PSI/qui-quadrado mensal
2. Verificar se a mudança é real ou é artefato de bug no pipeline de dados (validar contagem de registros, ausência de valores nulos incomuns, datas de corte corretas)
3. **Se for bug de pipeline:** corrigir a ingestão e re-executar a avaliação com dados limpos
4. **Se for mudança real na população:**
   - Coletar dados representativos do novo perfil de clientes
   - Executar re-treino: `python -m src.pipeline.retrain`
   - Se o dataset original não captura o novo perfil, considerar substituição parcial do conjunto de treino com dados mais recentes
5. Monitorar métricas na semana seguinte ao re-treino para confirmar recuperação
6. Atualizar os valores de referência (baseline) neste plano se a mudança for permanente

#### Cenário C — Data Drift sem Degradação de Performance

**Sintomas:** PSI > 0.10 em features de entrada, mas métricas de performance estão dentro do SLO.

**Causa provável:** Mudança no perfil da base que o modelo já generaliza bem — não é emergencial.

**Passos:**

1. Registrar o drift observado no relatório mensal
2. Marcar como observação no MLflow para rastreabilidade histórica
3. Incluir no próximo ciclo de re-treino agendado (`.github/workflows/retrain.yml` já cobre isso automaticamente)
4. Se o drift persistir por 2+ meses consecutivos, reavaliar se o baseline de referência deve ser atualizado

#### Cenário D — Degradação Operacional (API)

**Sintomas:** Latência p95 > 800 ms ou taxa de erro HTTP 5xx > 1%.

**Passos:**

1. Verificar logs do container: `docker logs tc1-churn-api`
2. Confirmar que o modelo carregou corretamente: `curl http://localhost:8000/health` deve retornar `{"status": "ok"}`
3. **Se o modelo não carregou:** reiniciar o container e verificar integridade de `models/artifacts/mlp_weights.pt`
4. **Se a latência é alta mas funcional:** verificar uso de CPU/memória do container; considerar escalonamento horizontal se a causa for volume de requisições
5. Escalara para equipe de infraestrutura se os passos anteriores não resolverem

#### Cenário E — Re-treino Automático Mensal (Fluxo Normal)

**Este é o fluxo esperado — não é incidente.**

O workflow `.github/workflows/retrain.yml` executa no dia 1 de cada mês às 06h UTC:

1. `retrain.py` treina novo modelo e compara AUC-ROC com `model_config.json` atual
2. **Exit 0 (promovido):** novo AUC-ROC > AUC-ROC atual → artefatos atualizados automaticamente, commit com mensagem `chore: promote retrained model (YYYY-MM-DD)`, run no MLflow com tag `promoted=true`
3. **Exit 2 (rejeitado):** novo AUC-ROC ≤ atual → artefatos mantidos, run no MLflow com tag `promoted=false`
4. Em ambos os casos: verificar o log da GitHub Action para confirmar execução sem erros
5. Após promoção: comparar métricas do novo modelo vs. anterior no MLflow (`make mlflow`)

---

### 4.4 Frequência e Responsabilidades

| Atividade | Frequência | Responsável |
|---|---|---|
| Re-treino automatizado | Mensal (dia 1, 06h UTC) | GitHub Actions (automático) |
| Revisão de métricas de performance | Mensal | Equipe de ML |
| Cálculo de PSI e drift de features | Mensal | Equipe de ML |
| Revisão de SLOs operacionais | Semanal | Equipe de DevOps/ML |
| Atualização deste plano de monitoramento | Trimestral ou após incidente | Responsável pelo modelo |
| Auditoria completa (fairness, viés) | Semestral | Equipe de ML + Negócio |

---

## Informações Éticas

- **Privacidade:** O modelo não utiliza dados de identificação pessoal (PII) — o campo `customerID` é descartado no pré-processamento
- **Equidade:** Não foram realizadas análises de fairness por subgrupos (gênero, idade). Recomenda-se auditoria antes de usar em decisões que afetem grupos protegidos
- **Transparência:** Todos os experimentos estão registrados no MLflow (`notebooks/mlruns/`), garantindo rastreabilidade completa
- **Supervisão humana:** Scores de risco devem ser revisados pela equipe de CRM antes de ações de retenção automatizadas

---

## Artefatos

| Arquivo | Descrição |
|---|---|
| `models/artifacts/mlp_weights.pt` | Pesos do modelo PyTorch |
| `models/artifacts/preprocessor.joblib` | Pipeline sklearn de pré-processamento |
| `models/artifacts/model_config.json` | Metadados: arquitetura, threshold, métricas |
| `notebooks/mlruns/` | Experimentos MLflow (todos os modelos) |
| `docs/comparacao_modelos.png` | Gráfico comparativo de modelos |
| `docs/analise_custo.png` | Curva de custo vs. threshold |

---

## Citação

```
Gimenes, R. F. et al. (2025). Rede Neural para Previsão de Churn com Pipeline Profissional End-to-End.
FIAP MBA IA para Devs — Tech Challenge 1, Módulo 1.
```
