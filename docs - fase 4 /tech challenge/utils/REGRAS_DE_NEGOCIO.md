# Regras de Negócio - Tech Challenge LSTM

Este documento descreve as regras de negócio, contexto e funcionamento do projeto de predição de preços de ações utilizando redes neurais LSTM.

O documento inclui:
1 - Contexto do Projeto - Objetivo, escopo e entregáveis
2 - Por que LSTM? - Justificativa técnica com diagrama da célula LSTM
3 - Fluxo de Dados - Diferença clara entre treinamento e inferência com linha do tempo visual
4 - Funcionamento da API - Processamento interno passo a passo com diagramas
5 - Exemplo Completo de Uso - Cenário prático com código
6 - Testes e Validação - O que pode usar como dados de teste
7 - FAQ - Todas as perguntas que você levantou durante nossa conversa
8 - Resumo Executivo - Visão geral rápida do projeto

---

## 1. Contexto do Projeto

### 1.1 O que é este projeto?

Este é o **Tech Challenge da Fase 4** do curso de Machine Learning Engineering da POS TECH. Trata-se de uma atividade obrigatória que representa **90% da nota final**.

### 1.2 Objetivo Principal

> **Criar um modelo preditivo de redes neurais LSTM para prever o valor de fechamento de ações de uma empresa no mercado financeiro.**

### 1.3 Escopo Funcional

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FLUXO DO SISTEMA                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ENTRADA                    PROCESSAMENTO                    SAÍDA         │
│   ───────                    ────────────                    ─────         │
│                                                                             │
│   Dados históricos    ──►    Modelo LSTM       ──►    Previsão do preço    │
│   de preços de ações         (análise de              de fechamento do      │
│   (últimos 60 dias)          padrões temporais)       próximo dia           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.4 Entrega Final

O projeto exige demonstrar domínio do **ciclo completo de desenvolvimento**:

```
Coleta (yfinance) → Pré-processamento → Modelo LSTM → Treinamento → Avaliação → Deploy (API) → Monitoramento
```

**Entregáveis obrigatórios:**
- Código-fonte documentado no GitHub
- Scripts/containers Docker para deploy
- Link para API em produção (se deployada na nuvem)
- Vídeo explicativo demonstrando o funcionamento

---

## 2. Por que LSTM?

### 2.1 O Problema das Séries Temporais

Preços de ações são **séries temporais** - dados sequenciais onde a ordem importa. O preço de hoje depende dos padrões dos dias anteriores.

### 2.2 Limitações de RNNs Tradicionais

Redes Neurais Recorrentes (RNNs) comuns sofrem do problema de **vanishing gradient** - elas "esquecem" informações de longo prazo ao processar sequências longas.

### 2.3 A Solução: LSTM

A arquitetura **LSTM (Long Short-Term Memory)** resolve esse problema com células de memória especiais:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CÉLULA LSTM - ESTRUTURA                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                                                                     │   │
│   │   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐           │   │
│   │   │ FORGET GATE  │   │  INPUT GATE  │   │ OUTPUT GATE  │           │   │
│   │   │              │   │              │   │              │           │   │
│   │   │ "O que       │   │ "O que       │   │ "O que       │           │   │
│   │   │  esquecer?"  │   │  adicionar?" │   │  passar?"    │           │   │
│   │   └──────────────┘   └──────────────┘   └──────────────┘           │   │
│   │          │                  │                  │                    │   │
│   │          └──────────────────┼──────────────────┘                    │   │
│   │                             │                                       │   │
│   │                             ▼                                       │   │
│   │                    ┌──────────────┐                                 │   │
│   │                    │   MEMÓRIA    │                                 │   │
│   │                    │   DE LONGO   │                                 │   │
│   │                    │    PRAZO     │                                 │   │
│   │                    └──────────────┘                                 │   │
│   │                                                                     │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.4 Comparativo

| Característica do Problema | Por que LSTM resolve |
|---------------------------|----------------------|
| **Dados são séries temporais** | LSTM foi projetado para sequências onde a ordem importa |
| **Dependências de longo prazo** | Células de memória guardam contexto de semanas/meses |
| **RNNs comuns "esquecem"** | Portões (gates) controlam o que lembrar e esquecer |
| **Padrões complexos** | Múltiplas camadas capturam diferentes níveis de abstração |

---

## 3. Fluxo de Dados: Treinamento vs. Inferência

### 3.1 Conceito Fundamental

Existem **dois momentos completamente diferentes** no ciclo de vida do modelo:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   MOMENTO 1: TREINAMENTO                    MOMENTO 2: USO DA API           │
│   (acontece ANTES do deploy)                (acontece DEPOIS do deploy)     │
│                                                                             │
│   ┌─────────────────────────┐               ┌─────────────────────────┐     │
│   │  DESENVOLVEDOR (você)   │               │   USUÁRIO (cliente)     │     │
│   │                         │               │                         │     │
│   │  • Importa dados do     │               │  • Envia 60 preços      │     │
│   │    yfinance (anos)      │               │    recentes             │     │
│   │  • Treina o modelo      │               │  • Recebe 1 previsão    │     │
│   │  • Salva modelo.pth     │               │                         │     │
│   └───────────┬─────────────┘               └───────────┬─────────────┘     │
│               │                                         │                   │
│               ▼                                         ▼                   │
│        ┌─────────────┐                          ┌─────────────┐             │
│        │   MODELO    │                          │   MODELO    │             │
│        │  APRENDENDO │                          │  PREVENDO   │             │
│        │  (fit)      │                          │  (predict)  │             │
│        └─────────────┘                          └─────────────┘             │
│                                                                             │
│   QUANDO: Uma vez, antes de publicar          QUANDO: Sempre que quiser    │
│   DADOS: 6 anos de histórico (1500+ dias)     DADOS: 60 dias mais recentes │
│   OBJETIVO: Ensinar padrões ao modelo         OBJETIVO: Obter previsão     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Linha do Tempo do Projeto

```
══════════════════════════════════════════════════════════════════════════════
                              LINHA DO TEMPO
══════════════════════════════════════════════════════════════════════════════

    FASE DE DESENVOLVIMENTO                      FASE DE PRODUÇÃO
    (você faz uma vez)                           (usuários usam sempre)
    
    ────────────────────────────────────────────────────────────────────────►
                                                                        tempo
    
    ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ║  ┌──────────┐
    │ Importar │  │  Pré-    │  │ Treinar  │  │  Deploy  │  ║  │ Usuários │
    │  dados   │─▶│processar │─▶│  modelo  │─▶│   API    │  ║  │  usam    │
    │ yfinance │  │          │  │          │  │          │  ║  │   API    │
    └──────────┘  └──────────┘  └──────────┘  └──────────┘  ║  └──────────┘
         │                            │                     ║        │
         ▼                            ▼                     ║        ▼
    ┌──────────┐                ┌──────────┐               ║  ┌──────────┐
    │ 6 anos   │                │ modelo   │               ║  │ 60 dias  │
    │ de dados │                │ .pth     │               ║  │ enviados │
    │ (1500+   │                │ salvo    │               ║  │ pelo     │
    │  dias)   │                │          │               ║  │ usuário  │
    └──────────┘                └──────────┘               ║  └──────────┘
                                                           ║
    ◄─────────────── OFFLINE ──────────────────►          ║◄── ONLINE ──►
                                                           ║
                                                    API PUBLICADA
```

### 3.3 Comparativo dos Dados

| Aspecto | Dados de Treinamento | Dados da API (Inferência) |
|---------|---------------------|---------------------------|
| **Quem fornece** | Desenvolvedor | Usuário final |
| **Quando** | Antes do deploy | Durante uso da API |
| **Quantidade** | Anos de histórico (~1500 dias) | Apenas 60 dias |
| **Propósito** | Ensinar o modelo | Fazer uma previsão |
| **Frequência** | Uma vez (ou retreino periódico) | Quantas vezes quiser |
| **Fonte** | yfinance (importação em lote) | Qualquer fonte do usuário |

### 3.4 Analogia

Pense no modelo como um **aluno que estudou para uma prova**:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   📚 ESTUDAR (Treinamento)              ✍️ FAZER PROVA (Inferência)         │
│                                                                             │
│   • Acontece ANTES da prova             • Acontece DURANTE a prova          │
│   • Aluno lê MUITOS livros              • Aluno recebe UMA pergunta         │
│   • Leva semanas/meses                  • Responde em segundos              │
│   • Aprende padrões gerais              • Aplica o que aprendeu             │
│                                                                             │
│   ═══════════════════════════════════════════════════════════════════════   │
│                                                                             │
│   🤖 TREINAR MODELO                     🔮 USAR API                         │
│                                                                             │
│   • Acontece ANTES do deploy            • Acontece DEPOIS do deploy         │
│   • Modelo vê ANOS de dados             • Modelo recebe 60 DIAS             │
│   • Leva minutos/horas                  • Responde em milissegundos         │
│   • Aprende padrões de preços           • Aplica padrões para prever        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Funcionamento da API

### 4.1 Requisito do Tech Challenge

> *"Criação da API: desenvolva uma API RESTful utilizando Flask ou FastAPI para servir o modelo. A API deve permitir que o usuário forneça dados históricos de preços e receba previsões dos preços futuros."*

### 4.2 Fluxo da API

```
┌──────────────┐      POST /predict           ┌──────────────┐
│   USUÁRIO    │  ───────────────────────────▶│     API      │
│              │   { "prices": [32.5, 33.1,   │              │
│              │     33.8, 34.2, ... ] }      │  ┌────────┐  │
│              │                              │  │ MODELO │  │
│              │◀───────────────────────────  │  │  LSTM  │  │
│              │   { "predicted_price":       │  └────────┘  │
└──────────────┘     35.20 }                  └──────────────┘
```

**O usuário envia**: Lista com os últimos 60 preços de fechamento  
**A API retorna**: Previsão do próximo preço de fechamento

### 4.3 Processamento Interno Detalhado

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PROCESSAMENTO INTERNO DA API                         │
└─────────────────────────────────────────────────────────────────────────────┘

ENTRADA DO USUÁRIO (60 preços em R$):
┌─────────────────────────────────────────────────────────────────────────────┐
│ [36.42, 36.18, 36.55, ... , 42.45, 42.68, 42.55, 42.82]                     │
│  Dia 1   Dia 2   Dia 3        Dia 58  Dia 59  Dia 60  ← último dia conhecido│
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ ETAPA 1: VALIDAÇÃO                                                          │
│ ─────────────────                                                           │
│ ✓ Verificar se recebeu pelo menos 60 preços                                 │
│ ✓ Verificar se são números válidos                                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ ETAPA 2: NORMALIZAÇÃO (MinMaxScaler)                                        │
│ ────────────────────────────────────                                        │
│                                                                             │
│ O scaler foi treinado com dados históricos onde:                            │
│   • Mínimo histórico: R$ 15.00                                              │
│   • Máximo histórico: R$ 45.00                                              │
│                                                                             │
│ Fórmula: valor_normalizado = (valor - min) / (max - min)                    │
│                                                                             │
│ Exemplo para R$ 42.82:                                                      │
│   (42.82 - 15.00) / (45.00 - 15.00) = 27.82 / 30.00 = 0.927                │
│                                                                             │
│ ANTES (R$):  [36.42, 36.18, ... , 42.55, 42.82]                            │
│ DEPOIS (0-1): [0.714, 0.706, ... , 0.918, 0.927]                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ ETAPA 3: RESHAPE PARA TENSOR                                                │
│ ───────────────────────────────                                             │
│                                                                             │
│ A LSTM espera dados no formato: (batch_size, seq_length, features)          │
│                                                                             │
│ Dados normalizados: [0.714, 0.706, ... , 0.918, 0.927]  → shape: (60,)      │
│                                    │                                        │
│                                    ▼                                        │
│ Reshape para:        [[[0.714],                                             │
│                        [0.706],                                             │
│                        ...                                                  │
│                        [0.918],                                             │
│                        [0.927]]]                        → shape: (1, 60, 1) │
│                                                                             │
│                       ↑    ↑    ↑                                           │
│                       │    │    └── 1 feature (só preço de fechamento)      │
│                       │    └─────── 60 dias (sequência temporal)            │
│                       └──────────── 1 amostra (batch de 1 previsão)         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ ETAPA 4: PASSAGEM PELA REDE LSTM                                            │
│ ─────────────────────────────────                                           │
│                                                                             │
│   Dia 1      Dia 2      Dia 3           Dia 59     Dia 60                   │
│  [0.714] → [0.706] → [0.719] → ... → [0.918] → [0.927]                     │
│     │         │         │               │         │                         │
│     ▼         ▼         ▼               ▼         ▼                         │
│  ┌─────┐  ┌─────┐  ┌─────┐          ┌─────┐  ┌─────┐                       │
│  │LSTM │→│LSTM │→│LSTM │→  ...  →│LSTM │→│LSTM │                       │
│  │Cell │  │Cell │  │Cell │          │Cell │  │Cell │                       │
│  └─────┘  └─────┘  └─────┘          └─────┘  └──┬──┘                       │
│     │         │         │               │       │                           │
│   h₁,c₁ →   h₂,c₂ →   h₃,c₃ →      h₅₉,c₅₉→  h₆₀,c₆₀                     │
│  (memória passa de célula em célula)            │                           │
│                                                 ▼                           │
│                                          ┌──────────┐                       │
│                                          │ Dropout  │                       │
│                                          └────┬─────┘                       │
│                                               ▼                             │
│                                          ┌──────────┐                       │
│                                          │  Linear  │                       │
│                                          └────┬─────┘                       │
│                                               ▼                             │
│                                            [0.943]                          │
│                                     (previsão normalizada)                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ ETAPA 5: DESNORMALIZAÇÃO                                                    │
│ ─────────────────────────                                                   │
│                                                                             │
│ Saída da LSTM (normalizada): 0.943                                          │
│                                                                             │
│ Fórmula inversa: valor_real = valor_norm × (max - min) + min                │
│                                                                             │
│ Cálculo: 0.943 × (45.00 - 15.00) + 15.00                                   │
│        = 0.943 × 30.00 + 15.00                                              │
│        = 28.29 + 15.00                                                      │
│        = R$ 43.29                                                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ ETAPA 6: RESPOSTA                                                           │
│ ─────────────────                                                           │
│                                                                             │
│ {                                                                           │
│   "predicted_price": 43.29,                                                 │
│   "confidence_info": "Previsão baseada em modelo LSTM",                     │
│   "processing_time_ms": 12.45                                               │
│ }                                                                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.4 Visualização Temporal

```
          PASSADO (dados conhecidos)              │    FUTURO
                                                  │  (previsão)
    ─────────────────────────────────────────────────────────────────►
                                                  │    tempo
    Dia 1    Dia 2    ...    Dia 59    Dia 60    │    Dia 61
   R$36.42  R$36.18         R$42.55   R$42.82   │   R$43.29
      │        │                │        │       │      │
      └────────┴────────────────┴────────┘       │      │
                      │                          │      │
              ┌───────▼───────┐                  │      │
              │  60 preços    │                  │      │
              │  enviados     │                  │      │
              │  para a API   │                  │      │
              └───────┬───────┘                  │      │
                      │                          │      │
              ┌───────▼───────┐                  │      │
              │    MODELO     │──────────────────┼──────┘
              │     LSTM      │   previsão       │
              └───────────────┘                  │
```

---

## 5. Exemplo Completo de Uso

### 5.1 Cenário

Maria é analista financeira e quer saber qual será o **preço de fechamento da PETR4** amanhã.

### 5.2 Passo 1: Obter dados históricos

```python
import yfinance as yf

df = yf.download("PETR4.SA", start="2024-10-01", end="2024-12-31")
ultimos_60_precos = df['Close'].tail(60).tolist()
```

### 5.3 Passo 2: Chamar a API

```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={"prices": ultimos_60_precos}
)

resultado = response.json()
```

### 5.4 Passo 3: Interpretar resultado

```python
print(f"Último preço conhecido: R$ {ultimos_60_precos[-1]:.2f}")
print(f"Previsão para amanhã:   R$ {resultado['predicted_price']:.2f}")

variacao = ((resultado['predicted_price'] / ultimos_60_precos[-1]) - 1) * 100
print(f"Variação esperada:      {variacao:+.2f}%")
```

**Saída:**
```
Último preço conhecido: R$ 42.82
Previsão para amanhã:   R$ 43.29
Variação esperada:      +1.10%
```

### 5.5 Uso Contínuo

O usuário pode chamar a API quantas vezes quiser:

```python
# Segunda-feira
requests.post("/predict", json={"prices": precos_segunda})  # → R$ 43.29

# Terça-feira
requests.post("/predict", json={"prices": precos_terca})    # → R$ 43.85

# Quarta-feira
requests.post("/predict", json={"prices": precos_quarta})   # → R$ 44.12
```

---

## 6. Testes e Validação

### 6.1 Dados para Teste

A API aceita qualquer lista de 60 números válidos. Durante desenvolvimento e testes, você pode usar:

| Tipo de Dado | Válido? | Uso Recomendado |
|--------------|---------|-----------------|
| Dados reais do yfinance | ✅ Sim | Validação do modelo |
| Dados mockados simples (`[100.0] * 60`) | ✅ Sim | Testar se API funciona |
| Dados com variação (`range(100, 160)`) | ✅ Sim | Testar comportamento |
| Dados aleatórios | ✅ Sim | Teste de carga |
| Dados históricos de outro período | ✅ Sim | Validar generalização |

### 6.2 O que a API NÃO valida

```python
# A API NÃO verifica:
# ❌ Se são preços reais de uma ação
# ❌ Se são dos últimos 60 dias
# ❌ Se são de uma ação específica
# ❌ Se fazem sentido financeiramente

# A API APENAS verifica:
# ✓ Se recebeu pelo menos 60 números
# ✓ Se são valores numéricos válidos
```

### 6.3 Exemplo de Teste Rápido

```bash
# Teste com dados mockados
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"prices": [40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40]}'
```

---

## 7. FAQ - Perguntas Frequentes

### 7.1 Sobre Importação de Dados

**P: A importação dos dados do yfinance deve ser feita diariamente?**

R: **Não é exigido pelo Tech Challenge.** O escopo do projeto é importar os dados históricos uma vez para treinar o modelo. A API recebe os dados do usuário como input, não busca automaticamente.

**P: Os dados de treinamento são os mesmos enviados pelo usuário na API?**

R: **Não.** São dados completamente diferentes:
- **Treinamento**: Você importa anos de histórico antes do deploy
- **API**: Usuário envia 60 dias para obter uma previsão

### 7.2 Sobre Treinamento

**P: O treinamento deve passar por tuning antes de apresentar?**

R: **Sim.** O Tech Challenge menciona explicitamente: *"ajuste os hiperparâmetros para otimizar o desempenho"*.

**P: O treinamento deve ser contínuo?**

R: **Não é exigido.** O Tech Challenge trata o treinamento como um processo único antes do deploy. Retreinamento contínuo seria um requisito de MLOps mais avançado.

### 7.3 Sobre a API

**P: O usuário pode usar a API sempre que quiser?**

R: **Sim.** A API pode ser chamada quantas vezes necessário, a qualquer momento.

**P: Posso testar com dados mockados?**

R: **Sim.** Para testar se a API funciona, qualquer lista de 60 números serve. Para validar a qualidade das previsões, use dados reais.

### 7.4 Sobre o Modelo

**P: Por que LSTM e não uma RNN comum?**

R: Porque RNNs comuns "esquecem" informações de longo prazo. A LSTM resolve isso com suas células de memória e portões (gates).

**P: Qual métrica é mais importante?**

R: **MAPE** é mais intuitivo (erro em percentual), mas **MAE** em R$ é mais tangível para o usuário final. O Tech Challenge aceita qualquer uma das métricas: MAE, RMSE ou MAPE.

---

## 8. Resumo Executivo

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RESUMO DO PROJETO                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  OBJETIVO:     Prever preço de fechamento de ações usando LSTM              │
│                                                                             │
│  ENTRADA:      60 dias de preços históricos (enviados pelo usuário)         │
│                                                                             │
│  SAÍDA:        Previsão do preço do próximo dia                             │
│                                                                             │
│  MODELO:       LSTM (Long Short-Term Memory)                                │
│                                                                             │
│  API:          RESTful com FastAPI                                          │
│                                                                             │
│  DEPLOY:       Container Docker                                             │
│                                                                             │
│  MÉTRICAS:     MAE, RMSE, MAPE                                              │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  FLUXO RESUMIDO:                                                            │
│                                                                             │
│  ┌──────────────┐     ┌───────────────────────────────┐     ┌────────────┐ │
│  │   USUÁRIO    │────▶│            API                │────▶│  USUÁRIO   │ │
│  │              │     │                               │     │            │ │
│  │  Envia 60    │     │  Valida → Normaliza → LSTM    │     │  Recebe    │ │
│  │  preços      │     │  → Desnormaliza → Responde    │     │  previsão  │ │
│  └──────────────┘     └───────────────────────────────┘     └────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

*Documento gerado com base nas especificações do Tech Challenge - Fase 4 - Machine Learning Engineering*
