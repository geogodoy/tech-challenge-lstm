# 📌 ETAPA 2: Coleta de Dados

## 📋 Resumo
| Item | Valor |
|------|-------|
| **Status** | ✅ Concluída |
| **Data** | 2026-02-17 |
| **Tempo Estimado** | 30 min |
| **Tempo Real** | ~10 min |

---

## 🎯 Objetivo
Baixar dados históricos de preços de ações usando a biblioteca `yfinance` para treinar o modelo LSTM.

---

## 🎓 Conexão com as Aulas

### Aula 03 - RNNs para Séries Temporais
**Arquivo:** `docs - fase 4 /etapa 1 - redes neurais e deep learning/Aula 03 - Arquiteturas de Redes Neurais Profundas.txt`

> *"As RNNs também são aplicadas na previsão de séries temporais para prever ações de mercado, condições climáticas ou demanda de energia. Neste contexto, elas podem capturar padrões temporais complexos e dependências de longo prazo."* (Linha ~456)

> *"No mercado financeiro, as RNNs podem ajudar a prever os movimentos dos preços das ações com base em séries temporais de preços passados."*

### Aula 04 - Dados Estruturados
**Conceito aplicado:**
- Preços de ações são **dados tabulares organizados** que alimentam redes neurais para tarefas de **regressão**
- Séries temporais requerem ordenação cronológica preservada

---

## 📁 Arquivo Implementado

### `src/data_collection.py`

#### Estrutura do Código

```python
# Linhas 1-5: Cabeçalho com referência ao guia
# ═══════════════════════════════════════════════════════════════
# 📌 ETAPA 2: Coleta de Dados
# 🎯 Objetivo: Baixar preços históricos usando a biblioteca yfinance
# ═══════════════════════════════════════════════════════════════
```

#### Configurações (Linhas 15-22)
```python
# Ticker escolhido: Petrobras (ação brasileira)
TICKER = "PETR4.SA"
START_DATE = "2018-01-01"
END_DATE = "2024-01-01"
DATA_DIR = Path(__file__).parent.parent / "data"
```

**Por que PETR4.SA?**
- Ação líquida do mercado brasileiro
- Volatilidade interessante para aprendizado
- Dados consistentes no período escolhido

#### Função Principal: `download_stock_data()` (Linhas 25-83)

| Etapa | Linha | Descrição |
|-------|-------|-----------|
| 1️⃣ | 47 | `yf.download()` - Baixa dados da API Yahoo Finance |
| 2️⃣ | 50-51 | Validação: verifica se DataFrame não está vazio |
| 3️⃣ | 65-69 | Estatísticas: min, max, média do preço de fechamento |
| 4️⃣ | 71-81 | Salvamento em CSV para uso posterior |

#### Função Auxiliar: `load_stock_data()` (Linhas 86-105)
- Carrega dados previamente salvos
- Útil para não baixar repetidamente da API

---

## 📊 Dados Coletados

### Estatísticas do Dataset

```
┌─────────────────────────────────────────────────┐
│           PETR4.SA - Petrobras                  │
├─────────────────────────────────────────────────┤
│  Período: 2018-01-02 até 2023-12-28             │
│  Registros: 1487 dias de negociação             │
│  Colunas: Date, Open, High, Low, Close, Volume  │
├─────────────────────────────────────────────────┤
│  Preço de Fechamento (Close):                   │
│    Mínimo:  R$ 3.24                             │
│    Máximo:  R$ 27.38                            │
│    Média:   R$ 10.17                            │
└─────────────────────────────────────────────────┘
```

### Arquivo Gerado
- **Caminho:** `data/data_PETR4_SA.csv`
- **Tamanho:** ~70KB
- **Formato:** CSV com índice de datas

---

## 🔬 Análise do Código vs Teoria

### Conceito da Aula → Implementação

| Conceito Teórico | Onde na Aula | Implementação no Código |
|------------------|--------------|------------------------|
| "Séries temporais requerem janela temporal longa" | Aula 03 | `START_DATE = "2018-01-01"` (6 anos de dados) |
| "RNNs capturam padrões temporais" | Aula 03, linha ~456 | Dados ordenados cronologicamente por `Date` |
| "Dados estruturados para regressão" | Aula 04 | DataFrame com colunas OHLCV |

### Estrutura dos Dados Baixados

```
Date (índice)  Open      High      Low       Close     Volume
2018-01-02     4.31      4.40      4.31      4.40      33461800
2018-01-03     4.39      4.45      4.36      4.44      55940900
...            ...       ...       ...       ...       ...
2023-12-28     27.28     27.37     27.14     27.28     21421900
```

**Coluna utilizada:** `Close` (preço de fechamento)
- Representa o preço final do dia
- Mais estável que Open/High/Low para previsão

---

## 💻 Execução

### Comando
```bash
cd src && python data_collection.py
```

### Saída
```
📥 Baixando dados de PETR4.SA...
   Período: 2018-01-01 até 2024-01-01
[*********************100%***********************]  1 of 1 completed

✅ Dados baixados com sucesso!
   Shape: (1487, 5)
   Período real: 2018-01-02 até 2023-12-28

📊 Primeiras linhas:
Date          Close      High       Low      Open    Volume
2018-01-02    4.409      4.409      4.313    4.313   33461800
...

💾 Dados salvos em: data/data_PETR4_SA.csv

🎉 CHECKPOINT: Dados coletados com sucesso!
```

---

## ✅ Checklist de Conclusão

- [x] Biblioteca `yfinance` funcionando
- [x] Ticker `PETR4.SA` selecionado
- [x] Período de 6 anos definido (2018-2024)
- [x] Dados baixados e validados
- [x] CSV salvo em `data/`
- [x] Função de carregamento implementada

---

## 🔗 Próxima Etapa

**→ ETAPA 3: Pré-processamento**
- Normalizar dados com MinMaxScaler
- Criar janelas deslizantes de 60 dias
- Dividir em treino/teste
