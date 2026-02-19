# 📈 Stock Price Predictor - LSTM

> Projeto de Machine Learning para previsão de preços de ações usando Deep Learning (LSTM)
> 
> **Tech Challenge - Fase 4** | Pós-graduação em Machine Learning Engineering

> **Alunos** | Cleiton Cardoso, Geovana Godoy

---

## 🎯 Objetivo

Desenvolver um modelo de **rede neural LSTM (Long Short-Term Memory)** capaz de prever o preço de fechamento do próximo dia de uma ação, utilizando os últimos 60 dias de histórico como entrada.

---

## 📊 Resultados Obtidos

| Métrica | Valor | Interpretação |
|---------|-------|---------------|
| **MAPE** | **3.83%** | Erro percentual médio (Excelente: < 5%) |
| **RMSE** | R$ 0.89 | Erro médio em reais |
| **MAE** | R$ 0.70 | Erro absoluto médio |

### Diagnóstico: ✅ EXCELENTE
O modelo apresenta alta precisão, errando em média apenas R$ 0.89 por previsão.

---

## 🏗️ Arquitetura do Modelo

```
┌─────────────────────────────────────────────────────────┐
│                    StockLSTM                            │
├─────────────────────────────────────────────────────────┤
│  Input: (batch, 60, 1) - 60 dias de preços              │
│                         ↓                               │
│  ┌─────────────────────────────────────────────────┐    │
│  │  LSTM Layer                                     │    │
│  │  - input_size: 1                                │    │
│  │  - hidden_size: 100                             │    │
│  │  - num_layers: 2                                │    │
│  │  - dropout: 0.2                                 │    │
│  └─────────────────────────────────────────────────┘    │
│                         ↓                               │
│  ┌─────────────────────────────────────────────────┐    │
│  │  Dropout (0.2)                                  │    │
│  └─────────────────────────────────────────────────┘    │
│                         ↓                               │
│  ┌─────────────────────────────────────────────────┐    │
│  │  Linear Layer (100 → 1)                         │    │
│  └─────────────────────────────────────────────────┘    │
│                         ↓                               │
│  Output: (batch, 1) - Preço previsto                    │
└─────────────────────────────────────────────────────────┘
```

**Parâmetros treináveis:** ~42.000

---

## 📁 Estrutura do Projeto

```
tech-challenge-lstm/
├── 📄 README.md                 # Este arquivo
├── 📄 PROGRESS.md               # Acompanhamento do projeto
├── 📄 requirements.txt          # Dependências Python
├── 🐳 Dockerfile                # Containerização
├── 🐳 docker-compose.yml        # Orquestração
│
├── 📁 src/                      # Código-fonte
│   ├── data_collection.py       # Coleta de dados (yfinance)
│   ├── preprocessing.py         # Normalização e janelas
│   ├── model.py                 # Arquitetura LSTM
│   ├── train.py                 # Loop de treinamento
│   ├── evaluate.py              # Métricas de avaliação
│   └── app.py                   # API FastAPI
│
├── 📁 models/                   # Artefatos salvos
│   ├── model_lstm.pth           # Modelo treinado
│   ├── scaler.pkl               # Normalizador MinMaxScaler
│   ├── config.pkl               # Configurações
│   ├── training_history.png     # Gráfico de treino
│   └── predictions_vs_actual.png # Gráfico de previsões
│
└── 📁 data/                     # Dados históricos
    └── data_PETR4_SA.csv        # PETR4.SA (2018-2024)
```

---

## 🚀 Como Executar

### Opção 1: Docker (Recomendado)

```bash
# Build da imagem
docker build -t stock-predictor-api .

# Executar container
docker run -d --name stock-api -p 8000:8000 stock-predictor-api

# Verificar se está rodando
curl http://localhost:8000/health
```

### Opção 2: Localmente

```bash
# Criar ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou: venv\Scripts\activate  # Windows

# Instalar dependências
pip install -r requirements.txt

# Executar API
cd src
uvicorn app:app --reload
```

---

## 🔌 API Endpoints

### GET /health
Verifica o status da API e do modelo.

```bash
curl http://localhost:8000/health
```

**Resposta:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu",
  "ticker": "PETR4.SA",
  "seq_length": 60
}
```

### POST /predict
Recebe preços históricos e retorna a previsão.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"prices": [36.5, 36.8, 37.1, ... (60 valores)]}'
```

**Resposta:**
```json
{
  "predicted_price": 38.03,
  "currency": "BRL",
  "ticker": "PETR4.SA",
  "input_days": 60,
  "processing_time_ms": 4.62,
  "model_info": {
    "type": "LSTM",
    "hidden_size": 100,
    "num_layers": 2,
    "device": "cpu"
  }
}
```

### GET /docs
Documentação interativa Swagger/OpenAPI.

```
http://localhost:8000/docs
```

---

## 📋 Dados Utilizados

| Propriedade | Valor |
|-------------|-------|
| **Ticker** | PETR4.SA (Petrobras) |
| **Período** | 2018-01-01 a 2024-01-01 |
| **Registros** | ~1.400 dias |
| **Feature** | Preço de Fechamento (Close) |
| **Fonte** | Yahoo Finance (yfinance) |

---

## 🧠 Treinamento

| Parâmetro | Valor |
|-----------|-------|
| Épocas | 100 |
| Otimizador | Adam |
| Learning Rate | 0.001 |
| Loss Function | MSELoss |
| Batch Size | Full batch |
| Train/Test Split | 80% / 20% |
| Sequência (janela) | 60 dias |

---

## 🛠️ Tecnologias

- **Python 3.10+**
- **PyTorch** - Framework de Deep Learning
- **FastAPI** - API REST
- **scikit-learn** - Pré-processamento (MinMaxScaler)
- **yfinance** - Coleta de dados
- **Docker** - Containerização
- **Uvicorn** - Servidor ASGI

---

## 📚 Referências Teóricas

- **LSTM (Long Short-Term Memory):** Arquitetura de rede neural recorrente com células de memória e portões (gates) para capturar dependências de longo prazo em sequências.
- **Backpropagation Through Time (BPTT):** Algoritmo de treinamento que propaga o erro ao longo do tempo.
- **MinMaxScaler:** Normalização dos dados entre 0 e 1 para melhor convergência.

---

## 👨‍💻 Autor

**Tech Challenge - Fase 4**  
Pós-graduação em Machine Learning Engineering

---

## 📄 Licença

Este projeto foi desenvolvido para fins educacionais como parte do Tech Challenge da Fase 4.
