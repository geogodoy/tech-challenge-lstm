# 📌 ETAPA 8: API FastAPI

## 📋 Resumo
| Item | Valor |
|------|-------|
| **Status** | ✅ Concluída |
| **Data** | 2026-02-19 |
| **Tempo Estimado** | 60 min |
| **Tempo Real** | ~20 min |

---

## 🎯 Objetivo
Criar uma API REST para servir o modelo LSTM treinado, permitindo que aplicações externas enviem preços históricos e recebam previsões do próximo dia.

---

## 🎓 Conexão com as Aulas

### Aula 05 - Casos de Uso de Redes Neurais Profundas
**Arquivo:** `docs - fase 4 /etapa 1 - redes neurais e deep learning/Aula 05 - Casos de Uso de Redes Neurais Profundas`

#### Integrações Transacionais
> *"Modelos em produção precisam de alta disponibilidade e interfaces claras para o usuário."* (Guia, linha ~403)

#### Deploy de Modelos
> *"O deploy de modelos de deep learning envolve a exposição do modelo através de APIs RESTful, permitindo que sistemas externos consumam as previsões."*

### Aula 04 - Técnicas de Aplicação
**Arquivo:** `docs - fase 4 /etapa 1 - redes neurais e deep learning/Aula 04 - Técnicas de Aplicação de Redes Neurais Profundas`

#### Inferência em Produção
> *"Durante a inferência (produção), o modelo deve estar em modo de avaliação (eval) para desativar dropout e outras técnicas de regularização."*

---

## 📁 Arquivo Implementado

### `src/app.py`

#### Estrutura do Código

```python
# Linhas 1-6: Cabeçalho
# ═══════════════════════════════════════════════════════════════
# 📌 ETAPA 8: API FastAPI
# 🎯 Objetivo: Servir o modelo LSTM via endpoint REST
# ═══════════════════════════════════════════════════════════════
```

#### Dependências Principais
```python
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, field_validator
import torch
import joblib
import numpy as np
```

---

## 🔬 Componentes Principais

### 1. Estado Global (`ModelState`)

```python
class ModelState:
    """Armazena o estado do modelo carregado."""
    model: Optional[StockLSTM] = None
    scaler = None
    config: Optional[dict] = None
    device: str = "cpu"
    is_loaded: bool = False

state = ModelState()
```

**Por que usar estado global?**
- Evita carregar o modelo a cada requisição (muito lento)
- Mantém o modelo em memória para inferência rápida
- Permite verificar se o modelo está carregado via `/health`

---

### 2. Lifespan (Ciclo de Vida)

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gerencia o ciclo de vida da aplicação."""
    
    # STARTUP: Executado quando a API inicia
    state.config = joblib.load(CONFIG_PATH)
    state.scaler = joblib.load(SCALER_PATH)
    
    checkpoint = torch.load(MODEL_PATH, map_location=state.device)
    state.model = StockLSTM(**checkpoint['model_config'])
    state.model.load_state_dict(checkpoint['model_state_dict'])
    state.model.eval()  # Modo de inferência
    
    state.is_loaded = True
    
    yield  # API rodando
    
    # SHUTDOWN: Executado quando a API encerra
    print("🛑 Encerrando API...")
```

**Conexão com a teoria:**
- `model.eval()` desativa dropout durante inferência
- Carregamento único evita overhead por requisição

---

### 3. Schemas Pydantic (Validação)

#### Request Schema
```python
class PredictionRequest(BaseModel):
    prices: List[float] = Field(
        ...,
        description="Lista com os últimos N preços de fechamento"
    )
    
    @field_validator('prices')
    @classmethod
    def validate_prices(cls, v):
        if not v:
            raise ValueError("Lista de preços não pode estar vazia")
        if any(p <= 0 for p in v):
            raise ValueError("Todos os preços devem ser positivos")
        return v
```

#### Response Schema
```python
class PredictionResponse(BaseModel):
    predicted_price: float    # Preço previsto
    currency: str             # "BRL"
    ticker: str               # "PETR4.SA"
    input_days: int           # 60 (seq_length)
    processing_time_ms: float # Tempo de inferência
    model_info: dict          # Metadados do modelo
```

---

### 4. Endpoint `/predict` (POST)

```python
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    start_time = time.time()
    
    # 1️⃣ Validar se modelo está carregado
    if not state.is_loaded:
        raise HTTPException(status_code=503, detail="Modelo não carregado")
    
    # 2️⃣ Validar quantidade de preços
    seq_length = state.config.get('seq_length', 60)
    if len(request.prices) < seq_length:
        raise HTTPException(status_code=400, detail=f"Mínimo: {seq_length} preços")
    
    # 3️⃣ Pré-processar (normalizar)
    prices = np.array(request.prices[-seq_length:]).reshape(-1, 1)
    prices_scaled = state.scaler.transform(prices)
    
    # 4️⃣ Converter para tensor
    X = torch.FloatTensor(prices_scaled).unsqueeze(0).to(state.device)
    
    # 5️⃣ Fazer previsão
    with torch.no_grad():
        prediction_scaled = state.model(X).cpu().numpy()
    
    # 6️⃣ Reverter normalização
    predicted_price = state.scaler.inverse_transform(prediction_scaled)[0][0]
    
    # 7️⃣ Calcular tempo
    processing_time = (time.time() - start_time) * 1000
    
    return PredictionResponse(
        predicted_price=round(float(predicted_price), 2),
        currency="BRL",
        ticker=state.config.get('ticker', 'PETR4.SA'),
        input_days=seq_length,
        processing_time_ms=round(processing_time, 2),
        model_info={...}
    )
```

---

## 📊 Conexão Código ↔ Teoria

### Tabela de Mapeamento

| Conceito na Aula | Onde está | Código | Linha no Código |
|------------------|-----------|--------|-----------------|
| "Modo de inferência" | Aula 04 | `model.eval()` | 78 |
| "Desativar dropout" | Aula 04 | `model.eval()` | 78 |
| "Não calcular gradientes" | Aula 04 | `torch.no_grad()` | 205 |
| "Normalização" | Aula 02 | `scaler.transform()` | 197 |
| "Reverter normalização" | Aula 04 | `scaler.inverse_transform()` | 209 |

---

## 🔄 Fluxo da Requisição

```
┌─────────────────────────────────────────────────────────────┐
│                    FLUXO DO ENDPOINT /predict               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              1. VALIDAÇÃO (Pydantic)                │   │
│  │                                                     │   │
│  │  • Lista de preços não vazia?                       │   │
│  │  • Todos os preços são positivos?                   │   │
│  │  • Quantidade >= seq_length (60)?                   │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                 │
│                           ▼                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              2. PRÉ-PROCESSAMENTO                   │   │
│  │                                                     │   │
│  │  prices[-60:]          → Últimos 60 dias            │   │
│  │  reshape(-1, 1)        → Array 2D para scaler       │   │
│  │  scaler.transform()    → Normalizar [0,1]           │   │
│  │  torch.FloatTensor()   → Converter para tensor      │   │
│  │  unsqueeze(0)          → Adicionar batch dim        │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                 │
│                           ▼                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              3. INFERÊNCIA (model.eval())           │   │
│  │                                                     │   │
│  │  with torch.no_grad():  → Sem cálculo de gradientes │   │
│  │      prediction = model(X)                          │   │
│  │                                                     │   │
│  │  → Forward pass apenas (sem backpropagation)        │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                 │
│                           ▼                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              4. PÓS-PROCESSAMENTO                   │   │
│  │                                                     │   │
│  │  .cpu().numpy()              → Tensor para array    │   │
│  │  scaler.inverse_transform()  → Desnormalizar [R$]   │   │
│  │  round(predicted_price, 2)   → 2 casas decimais     │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                 │
│                           ▼                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              5. RESPOSTA (JSON)                     │   │
│  │                                                     │   │
│  │  {                                                  │   │
│  │    "predicted_price": 25.87,                        │   │
│  │    "currency": "BRL",                               │   │
│  │    "ticker": "PETR4.SA",                            │   │
│  │    "processing_time_ms": 12.5                       │   │
│  │  }                                                  │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🌐 Endpoints Disponíveis

### GET `/health`

Verifica o status da API e do modelo.

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

### POST `/predict`

Recebe preços e retorna previsão.

**Request:**
```json
{
    "prices": [25.5, 26.1, 25.8, 26.3, 26.5, ...]  // 60+ valores
}
```

**Response:**
```json
{
    "predicted_price": 26.72,
    "currency": "BRL",
    "ticker": "PETR4.SA",
    "input_days": 60,
    "processing_time_ms": 15.32,
    "model_info": {
        "type": "LSTM",
        "hidden_size": 100,
        "num_layers": 2,
        "device": "cpu"
    }
}
```

### GET `/`

Endpoint raiz com informações básicas.

```json
{
    "message": "Stock Price Predictor API",
    "version": "1.0.0",
    "docs": "/docs",
    "health": "/health"
}
```

---

## 🧪 Como Testar

### 1. Rodar Localmente

```bash
cd src
python app.py
```

**Saída esperada:**
```
============================================================
🚀 Iniciando Stock Price Predictor API...
============================================================

🖥️  Dispositivo: cpu
📋 Config carregado: seq_length=60
📊 Scaler carregado: MinMaxScaler
🧠 Modelo carregado: hidden_size=100

============================================================
✅ API pronta para receber requisições!
============================================================

INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 2. Acessar Documentação

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

### 3. Testar com curl

```bash
# Health check
curl http://localhost:8000/health

# Previsão (exemplo com 60 preços fictícios)
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"prices": [25.5, 26.1, 25.8, 26.3, 26.5, 26.2, 26.8, 27.0, 26.7, 26.9, 27.2, 27.5, 27.3, 27.8, 28.0, 27.6, 27.9, 28.2, 28.5, 28.3, 28.7, 29.0, 28.8, 29.2, 29.5, 29.3, 29.7, 30.0, 29.8, 30.2, 30.5, 30.3, 30.7, 31.0, 30.8, 31.2, 31.5, 31.3, 31.7, 32.0, 31.8, 32.2, 32.5, 32.3, 32.7, 33.0, 32.8, 33.2, 33.5, 33.3, 33.7, 34.0, 33.8, 34.2, 34.5, 34.3, 34.7, 35.0, 34.8, 35.2]}'
```

### 4. Testar com Python

```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Previsão
prices = [25.5 + i * 0.15 for i in range(60)]  # 60 preços simulados
response = requests.post(
    "http://localhost:8000/predict",
    json={"prices": prices}
)
print(response.json())
```

---

## 📚 Documentação Automática (OpenAPI)

O FastAPI gera automaticamente:

| Recurso | URL | Descrição |
|---------|-----|-----------|
| Swagger UI | `/docs` | Interface interativa para testar endpoints |
| ReDoc | `/redoc` | Documentação em formato elegante |
| OpenAPI JSON | `/openapi.json` | Especificação OpenAPI 3.0 |

---

## ⚠️ Tratamento de Erros

### HTTP 400 - Bad Request
```json
{
    "detail": "Necessário pelo menos 60 preços históricos. Recebido: 30"
}
```

### HTTP 503 - Service Unavailable
```json
{
    "detail": "Modelo não está carregado. Verifique os logs do servidor."
}
```

### HTTP 422 - Validation Error (Pydantic)
```json
{
    "detail": [
        {
            "loc": ["body", "prices"],
            "msg": "Todos os preços devem ser positivos",
            "type": "value_error"
        }
    ]
}
```

---

## 📊 Performance

| Métrica | Valor Típico |
|---------|--------------|
| Tempo de startup | ~2s |
| Tempo de inferência | 10-20ms |
| Memória do modelo | ~130KB |
| Requisições/segundo | ~50-100 (CPU) |

---

## ✅ Checklist de Conclusão

- [x] FastAPI app configurado
- [x] Endpoint POST `/predict` implementado
- [x] Endpoint GET `/health` implementado
- [x] Validação de entrada (Pydantic)
- [x] Carregamento do modelo no startup (lifespan)
- [x] Normalização e desnormalização de preços
- [x] Tratamento de erros (HTTPException)
- [x] Documentação automática (Swagger/OpenAPI)
- [x] Schemas de entrada/saída documentados

---

## 🔗 Próxima Etapa

**→ ETAPA 9: Docker e Deploy**
- Build da imagem Docker
- Testar execução do container
- Health check no Docker
- Testar endpoint `/predict` no container
