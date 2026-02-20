# ═══════════════════════════════════════════════════════════════
# 📌 ETAPA 8: API FastAPI
# 🎯 Objetivo: Servir o modelo LSTM via endpoint REST
# ═══════════════════════════════════════════════════════════════

import os
import time
from pathlib import Path
from typing import List, Optional
from contextlib import asynccontextmanager

import torch
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

try:
    from model import StockLSTM
except ImportError:
    from src.model import StockLSTM

# ══════════════════════════════════════════════════════════════════
# 📁 CONFIGURAÇÃO DE PATHS
# ══════════════════════════════════════════════════════════════════

BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"

MODEL_PATH = MODELS_DIR / "model_lstm.pth"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
CONFIG_PATH = MODELS_DIR / "config.pkl"

# ══════════════════════════════════════════════════════════════════
# 🗄️ ESTADO GLOBAL DA APLICAÇÃO
# ══════════════════════════════════════════════════════════════════

class ModelState:
    """Armazena o estado do modelo carregado."""
    model: Optional[StockLSTM] = None
    scaler = None
    config: Optional[dict] = None
    device: str = "cpu"
    is_loaded: bool = False

state = ModelState()

# ══════════════════════════════════════════════════════════════════
# 🚀 LIFESPAN (STARTUP/SHUTDOWN)
# ══════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gerencia o ciclo de vida da aplicação."""
    # STARTUP: Carregar modelo
    print("="*60)
    print("🚀 Iniciando Stock Price Predictor API...")
    print("="*60)
    
    try:
        # Verificar se arquivos existem
        for path, name in [(MODEL_PATH, "Modelo"), (SCALER_PATH, "Scaler"), (CONFIG_PATH, "Config")]:
            if not path.exists():
                raise FileNotFoundError(f"{name} não encontrado: {path}")
        
        # Configurar dispositivo
        state.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\n🖥️  Dispositivo: {state.device}")
        
        # Carregar configurações
        state.config = joblib.load(CONFIG_PATH)
        print(f"📋 Config carregado: seq_length={state.config.get('seq_length', 60)}")
        
        # Carregar scaler
        state.scaler = joblib.load(SCALER_PATH)
        print(f"📊 Scaler carregado: MinMaxScaler")
        
        # Carregar modelo
        checkpoint = torch.load(MODEL_PATH, map_location=state.device, weights_only=False)
        model_config = checkpoint.get('model_config', {})
        
        # Criar instância do modelo com a configuração salva
        state.model = StockLSTM(
            input_size=model_config.get('input_size', 1),
            hidden_size=model_config.get('hidden_size', 100),
            num_layers=model_config.get('num_layers', 2),
            dropout=model_config.get('dropout', 0.2)
        )
        
        # Carregar pesos
        state.model.load_state_dict(checkpoint['model_state_dict'])
        state.model.to(state.device)
        state.model.eval()
        
        state.is_loaded = True
        
        print(f"🧠 Modelo carregado: hidden_size={model_config.get('hidden_size', 100)}")
        print("\n" + "="*60)
        print("✅ API pronta para receber requisições!")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ Erro ao carregar modelo: {e}")
        state.is_loaded = False
    
    yield
    
    # SHUTDOWN
    print("\n🛑 Encerrando API...")

# ══════════════════════════════════════════════════════════════════
# 📦 SCHEMAS (PYDANTIC)
# ══════════════════════════════════════════════════════════════════

class PredictionRequest(BaseModel):
    """Schema de entrada para previsão."""
    prices: List[float] = Field(
        ...,
        description="Lista com os últimos N preços de fechamento (mínimo: seq_length dias)",
        examples=[[25.50, 26.10, 25.80, 26.30, 26.50]]
    )
    
    @field_validator('prices')
    @classmethod
    def validate_prices(cls, v):
        if not v:
            raise ValueError("Lista de preços não pode estar vazia")
        if any(p <= 0 for p in v):
            raise ValueError("Todos os preços devem ser positivos")
        return v


class PredictionResponse(BaseModel):
    """Schema de saída para previsão."""
    predicted_price: float = Field(..., description="Preço previsto para o próximo dia")
    currency: str = Field(default="BRL", description="Moeda")
    ticker: str = Field(..., description="Ticker da ação")
    input_days: int = Field(..., description="Quantidade de dias usados na previsão")
    processing_time_ms: float = Field(..., description="Tempo de processamento em ms")
    model_info: dict = Field(..., description="Informações do modelo")


class HealthResponse(BaseModel):
    """Schema de resposta do health check."""
    status: str
    model_loaded: bool
    device: str
    ticker: Optional[str] = None
    seq_length: Optional[int] = None


class ErrorResponse(BaseModel):
    """Schema de resposta de erro."""
    detail: str
    error_type: str

# ══════════════════════════════════════════════════════════════════
# 🌐 APLICAÇÃO FASTAPI
# ══════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Stock Price Predictor API",
    description="""
    ## 📈 API de Previsão de Preços de Ações com LSTM
    
    Esta API utiliza um modelo de Deep Learning (LSTM) para prever o preço 
    de fechamento do próximo dia com base nos últimos 60 dias de histórico.
    
    ### Endpoints:
    - **POST /predict**: Envia preços históricos e recebe a previsão
    - **GET /health**: Verifica o status da API e do modelo
    
    ### Tech Challenge - Fase 4
    Pós-graduação em Machine Learning Engineering
    """,
    version="1.0.0",
    lifespan=lifespan
)

# ══════════════════════════════════════════════════════════════════
# 🔌 ENDPOINTS
# ══════════════════════════════════════════════════════════════════

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Status"],
    summary="Health Check",
    description="Verifica se a API está funcionando e o modelo está carregado."
)
async def health_check():
    """Retorna o status da API e do modelo."""
    return HealthResponse(
        status="healthy" if state.is_loaded else "unhealthy",
        model_loaded=state.is_loaded,
        device=state.device,
        ticker=state.config.get('ticker') if state.config else None,
        seq_length=state.config.get('seq_length') if state.config else None
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Previsão"],
    summary="Prever Próximo Preço",
    description="Recebe uma lista de preços históricos e retorna a previsão do próximo dia.",
    responses={
        400: {"model": ErrorResponse, "description": "Dados inválidos"},
        503: {"model": ErrorResponse, "description": "Modelo não carregado"}
    }
)
async def predict(request: PredictionRequest):
    """
    Realiza a previsão do preço do próximo dia.
    
    - **prices**: Lista com pelo menos `seq_length` (60) preços de fechamento
    
    Retorna o preço previsto para o próximo dia útil.
    """
    start_time = time.time()
    
    # Verificar se modelo está carregado
    if not state.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo não está carregado. Verifique os logs do servidor."
        )
    
    seq_length = state.config.get('seq_length', 60)
    
    # Validar quantidade de preços
    if len(request.prices) < seq_length:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Necessário pelo menos {seq_length} preços históricos. Recebido: {len(request.prices)}"
        )
    
    try:
        # Pegar os últimos seq_length preços
        prices = np.array(request.prices[-seq_length:]).reshape(-1, 1)
        
        # Normalizar usando o scaler treinado
        prices_scaled = state.scaler.transform(prices)
        
        # Converter para tensor PyTorch
        X = torch.FloatTensor(prices_scaled).unsqueeze(0).to(state.device)
        
        # Fazer previsão
        with torch.no_grad():
            prediction_scaled = state.model(X).cpu().numpy()
        
        # Reverter normalização para obter preço em R$
        predicted_price = state.scaler.inverse_transform(prediction_scaled)[0][0]
        
        # Calcular tempo de processamento
        processing_time = (time.time() - start_time) * 1000
        
        return PredictionResponse(
            predicted_price=round(float(predicted_price), 2),
            currency="BRL",
            ticker=state.config.get('ticker', 'PETR4.SA'),
            input_days=seq_length,
            processing_time_ms=round(processing_time, 2),
            model_info={
                "type": "LSTM",
                "hidden_size": state.model.hidden_size,
                "num_layers": state.model.num_layers,
                "device": state.device
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao processar previsão: {str(e)}"
        )


@app.get(
    "/",
    tags=["Status"],
    summary="Root",
    description="Redireciona para a documentação da API."
)
async def root():
    """Endpoint raiz com informações básicas."""
    return {
        "message": "Stock Price Predictor API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

# ══════════════════════════════════════════════════════════════════
# 🚀 EXECUÇÃO LOCAL
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*60)
    print("📌 ETAPA 8: API FastAPI")
    print("="*60)
    print("\n🌐 Iniciando servidor em http://localhost:8000")
    print("📚 Documentação disponível em http://localhost:8000/docs")
    print("\n💡 Pressione Ctrl+C para encerrar\n")
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
