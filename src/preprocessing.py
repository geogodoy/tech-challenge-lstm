# ═══════════════════════════════════════════════════════════════
# 📌 ETAPA 3: Pré-processamento
# 🎯 Objetivo: Normalizar dados e criar janelas temporais
# ═══════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
from typing import Tuple
import joblib

# Importar configurações da coleta de dados
from data_collection import TICKER, DATA_DIR, load_stock_data

# ══════════════════════════════════════════════════════════════════
# 🔧 CONFIGURAÇÕES
# ══════════════════════════════════════════════════════════════════

# Tamanho da janela temporal (60 dias = ~3 meses de histórico)
SEQ_LENGTH = 60

# Proporção de dados para treino
TRAIN_SPLIT = 0.8

# Diretório para salvar artefatos processados
MODELS_DIR = Path(__file__).parent.parent / "models"


def normalize_data(data: np.ndarray) -> Tuple[np.ndarray, MinMaxScaler]:
    """
    Normaliza os dados entre 0 e 1 usando MinMaxScaler.
    
    Por que: Redes neurais funcionam melhor com valores pequenos e uniformes.
    Isso evita que valores grandes dominem o cálculo do erro.
    
    Args:
        data: Array numpy com os dados originais
        
    Returns:
        Tuple com (dados normalizados, scaler para reverter depois)
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    
    print(f"📊 Dados normalizados:")
    print(f"   Original - Min: {data.min():.2f}, Max: {data.max():.2f}")
    print(f"   Normalizado - Min: {data_scaled.min():.4f}, Max: {data_scaled.max():.4f}")
    
    return data_scaled, scaler


def create_sequences(data: np.ndarray, seq_length: int = SEQ_LENGTH) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cria sequências de entrada (X) e saída (y) para a LSTM.
    
    X: últimos `seq_length` dias (entrada da rede)
    y: dia seguinte (o que queremos prever)
    
    Exemplo com seq_length=3:
        Dados: [10, 11, 12, 13, 14, 15]
        X[0] = [10, 11, 12] → y[0] = 13
        X[1] = [11, 12, 13] → y[1] = 14
        X[2] = [12, 13, 14] → y[2] = 15
    
    Args:
        data: Dados normalizados
        seq_length: Número de dias para usar como entrada
        
    Returns:
        Tuple com (X, y) onde:
            X shape: (amostras, seq_length, 1)
            y shape: (amostras, 1)
    """
    X, y = [], []
    
    for i in range(len(data) - seq_length):
        # Sequência de entrada: dias i até i+seq_length
        X.append(data[i:i+seq_length])
        # Saída: dia i+seq_length (próximo dia)
        y.append(data[i+seq_length])
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"\n📐 Janelas temporais criadas:")
    print(f"   Tamanho da janela: {seq_length} dias")
    print(f"   Total de sequências: {len(X)}")
    print(f"   X shape: {X.shape} (amostras, seq_length, features)")
    print(f"   y shape: {y.shape} (amostras, 1)")
    
    return X, y


def train_test_split(
    X: np.ndarray, 
    y: np.ndarray, 
    train_ratio: float = TRAIN_SPLIT
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Divide os dados em conjuntos de treino e teste.
    
    Por que: Precisamos de dados que o modelo nunca viu para avaliar
    se ele realmente aprendeu padrões ou apenas decorou os dados.
    
    Args:
        X: Dados de entrada
        y: Dados de saída
        train_ratio: Proporção para treino (ex: 0.8 = 80% treino, 20% teste)
        
    Returns:
        Tuple com (X_train, X_test, y_train, y_test)
    """
    split_idx = int(len(X) * train_ratio)
    
    X_train = X[:split_idx]
    X_test = X[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]
    
    print(f"\n✂️ Divisão treino/teste ({int(train_ratio*100)}/{int((1-train_ratio)*100)}):")
    print(f"   Treino: {len(X_train)} amostras")
    print(f"   Teste:  {len(X_test)} amostras")
    
    return X_train, X_test, y_train, y_test


def to_tensors(
    X_train: np.ndarray, 
    X_test: np.ndarray, 
    y_train: np.ndarray, 
    y_test: np.ndarray
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Converte arrays NumPy para tensores PyTorch.
    
    Por que: PyTorch trabalha com tensores, que são estruturas otimizadas
    para operações matriciais em GPU/CPU.
    
    Args:
        X_train, X_test, y_train, y_test: Arrays NumPy
        
    Returns:
        Tuple com tensores PyTorch (FloatTensor para precisão de 32 bits)
    """
    X_train_t = torch.FloatTensor(X_train)
    X_test_t = torch.FloatTensor(X_test)
    y_train_t = torch.FloatTensor(y_train)
    y_test_t = torch.FloatTensor(y_test)
    
    print(f"\n🔢 Conversão para tensores PyTorch:")
    print(f"   X_train: {X_train_t.shape} ({X_train_t.dtype})")
    print(f"   y_train: {y_train_t.shape} ({y_train_t.dtype})")
    print(f"   X_test:  {X_test_t.shape} ({X_test_t.dtype})")
    print(f"   y_test:  {y_test_t.shape} ({y_test_t.dtype})")
    
    return X_train_t, X_test_t, y_train_t, y_test_t


def preprocess_data(
    ticker: str = TICKER,
    seq_length: int = SEQ_LENGTH,
    train_ratio: float = TRAIN_SPLIT,
    save_scaler: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, MinMaxScaler]:
    """
    Pipeline completa de pré-processamento.
    
    Executa todas as etapas:
    1. Carrega dados do CSV
    2. Normaliza entre 0-1
    3. Cria janelas temporais
    4. Divide treino/teste
    5. Converte para tensores
    
    Args:
        ticker: Símbolo da ação
        seq_length: Tamanho da janela temporal
        train_ratio: Proporção para treino
        save_scaler: Se True, salva o scaler para uso na inferência
        
    Returns:
        Tuple com (X_train, X_test, y_train, y_test, scaler)
    """
    print("="*60)
    print("📌 ETAPA 3: Pré-processamento de Dados")
    print("="*60)
    
    # 1️⃣ Carregar dados
    df = load_stock_data(ticker)
    
    # 2️⃣ Selecionar apenas a coluna 'Close' (preço de fechamento)
    data = df['Close'].values.reshape(-1, 1)
    print(f"\n📈 Coluna selecionada: 'Close'")
    print(f"   Total de registros: {len(data)}")
    
    # 3️⃣ Normalizar os dados entre 0 e 1
    data_scaled, scaler = normalize_data(data)
    
    # 4️⃣ Criar janelas deslizantes (sequências)
    X, y = create_sequences(data_scaled, seq_length)
    
    # 5️⃣ Dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_ratio)
    
    # 6️⃣ Converter para tensores PyTorch
    X_train_t, X_test_t, y_train_t, y_test_t = to_tensors(X_train, X_test, y_train, y_test)
    
    # 7️⃣ Salvar scaler e configurações para uso posterior
    if save_scaler:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Salvar scaler
        scaler_path = MODELS_DIR / "scaler.pkl"
        joblib.dump(scaler, scaler_path)
        print(f"\n💾 Scaler salvo em: {scaler_path}")
        
        # Salvar configurações
        config = {
            'seq_length': seq_length,
            'ticker': ticker,
            'train_ratio': train_ratio,
            'input_size': 1,  # Número de features (apenas Close)
            'n_train_samples': len(X_train),
            'n_test_samples': len(X_test)
        }
        config_path = MODELS_DIR / "config.pkl"
        joblib.dump(config, config_path)
        print(f"💾 Config salvo em: {config_path}")
    
    return X_train_t, X_test_t, y_train_t, y_test_t, scaler


# ══════════════════════════════════════════════════════════════════
# 🚀 EXECUÇÃO
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Executar pré-processamento
    X_train, X_test, y_train, y_test, scaler = preprocess_data()
    
    # ✅ Checkpoint: Dados prontos para alimentar a LSTM!
    print("\n" + "="*60)
    print("🎉 CHECKPOINT: Dados prontos para a LSTM!")
    print("="*60)
    print(f"\n📋 Resumo final:")
    print(f"   X_train shape: {X_train.shape}")
    print(f"   y_train shape: {y_train.shape}")
    print(f"   X_test shape:  {X_test.shape}")
    print(f"   y_test shape:  {y_test.shape}")
