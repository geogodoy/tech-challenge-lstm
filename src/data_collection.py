# ═══════════════════════════════════════════════════════════════
# ETAPA 2: Coleta de Dados
# Objetivo: Baixar precos historicos usando a biblioteca yfinance
# ═══════════════════════════════════════════════════════════════

import yfinance as yf
import pandas as pd
from pathlib import Path

# ══════════════════════════════════════════════════════════════════
# CONFIGURACOES
# ══════════════════════════════════════════════════════════════════

# Definir o ativo e o periodo
# Por que: Precisamos de uma janela temporal longa para a LSTM aprender padrões
TICKER = "PETR4.SA"  # Petrobras - Ação brasileira
START_DATE = "2018-01-01"
END_DATE = "2024-01-01"

# Diretório para salvar os dados
DATA_DIR = Path(__file__).parent.parent / "data"


def download_stock_data(
    ticker: str = TICKER,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    save: bool = True
) -> pd.DataFrame:
    """
    Baixa dados históricos de uma ação usando yfinance.
    
    Args:
        ticker: Símbolo da ação (ex: 'PETR4.SA', 'AAPL', 'MSFT')
        start_date: Data inicial no formato 'YYYY-MM-DD'
        end_date: Data final no formato 'YYYY-MM-DD'
        save: Se True, salva os dados em CSV
        
    Returns:
        DataFrame com os dados históricos (Open, High, Low, Close, Volume, etc.)
    """
    print(f"Baixando dados de {ticker}...")
    print(f"   Periodo: {start_date} ate {end_date}")
    
    # Baixar os dados
    df = yf.download(ticker, start=start_date, end=end_date, progress=True)
    
    # Verificar os dados
    if df.empty:
        raise ValueError(f"Nenhum dado encontrado para {ticker}")
    
    print(f"\nDados baixados com sucesso!")
    print(f"   Shape: {df.shape}")
    print(f"   Período real: {df.index[0].strftime('%Y-%m-%d')} até {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"   Colunas: {list(df.columns)}")
    
    # Mostrar primeiras e ultimas linhas
    print(f"\nPrimeiras linhas:")
    print(df.head())
    print(f"\nUltimas linhas:")
    print(df.tail())
    
    # Estatisticas basicas
    print(f"\nEstatisticas do preco de fechamento (Close):")
    close_col = ('Close', ticker) if isinstance(df.columns, pd.MultiIndex) else 'Close'
    print(f"   Mínimo: R$ {df[close_col].min():.2f}")
    print(f"   Máximo: R$ {df[close_col].max():.2f}")
    print(f"   Média:  R$ {df[close_col].mean():.2f}")
    
    # Salvar para uso posterior
    if save:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        # Flatten MultiIndex columns se necessario
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        
        filepath = DATA_DIR / f"data_{ticker.replace('.', '_')}.csv"
        df.to_csv(filepath)
        print(f"\nDados salvos em: {filepath}")
    
    return df


def load_stock_data(ticker: str = TICKER) -> pd.DataFrame:
    """
    Carrega dados previamente salvos de uma ação.
    
    Args:
        ticker: Símbolo da ação
        
    Returns:
        DataFrame com os dados históricos
    """
    filepath = DATA_DIR / f"data_{ticker.replace('.', '_')}.csv"
    
    if not filepath.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {filepath}")
    
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    print(f"Dados carregados de: {filepath}")
    print(f"   Shape: {df.shape}")
    
    return df


# ══════════════════════════════════════════════════════════════════
# EXECUCAO
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Executar coleta de dados
    df = download_stock_data()
    
    # Checkpoint: Se df.head() mostrar precos, voce tem os dados!
    print("\n" + "="*60)
    print("CHECKPOINT: Dados coletados com sucesso!")
    print("="*60)
