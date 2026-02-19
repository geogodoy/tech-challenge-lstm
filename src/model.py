# ═══════════════════════════════════════════════════════════════
# 📌 ETAPA 4: Modelo LSTM
# 🎯 Objetivo: Definir a arquitetura da rede neural
# 📍 Referência: Guia linhas 186-239
# ═══════════════════════════════════════════════════════════════

import torch
import torch.nn as nn

# ══════════════════════════════════════════════════════════════════
# 🧠 ARQUITETURA LSTM
# ══════════════════════════════════════════════════════════════════

class StockLSTM(nn.Module):
    """
    Modelo LSTM para previsão de preços de ações.
    
    Arquitetura:
    ┌─────────────────────────────────────────────────────────────┐
    │  Input: (batch_size, seq_length, input_size)                │
    │         Ex: (32, 60, 1) = 32 amostras, 60 dias, 1 feature   │
    │                                                              │
    │  ┌─────────┐                                                │
    │  │  LSTM   │  ← Captura padrões temporais (memória)         │
    │  │ layers  │    hidden_size=50, num_layers=2                │
    │  └────┬────┘                                                │
    │       │                                                      │
    │  ┌────▼────┐                                                │
    │  │ Dropout │  ← Regularização (evita overfitting)           │
    │  │  (0.2)  │    Desliga 20% dos neurônios aleatoriamente    │
    │  └────┬────┘                                                │
    │       │                                                      │
    │  ┌────▼────┐                                                │
    │  │ Linear  │  ← Transforma hidden_size → 1 (preço)          │
    │  │ (50→1)  │                                                │
    │  └────┬────┘                                                │
    │       │                                                      │
    │  Output: (batch_size, 1) = Preço previsto                   │
    └─────────────────────────────────────────────────────────────┘
    
    Por que LSTM?
    - RNNs comuns "esquecem" muito rápido (vanishing gradient)
    - LSTM possui "portões" (gates) que decidem o que manter na memória
    - Ideal para séries temporais como preços de ações
    """
    
    def __init__(
        self, 
        input_size: int = 1, 
        hidden_size: int = 50, 
        num_layers: int = 2, 
        dropout: float = 0.2
    ):
        """
        Inicializa o modelo LSTM.
        
        Args:
            input_size: Número de features de entrada (1 = apenas Close)
            hidden_size: Dimensão do estado oculto da LSTM
            num_layers: Número de camadas LSTM empilhadas
            dropout: Taxa de dropout para regularização
        """
        super(StockLSTM, self).__init__()
        
        # Salvar configurações para uso posterior
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout
        
        # 1️⃣ Camada LSTM: O coração que guarda o contexto temporal
        # batch_first=True: entrada no formato (batch, seq, features)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0  # Dropout entre camadas LSTM
        )
        
        # 2️⃣ Dropout: Evita que o modelo "decore" os preços passados (overfitting)
        # Durante o treino, desliga neurônios aleatoriamente
        self.dropout = nn.Dropout(dropout)
        
        # 3️⃣ Camada Linear: Transforma a memória da LSTM no preço final previsto
        # Entrada: hidden_size (50), Saída: 1 (preço)
        self.linear = nn.Linear(hidden_size, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: processa a entrada através da rede.
        
        Args:
            x: Tensor de entrada com shape (batch_size, seq_length, input_size)
               Exemplo: (32, 60, 1) = 32 amostras de 60 dias cada
               
        Returns:
            Tensor com previsões de shape (batch_size, 1)
        """
        # Passar pela LSTM
        # lstm_out shape: (batch_size, seq_length, hidden_size)
        # h_n shape: (num_layers, batch_size, hidden_size) - último hidden state
        # c_n shape: (num_layers, batch_size, hidden_size) - último cell state
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Pegar apenas o último passo da sequência
        # Queremos a "memória" após processar todos os 60 dias
        # last_output shape: (batch_size, hidden_size)
        last_output = lstm_out[:, -1, :]
        
        # Aplicar dropout (regularização)
        out = self.dropout(last_output)
        
        # Camada linear para obter o preço previsto
        # prediction shape: (batch_size, 1)
        prediction = self.linear(out)
        
        return prediction
    
    def get_config(self) -> dict:
        """Retorna configuração do modelo para salvamento."""
        return {
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'dropout': self.dropout_rate
        }


def create_model(
    input_size: int = 1,
    hidden_size: int = 100,  # Otimizado: era 50, agora 100 para MAPE < 5%
    num_layers: int = 2,
    dropout: float = 0.2,
    device: str = None
) -> StockLSTM:
    """
    Factory function para criar e configurar o modelo.
    
    Args:
        input_size: Número de features
        hidden_size: Dimensão do estado oculto
        num_layers: Número de camadas LSTM
        dropout: Taxa de dropout
        device: Dispositivo ('cuda', 'cpu', ou None para auto)
        
    Returns:
        Modelo configurado e movido para o dispositivo apropriado
    """
    # Determinar dispositivo automaticamente se não especificado
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Criar modelo
    model = StockLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    )
    
    # Mover para dispositivo
    model = model.to(device)
    
    return model


def count_parameters(model: nn.Module) -> int:
    """Conta o número de parâmetros treináveis do modelo."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ══════════════════════════════════════════════════════════════════
# 🚀 EXECUÇÃO
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("="*60)
    print("📌 ETAPA 4: Construção do Modelo LSTM")
    print("="*60)
    
    # Configurar dispositivo
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n🖥️ Dispositivo: {device}")
    if device == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Criar modelo
    model = create_model(device=device)
    
    # Exibir arquitetura
    print(f"\n🧠 Arquitetura do modelo:")
    print(model)
    
    # Contar parâmetros
    n_params = count_parameters(model)
    print(f"\n📊 Total de parâmetros treináveis: {n_params:,}")
    
    # Teste com dados simulados
    print(f"\n🧪 Teste com dados simulados:")
    batch_size = 32
    seq_length = 60
    input_size = 1
    
    # Criar tensor de entrada aleatório
    x_test = torch.randn(batch_size, seq_length, input_size).to(device)
    print(f"   Input shape:  {x_test.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(x_test)
    print(f"   Output shape: {output.shape}")
    
    # ✅ Checkpoint
    print("\n" + "="*60)
    print("🎉 CHECKPOINT: O cérebro nasceu!")
    print("="*60)
    print(f"\n📋 Configuração do modelo:")
    for key, value in model.get_config().items():
        print(f"   {key}: {value}")
