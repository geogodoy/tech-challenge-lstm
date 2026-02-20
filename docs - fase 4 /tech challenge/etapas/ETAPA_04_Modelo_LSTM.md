# 📌 ETAPA 4: Modelo LSTM

## 📋 Resumo
| Item | Valor |
|------|-------|
| **Status** | ✅ Concluída |
| **Data** | 2026-02-17 |
| **Tempo Estimado** | 45 min |
| **Tempo Real** | ~10 min |

---

## 🎯 Objetivo
Definir a arquitetura da rede neural LSTM para previsão de preços de ações.

---

## 🎓 Conexão com as Aulas

### Aula 03 - Arquiteturas de Redes Neurais Profundas
**Arquivo:** `docs - fase 4 /etapa 1 - redes neurais e deep learning/Aula 03 - Arquiteturas de Redes Neurais Profundas.txt`

#### Por que LSTM e não RNN comum?

> *"As RNNs enfrentam desafios como o problema do desvanecimento e da explosão de gradientes durante o treinamento, especialmente em sequências longas."* (Linha ~439)

> *"Para combater esses problemas, variantes de RNNs como Long Short-Term Memory (LSTM) e Gated Recurrent Units (GRU) foram desenvolvidas. Estas arquiteturas incluem mecanismos de portões que regulam o fluxo de informações."* (Linha ~443)

#### Estrutura da LSTM

> *"Eles permitem que a rede aprenda quais dados no estado devem ser lembrados ou esquecidos, melhorando a capacidade da rede de aprender dependências de longo prazo."* (Linha ~443)

#### Exemplo de código na aula (Linhas 114-129):
```python
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(input_size=28, hidden_size=128,
                           num_layers=2, batch_first=True)
        self.output_layer = nn.Linear(128, 10)
```

---

## 📁 Arquivo Implementado

### `src/model.py`

#### Classe Principal: `StockLSTM` (Linhas 14-127)

```python
class StockLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, 
                 num_layers=2, dropout=0.2):
        super(StockLSTM, self).__init__()
        
        # 1️⃣ Camada LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 2️⃣ Dropout para regularização
        self.dropout = nn.Dropout(dropout)
        
        # 3️⃣ Camada Linear para saída
        self.linear = nn.Linear(hidden_size, 1)
```

---

## 🧠 Arquitetura Detalhada

```
┌─────────────────────────────────────────────────────────────┐
│                    StockLSTM - Arquitetura                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  INPUT: (batch_size, seq_length, input_size)                │
│         Exemplo: (32, 60, 1)                                │
│         32 amostras, 60 dias, 1 feature (Close)             │
│                           │                                 │
│                           ▼                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                    nn.LSTM                          │   │
│  │  • input_size: 1 (apenas preço Close)               │   │
│  │  • hidden_size: 50 (dimensão do estado oculto)      │   │
│  │  • num_layers: 2 (LSTMs empilhadas)                 │   │
│  │  • batch_first: True                                │   │
│  │  • dropout: 0.2 (entre camadas)                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                 │
│                           ▼                                 │
│         lstm_out: (batch, seq_length, hidden_size)          │
│         Pegamos apenas: lstm_out[:, -1, :]                  │
│         = último passo temporal                             │
│                           │                                 │
│                           ▼                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                  nn.Dropout(0.2)                    │   │
│  │  Desliga 20% dos neurônios aleatoriamente           │   │
│  │  → Regularização para evitar overfitting            │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                 │
│                           ▼                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │               nn.Linear(50 → 1)                     │   │
│  │  Transforma hidden_size em preço previsto           │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                 │
│                           ▼                                 │
│  OUTPUT: (batch_size, 1) = Preço previsto                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔬 Hiperparâmetros e Conexão com a Teoria

### Tabela de Hiperparâmetros

| Hiperparâmetro | Valor | Justificativa | Referência na Aula |
|----------------|-------|---------------|-------------------|
| `input_size` | 1 | Apenas preço Close | Feature única para simplificar |
| `hidden_size` | 50 | Capacidade de memória | Similar ao exemplo da aula (128) |
| `num_layers` | 2 | Deep LSTM | "RNNs empilhadas" (linha ~446) |
| `dropout` | 0.2 | Regularização | Linha ~331, ~445 |
| `batch_first` | True | Formato (batch, seq, features) | Padrão PyTorch |

### Dropout - Conexão com a Aula

> *"A regularização também é uma parte crucial do treinamento... Métodos como dropout são frequentemente adaptados para RNNs, sendo aplicados não apenas às entradas e saídas da rede, mas também entre os passos de tempo."* (Linha ~445)

**No código:**
```python
# Linha 78: Dropout ENTRE camadas LSTM
dropout=dropout if num_layers > 1 else 0

# Linha 83: Dropout APÓS a LSTM (antes da camada linear)
self.dropout = nn.Dropout(dropout)
```

---

## 💻 Método `forward()` Detalhado

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    # x shape: (batch_size, seq_length, input_size)
    # Exemplo: (32, 60, 1)
    
    # 1. Passar pela LSTM
    lstm_out, (h_n, c_n) = self.lstm(x)
    # lstm_out: (32, 60, 50) - saída de cada passo temporal
    # h_n: (2, 32, 50) - hidden state final (por camada)
    # c_n: (2, 32, 50) - cell state final (por camada)
    
    # 2. Pegar apenas o ÚLTIMO passo da sequência
    last_output = lstm_out[:, -1, :]
    # last_output: (32, 50) - a "memória" após processar 60 dias
    
    # 3. Aplicar dropout
    out = self.dropout(last_output)
    
    # 4. Camada linear para previsão
    prediction = self.linear(out)
    # prediction: (32, 1) - preço previsto para cada amostra
    
    return prediction
```

**Por que `lstm_out[:, -1, :]`?**
- Queremos a "memória" da LSTM **após** processar toda a sequência
- O último hidden state contém informação acumulada de todos os 60 dias
- É como perguntar: "dado todo esse histórico, qual é a previsão?"

---

## 📊 Estatísticas do Modelo

```
StockLSTM(
  (lstm): LSTM(1, 50, num_layers=2, batch_first=True, dropout=0.2)
  (dropout): Dropout(p=0.2, inplace=False)
  (linear): Linear(in_features=50, out_features=1, bias=True)
)

Total de parâmetros treináveis: 31,051
```

### Cálculo dos Parâmetros

**Camada LSTM (2 camadas):**
- Camada 1: 4 × (1 × 50 + 50 × 50 + 50 + 50) = 4 × (50 + 2500 + 100) = 10,600
- Camada 2: 4 × (50 × 50 + 50 × 50 + 50 + 50) = 4 × (5000 + 100) = 20,400
- Total LSTM: ~31,000

**Camada Linear:**
- 50 × 1 + 1 (bias) = 51

---

## 🧪 Teste do Modelo

```python
# Criar tensor de teste
x_test = torch.randn(32, 60, 1)  # 32 amostras, 60 dias, 1 feature
print(f"Input shape:  {x_test.shape}")   # torch.Size([32, 60, 1])

# Forward pass
output = model(x_test)
print(f"Output shape: {output.shape}")   # torch.Size([32, 1])
```

---

## ✅ Checklist de Conclusão

- [x] Classe `StockLSTM` criada herdando `nn.Module`
- [x] Camada LSTM configurada (2 layers, hidden=50)
- [x] Dropout implementado (0.2)
- [x] Camada Linear para output
- [x] Método `forward()` implementado
- [x] Método `get_config()` para serialização
- [x] Factory function `create_model()`
- [x] Teste com dados sintéticos passou

---

## 🔗 Próxima Etapa

**→ ETAPA 5: Treinamento**
- Configurar MSELoss e Adam
- Implementar loop de treinamento
- Monitorar train_loss e val_loss
