# 📌 ETAPA 5: Treinamento

## 📋 Resumo
| Item | Valor |
|------|-------|
| **Status** | ✅ Concluída + Otimizada |
| **Data Inicial** | 2026-02-17 |
| **Data Otimização** | 2026-02-19 |
| **Tempo Estimado** | 45 min |
| **Tempo Real** | ~15 min (inicial) + ~30 min (otimização) |

---

## 🎯 Objetivo
Treinar o modelo LSTM ajustando os pesos através do algoritmo de backpropagation, monitorando as métricas de treino e validação.

---

## 🎓 Conexão com as Aulas

### Aula 02 - Teoria das Redes Neurais Profundas
**Conceitos fundamentais aplicados:**

#### Backpropagation e Gradiente Descendente
> *"A eficiência e a eficácia do método de backpropagation são amplamente impactadas pela escolha do algoritmo de otimização, como SGD, Adam ou RMSprop."* (Aula 03, linha ~330)

### Aula 03 - Arquiteturas de Redes Neurais Profundas
**Arquivo:** `docs - fase 4 /etapa 1 - redes neurais e deep learning/Aula 03 - Arquiteturas de Redes Neurais Profundas.txt`

#### Backpropagation Through Time (BPTT)
> *"Do ponto de vista da otimização, RNNs são geralmente treinadas usando variantes do algoritmo de backpropagation chamado Backpropagation Through Time (BPTT). O BPTT desenrola a rede no tempo e aplica o algoritmo de gradiente descendente."* (Linha ~444)

#### Otimizador Adam
> *"Em termos de otimização, as RNNs frequentemente utilizam técnicas como a normalização de gradientes ou o uso de algoritmos de otimização robustos, como RMSprop ou Adam, que são mais eficazes em lidar com as rápidas mudanças nos gradientes."* (Linha ~506)

#### Função de Perda (MSE)
> *"A atualização dos pesos w em cada camada, utilizando o gradiente descendente, é dada por: w_{n+1} = w_n - η(∂L/∂w)"* (Linha ~362-364)

---

## 📁 Arquivo Implementado

### `src/train.py`

#### Estrutura do Código

```python
# Linhas 1-5: Cabeçalho
# ═══════════════════════════════════════════════════════════════
# 📌 ETAPA 5: Treinamento
# 🎯 Objetivo: Treinar o modelo ajustando os pesos
# ═══════════════════════════════════════════════════════════════
```

#### Configurações (Linhas 22-29)
```python
EPOCHS = 100          # Número de iterações completas pelo dataset
LEARNING_RATE = 0.001 # Taxa de aprendizado (quão rápido os pesos mudam)
BATCH_SIZE = None     # None = batch gradient descent (todo dataset)
```

---

## 🔬 Função Principal: `train_model()` (Linhas 32-153)

### Estrutura do Loop de Treinamento

```python
def train_model(model, X_train, y_train, X_test, y_test, 
                epochs=100, learning_rate=0.001, device=None):
    
    # 1️⃣ Configurar dispositivo (GPU/CPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    # 2️⃣ Mover dados para o dispositivo
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    
    # 3️⃣ Definir função de perda e otimizador
    criterion = nn.MSELoss()                          # Mean Squared Error
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 4️⃣ Loop de treinamento
    for epoch in range(epochs):
        # FASE DE TREINO
        model.train()
        outputs = model(X_train)           # Forward pass
        loss = criterion(outputs, y_train) # Calcular erro
        
        optimizer.zero_grad()              # Limpar gradientes
        loss.backward()                    # Backpropagation
        optimizer.step()                   # Atualizar pesos
        
        # FASE DE VALIDAÇÃO
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test)
            val_loss = criterion(val_outputs, y_test)
    
    return model, train_losses, val_losses
```

---

## 📊 Conexão Código ↔ Teoria

### Tabela de Mapeamento

| Conceito na Aula | Linha na Aula | Código | Linha no Código |
|------------------|---------------|--------|-----------------|
| "Gradiente descendente" | ~330 | `optimizer.step()` | 113 |
| "Backpropagation" | ~330, ~444 | `loss.backward()` | 112 |
| "Adam optimizer" | ~506 | `optim.Adam(...)` | 75 |
| "Função de perda" | ~362 | `nn.MSELoss()` | 72 |
| "Taxa de aprendizado η" | ~364 | `lr=learning_rate` | 75 |

### MSELoss - Por que escolhemos?

> *"A função de perda MSE (Mean Squared Error) é ideal para problemas de regressão onde queremos minimizar a diferença quadrática entre previsões e valores reais."*

```python
# MSE = (1/n) * Σ(y_pred - y_real)²
criterion = nn.MSELoss()
loss = criterion(outputs, y_train)
```

### Adam - Por que escolhemos?

> *"Adam: Otimizador adaptativo que ajusta a taxa de aprendizado de cada parâmetro de forma adaptativa."* (Aula 03, linha ~580)

**Vantagens do Adam:**
- Combina vantagens de AdaGrad e RMSprop
- Funciona bem com dados esparsos
- Ajusta automaticamente o learning rate por parâmetro
- Ideal para RNNs/LSTMs

---

## 🔄 Fluxo do Treinamento

```
┌─────────────────────────────────────────────────────────────┐
│                    LOOP DE TREINAMENTO                      │
│                      (100 épocas)                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              FASE DE TREINO (model.train())         │   │
│  │                                                     │   │
│  │  1. Forward Pass                                    │   │
│  │     outputs = model(X_train)                        │   │
│  │     → Previsões do modelo                           │   │
│  │                                                     │   │
│  │  2. Calcular Loss                                   │   │
│  │     loss = MSE(outputs, y_train)                    │   │
│  │     → Quão errado o modelo está                     │   │
│  │                                                     │   │
│  │  3. Backward Pass (Backpropagation)                 │   │
│  │     optimizer.zero_grad()  → Limpa gradientes       │   │
│  │     loss.backward()        → Calcula ∂L/∂w          │   │
│  │     optimizer.step()       → w = w - η * ∂L/∂w      │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                 │
│                           ▼                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │            FASE DE VALIDAÇÃO (model.eval())         │   │
│  │                                                     │   │
│  │  with torch.no_grad():  → Não calcula gradientes    │   │
│  │      val_outputs = model(X_test)                    │   │
│  │      val_loss = MSE(val_outputs, y_test)            │   │
│  │                                                     │   │
│  │  → Monitora overfitting (se val_loss sobe)          │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 📈 Resultados do Treinamento

### Versão Inicial (v1) - hidden_size=50

```
============================================================
🏋️ Treinamento v1 (hidden_size=50)
============================================================

Epoch [ 10/100] | Train Loss: 0.027582 | Val Loss: 0.293079
Epoch [ 50/100] | Train Loss: 0.012421 | Val Loss: 0.150984
Epoch [100/100] | Train Loss: 0.002085 | Val Loss: 0.003514

📊 Resumo v1:
   Train Loss final: 0.002085
   Val Loss final: 0.003514
   MAPE resultante: 6.74% (Bom, mas abaixo da meta)
```

### Versão Otimizada (v2) - hidden_size=100 ✅

Após análise de hiperparâmetros (ver `src/hyperparameter_tuning.py`), identificamos que o **hidden_size** era o fator mais impactante.

```
============================================================
🏋️ Treinamento v2 (hidden_size=100) - MODELO ATUAL
============================================================

Epoch [ 10/100] | Train Loss: 0.014841 | Val Loss: 0.205917
Epoch [ 50/100] | Train Loss: 0.001040 | Val Loss: 0.004649
Epoch [ 70/100] | Train Loss: 0.000830 | Val Loss: 0.001263
Epoch [100/100] | Train Loss: 0.000693 | Val Loss: 0.001367

📊 Resumo v2:
   Tempo total: 30.8s
   Train Loss final: 0.000693
   Val Loss final: 0.001367
   Melhor Val Loss: 0.001190 (época 68)
   MAPE resultante: 3.83% (EXCELENTE!) ✅
```

### Comparativo v1 vs v2

| Métrica | v1 (hidden=50) | v2 (hidden=100) | Melhoria |
|---------|----------------|-----------------|----------|
| Train Loss | 0.002085 | 0.000693 | -67% |
| Val Loss | 0.003514 | 0.001367 | -61% |
| MAPE | 6.74% | 3.83% | -43% |
| Status | Bom | **Excelente** | ↑↑↑ |

**Diagnóstico:**
- ✅ **Train Loss caindo:** Modelo está aprendendo
- ✅ **Val Loss caindo:** Modelo está generalizando
- ✅ **Val Loss > Train Loss:** Normal, indica que não há underfitting
- ✅ **Gap estável:** Não há sinais graves de overfitting
- ✅ **MAPE < 5%:** Meta atingida!

---

## 📊 Gráfico de Treinamento

O arquivo `models/training_history.png` mostra:

```
┌──────────────────────────────────────────────────────────┐
│  Histórico de Treinamento                                │
│                                                          │
│  Loss │                                                  │
│  0.01 │╲                                                 │
│       │ ╲                                                │
│       │  ╲   Val Loss (vermelho)                         │
│       │   ╲                                              │
│       │    ╲                                             │
│  0.005│     ╲───────────────────────────                 │
│       │      ╲                                           │
│       │       ╲  Train Loss (azul)                       │
│       │        ╲                                         │
│  0.001│         ╲────────────────────────                │
│       │                                                  │
│       └────────────────────────────────────────► Época   │
│        0    20    40    60    80    100                  │
└──────────────────────────────────────────────────────────┘
```

---

## 💾 Artefatos Salvos

### `models/model_lstm.pth` (Modelo Otimizado v2)

```python
torch.save({
    'model_state_dict': model.state_dict(),    # Pesos treinados
    'model_config': {
        'input_size': 1,
        'hidden_size': 100,    # OTIMIZADO: era 50, agora 100
        'num_layers': 2,
        'dropout': 0.2
    },
    'train_losses': train_losses,              # Histórico
    'val_losses': val_losses,
    'final_train_loss': 0.000693,
    'final_val_loss': 0.001367
}, 'models/model_lstm.pth')
```

### `src/hyperparameter_tuning.py` (Script de Otimização)

Script que executa experimentos sistemáticos variando:
- Learning rate: [0.0005, 0.001, 0.0001]
- Epochs: [100, 150, 200]
- Hidden size: [50, 64, 100]
- Dropout: [0.1, 0.2, 0.3]

**Descoberta principal:** Dobrar o hidden_size (50→100) foi a mudança mais impactante, reduzindo o MAPE de 6.74% para 3.83%.

---

## 🧪 Como Carregar o Modelo

```python
# Carregar checkpoint
checkpoint = torch.load('models/model_lstm.pth')

# Recriar modelo com mesma arquitetura
model = StockLSTM(**checkpoint['model_config'])

# Carregar pesos
model.load_state_dict(checkpoint['model_state_dict'])

# Modo de inferência
model.eval()
```

---

## ✅ Checklist de Conclusão

### Treinamento Base
- [x] MSELoss configurado
- [x] Adam optimizer com lr=0.001
- [x] Loop de treinamento (100 épocas)
- [x] Forward e Backward pass funcionando
- [x] Train e Val loss monitorados
- [x] Best model salvo
- [x] Gráfico de histórico gerado
- [x] Model checkpoint salvo em .pth

### Otimização de Hiperparâmetros
- [x] Script de tuning criado (`src/hyperparameter_tuning.py`)
- [x] 12 experimentos executados
- [x] Parâmetro mais impactante identificado (hidden_size)
- [x] Modelo re-treinado com hidden_size=100
- [x] MAPE reduzido de 6.74% para 3.83%
- [x] Meta de MAPE < 5% atingida

---

## 🔗 Próxima Etapa

**→ ETAPA 6: Avaliação** (Concluída)
- Calcular métricas: MAE, RMSE, MAPE
- Plotar previsões vs valores reais
- Reverter normalização para R$
