# 📌 ETAPA 3: Pré-processamento

## 📋 Resumo
| Item | Valor |
|------|-------|
| **Status** | ✅ Concluída |
| **Data** | 2026-02-17 |
| **Tempo Estimado** | 45 min |
| **Tempo Real** | ~10 min |

---

## 🎯 Objetivo
Transformar os dados brutos em um formato adequado para a LSTM: normalização, criação de sequências temporais e conversão para tensores PyTorch.

---

## 🎓 Conexão com as Aulas

### Aula 02 - Teoria das Redes Neurais Profundas
**Arquivo:** `docs - fase 4 /etapa 1 - redes neurais e deep learning/Aula 02 - Teoria das Redes Neurais Profundas.txt`

**Conceito: Normalização**
> A normalização é essencial para evitar que valores grandes dominem o cálculo do erro durante o treinamento.

**Por que normalizar entre 0 e 1?**
- Redes neurais funcionam melhor com valores pequenos e uniformes
- Evita problemas de *exploding gradients*
- Acelera a convergência do treinamento

### Aula 03 - Arquiteturas de Redes Neurais Profundas
**Arquivo:** `docs - fase 4 /etapa 1 - redes neurais e deep learning/Aula 03 - Arquiteturas de Redes Neurais Profundas.txt`

**Conceito: Processamento de Sequências**
> *"As RNNs processam a entrada passo a passo, mantendo um estado interno que captura informações de entradas anteriores."* (Linha ~421)

> *"A capacidade de transição entre estados permite que a RNN capture dependências temporais ou sequenciais dos dados."* (Linha ~438)

**Por que janelas deslizantes?**
- RNNs/LSTMs processam **sequências** de dados
- Precisamos "fatiar" a série temporal em blocos
- Cada bloco = contexto histórico para prever o próximo valor

---

## 📁 Arquivo Implementado

### `src/preprocessing.py`

#### Estrutura do Código

```python
# Linhas 1-5: Cabeçalho
# ═══════════════════════════════════════════════════════════════
# 📌 ETAPA 3: Pré-processamento
# 🎯 Objetivo: Normalizar dados e criar janelas temporais
# ═══════════════════════════════════════════════════════════════
```

#### Configurações (Linhas 22-29)
```python
SEQ_LENGTH = 60      # 60 dias = ~3 meses de histórico
TRAIN_SPLIT = 0.8    # 80% treino, 20% teste
MODELS_DIR = Path(__file__).parent.parent / "models"
```

---

## 🔬 Funções Implementadas

### 1️⃣ `normalize_data()` (Linhas 32-52)

**Propósito:** Escalar os dados entre 0 e 1

```python
def normalize_data(data: np.ndarray) -> Tuple[np.ndarray, MinMaxScaler]:
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    return data_scaled, scaler
```

**Conexão com a Aula:**
| Conceito | Aula | Código |
|----------|------|--------|
| "Normalização evita valores grandes" | Aula 02 | `MinMaxScaler(feature_range=(0, 1))` |
| "Scaler deve ser salvo para inferência" | Prática ML | `return data_scaled, scaler` |

**Resultado:**
```
Original - Min: 3.24, Max: 27.38
Normalizado - Min: 0.0000, Max: 1.0000
```

---

### 2️⃣ `create_sequences()` (Linhas 55-94)

**Propósito:** Criar janelas deslizantes para a LSTM

```python
def create_sequences(data: np.ndarray, seq_length: int = SEQ_LENGTH):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])      # 60 dias de entrada
        y.append(data[i+seq_length])         # 1 dia de saída (próximo)
    return np.array(X), np.array(y)
```

**Conexão com a Aula:**
> *"A capacidade de lembrar informações anteriores permite que as RNNs considerem o contexto amplo."* (Aula 03, linha ~454)

**Visualização:**
```
Dados: [P1, P2, P3, P4, P5, P6, ..., P60, P61, P62, ...]

Sequência 1: [P1, P2, ..., P60]  → Prever P61
Sequência 2: [P2, P3, ..., P61]  → Prever P62
Sequência 3: [P3, P4, ..., P62]  → Prever P63
...
```

**Por que 60 dias?**
- ~3 meses de histórico = contexto temporal suficiente
- Captura tendências de curto/médio prazo
- Recomendação do guia do Tech Challenge

---

### 3️⃣ `train_test_split()` (Linhas 97-127)

**Propósito:** Dividir dados em treino e teste

```python
def train_test_split(X, y, train_ratio=0.8):
    split_idx = int(len(X) * train_ratio)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    return X_train, X_test, y_train, y_test
```

**Conexão com a Aula:**
> *"A regularização é uma técnica fundamental para evitar o sobreajuste, garantindo que o modelo generalize bem para novos dados."* (Aula 03, linha ~331)

**Por que separar?**
- **Treino (80%):** Dados para o modelo aprender
- **Teste (20%):** Dados que o modelo NUNCA viu → avalia generalização

**⚠️ Importante:** Em séries temporais, **NÃO** embaralhamos os dados! A ordem cronológica é preservada.

---

### 4️⃣ `to_tensors()` (Linhas 130-159)

**Propósito:** Converter NumPy arrays para tensores PyTorch

```python
def to_tensors(X_train, X_test, y_train, y_test):
    X_train_t = torch.FloatTensor(X_train)
    X_test_t = torch.FloatTensor(X_test)
    y_train_t = torch.FloatTensor(y_train)
    y_test_t = torch.FloatTensor(y_test)
    return X_train_t, X_test_t, y_train_t, y_test_t
```

**Conexão com a Aula:**
> *"PyTorch trabalha com tensores, que são estruturas otimizadas para operações matriciais em GPU/CPU."* (Conceito fundamental de Deep Learning)

**Por que FloatTensor?**
- Precisão de 32 bits (float32)
- Suficiente para treinamento
- Compatível com operações da GPU

---

### 5️⃣ `preprocess_data()` (Linhas 162-233) - Pipeline Completa

**Propósito:** Orquestrar todo o pré-processamento

```python
def preprocess_data(ticker, seq_length, train_ratio, save_scaler):
    # 1. Carregar dados
    df = load_stock_data(ticker)
    
    # 2. Selecionar coluna Close
    data = df['Close'].values.reshape(-1, 1)
    
    # 3. Normalizar
    data_scaled, scaler = normalize_data(data)
    
    # 4. Criar sequências
    X, y = create_sequences(data_scaled, seq_length)
    
    # 5. Dividir treino/teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_ratio)
    
    # 6. Converter para tensores
    X_train_t, X_test_t, y_train_t, y_test_t = to_tensors(...)
    
    # 7. Salvar scaler e config
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(config, 'models/config.pkl')
    
    return X_train_t, X_test_t, y_train_t, y_test_t, scaler
```

---

## 📊 Resultado do Pré-processamento

```
┌─────────────────────────────────────────────────┐
│        Dados Originais → Dados Processados      │
├─────────────────────────────────────────────────┤
│  Registros originais:   1487                    │
│  Após sequenciamento:   1427 amostras           │
│  (1487 - 60 = 1427)                             │
├─────────────────────────────────────────────────┤
│  TREINO (80%)         │  TESTE (20%)            │
│  X: (1141, 60, 1)     │  X: (286, 60, 1)        │
│  y: (1141, 1)         │  y: (286, 1)            │
├─────────────────────────────────────────────────┤
│  Shape explicado:                               │
│  (amostras, seq_length, features)               │
│  (1141, 60, 1) = 1141 sequências de 60 dias     │
│                  com 1 feature (Close)          │
└─────────────────────────────────────────────────┘
```

---

## 💾 Artefatos Salvos

| Arquivo | Descrição | Uso |
|---------|-----------|-----|
| `models/scaler.pkl` | MinMaxScaler treinado | Reverter normalização na inferência |
| `models/config.pkl` | Configurações (seq_length, ticker) | Carregar modelo corretamente |

---

## ✅ Checklist de Conclusão

- [x] Normalização implementada (0-1)
- [x] Janelas deslizantes de 60 dias
- [x] Split 80/20 sem embaralhamento
- [x] Conversão para tensores PyTorch
- [x] Scaler salvo para inferência
- [x] Config salvo para reprodutibilidade

---

## 🔗 Próxima Etapa

**→ ETAPA 4: Modelo LSTM**
- Criar classe `StockLSTM` com PyTorch
- Definir camadas: LSTM → Dropout → Linear
- Configurar hiperparâmetros
