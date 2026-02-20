# O Que Você Controla vs O Que o PyTorch Faz vs O Que é o LSTM

Este documento separa claramente **quem decide o quê** no projeto:
- O que **VOCÊ** (desenvolvedor) escolheu e pode mudar
- O que o **PyTorch** implementa automaticamente (caixa preta)
- O que é a **DEFINIÇÃO do LSTM** (não pode mudar, senão não é LSTM)

---

## Visão Geral: As Três Camadas de Responsabilidade

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    QUEM DECIDE O QUÊ?                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  CAMADA 1: VOCÊ (Desenvolvedor)                                     │   │
│  │  ══════════════════════════════                                     │   │
│  │                                                                     │   │
│  │  • Decisões de NEGÓCIO e ARQUITETURA                               │   │
│  │  • Hiperparâmetros                                                  │   │
│  │  • Preparação de dados                                              │   │
│  │  • VOCÊ PODE MUDAR TUDO AQUI                                       │   │
│  │                                                                     │   │
│  │  Exemplo: "Vou usar 60 dias de janela" ← SUA ESCOLHA               │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  CAMADA 2: PyTorch (Biblioteca)                                     │   │
│  │  ══════════════════════════════                                     │   │
│  │                                                                     │   │
│  │  • Implementa os algoritmos                                         │   │
│  │  • Você USA, mas não precisa IMPLEMENTAR                           │   │
│  │  • É a "caixa preta" que faz o trabalho pesado                     │   │
│  │                                                                     │   │
│  │  Exemplo: "Backpropagation" ← PyTorch faz pra você                 │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  CAMADA 3: Definição do LSTM (Teoria/Matemática)                   │   │
│  │  ═══════════════════════════════════════════════                    │   │
│  │                                                                     │   │
│  │  • É o que faz LSTM ser LSTM                                       │   │
│  │  • Não é "decisão" - é a natureza do modelo                        │   │
│  │  • Se mudar isso, não é mais LSTM                                  │   │
│  │                                                                     │   │
│  │  Exemplo: "Ter 3 portões" ← DEFINIÇÃO, não escolha                 │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. O Que VOCÊ Decide e Controla

Estas são **suas escolhas**. Você pode mudar qualquer uma delas e o projeto ainda funciona (talvez melhor, talvez pior).

### 1.1 Decisões de Negócio (baseadas no Tech Challenge)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                 DECISÕES DE NEGÓCIO - VOCÊ ESCOLHEU                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  DECISÃO                     │ POR QUÊ?                │ PODERIA SER?       │
│  ════════════════════════════════════════════════════════════════════════  │
│                                                                             │
│  Usar modelo LSTM            │ Tech Challenge pediu    │ GRU, Transformer,  │
│                              │ especificamente LSTM    │ ARIMA              │
│                                                                             │
│  Prever preço de FECHAMENTO  │ Tech Challenge pediu    │ Preço de abertura, │
│                              │                         │ máximo, volume     │
│                                                                             │
│  Usar ação PETR4.SA          │ Você escolheu          │ Qualquer ação      │
│                              │                         │ (VALE3, ITUB4...)  │
│                                                                             │
│  Período 2018-2024           │ Você escolheu          │ Qualquer período   │
│                              │ (~6 anos de dados)     │                     │
│                                                                             │
│  API com FastAPI             │ Tech Challenge sugeriu │ Flask, Django      │
│                              │                         │                     │
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────────  │
│  ONDE NO CÓDIGO:                                                            │
│  • data_collection.py, linha 17: TICKER = "PETR4.SA"                       │
│  • data_collection.py, linha 18-19: START_DATE, END_DATE                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Hiperparâmetros do Modelo (VOCÊ escolhe!)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              HIPERPARÂMETROS - VOCÊ ESCOLHEU E PODE MUDAR                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ╔═══════════════════════════════════════════════════════════════════════╗ │
│  ║  HIPERPARÂMETRO      │ SEU VALOR │ ONDE NO CÓDIGO    │ ALTERNATIVAS  ║ │
│  ╠═══════════════════════════════════════════════════════════════════════╣ │
│  ║                      │           │                    │               ║ │
│  ║  hidden_size         │ 50        │ model.py, L49      │ 32, 64, 128  ║ │
│  ║  (neurônios/camada)  │           │                    │               ║ │
│  ║                      │           │                    │               ║ │
│  ║  num_layers          │ 2         │ model.py, L51      │ 1, 3, 4      ║ │
│  ║  (camadas LSTM)      │           │                    │               ║ │
│  ║                      │           │                    │               ║ │
│  ║  dropout             │ 0.2 (20%) │ model.py, L52      │ 0.1, 0.3, 0.5║ │
│  ║  (regularização)     │           │                    │               ║ │
│  ║                      │           │                    │               ║ │
│  ║  seq_length          │ 60 dias   │ preprocessing.py   │ 30, 90, 120  ║ │
│  ║  (janela temporal)   │           │ L23                │               ║ │
│  ║                      │           │                    │               ║ │
│  ║  epochs              │ 100       │ train.py, L24      │ 50, 200, 500 ║ │
│  ║  (iterações treino)  │           │                    │               ║ │
│  ║                      │           │                    │               ║ │
│  ║  learning_rate       │ 0.001     │ train.py, L25      │ 0.01, 0.0001 ║ │
│  ║  (velocidade apren.) │           │                    │               ║ │
│  ║                      │           │                    │               ║ │
│  ║  train_split         │ 80%       │ preprocessing.py   │ 70%, 90%     ║ │
│  ║  (% para treino)     │           │ L26                │               ║ │
│  ║                      │           │                    │               ║ │
│  ╚═══════════════════════════════════════════════════════════════════════╝ │
│                                                                             │
│  EXEMPLO DE MUDANÇA:                                                        │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  # Se você quiser experimentar com mais neurônios:                         │
│  # Abra model.py e mude:                                                   │
│                                                                             │
│  # ANTES (sua escolha atual)                                               │
│  hidden_size: int = 50                                                     │
│                                                                             │
│  # DEPOIS (nova escolha)                                                   │
│  hidden_size: int = 128                                                    │
│                                                                             │
│  # Retreine e veja se melhora!                                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.3 Decisões de Pré-processamento (VOCÊ escolhe!)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│           PRÉ-PROCESSAMENTO - VOCÊ ESCOLHEU E PODE MUDAR                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  DECISÃO                     │ SEU VALOR      │ ALTERNATIVAS               │
│  ════════════════════════════════════════════════════════════════════════  │
│                                                                             │
│  Tipo de normalização        │ MinMaxScaler   │ StandardScaler (z-score),  │
│                              │ (0 a 1)        │ RobustScaler, Log transform│
│                                                                             │
│  Feature usada               │ Só 'Close'     │ Close + Volume,            │
│                              │                │ OHLCV completo, indicadores│
│                                                                             │
│  Tratamento de missing       │ Não tem no     │ Interpolação, drop,        │
│                              │ yfinance       │ forward fill               │
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  ONDE NO CÓDIGO:                                                            │
│  • preprocessing.py, linha 45: MinMaxScaler(feature_range=(0, 1))          │
│  • preprocessing.py, linha 195: df['Close'].values                         │
│                                                                             │
│  SE QUISESSE MUDAR:                                                         │
│  ─────────────────                                                          │
│  # Para usar z-score em vez de MinMax:                                     │
│  from sklearn.preprocessing import StandardScaler                          │
│  scaler = StandardScaler()  # Em vez de MinMaxScaler                       │
│                                                                             │
│  # Para usar mais features:                                                 │
│  data = df[['Close', 'Volume']].values  # Em vez de só Close              │
│  # Mas aí input_size no model.py seria 2, não 1                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.4 Decisões de Arquitetura (VOCÊ escolhe!)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              ARQUITETURA - VOCÊ ESCOLHEU E PODE MUDAR                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  SUA ARQUITETURA ATUAL:                                                     │
│  ══════════════════════                                                     │
│                                                                             │
│  Input (1) → LSTM (50, 2 camadas) → Dropout (0.2) → Linear (50→1) → Output │
│                                                                             │
│  VOCÊ PODERIA TER FEITO:                                                   │
│  ═══════════════════════                                                    │
│                                                                             │
│  Opção A: Mais simples                                                      │
│  Input (1) → LSTM (32, 1 camada) → Linear (32→1) → Output                  │
│                                                                             │
│  Opção B: Mais complexa                                                     │
│  Input (1) → LSTM (128, 3 camadas) → Dropout (0.3) → Linear (128→64)      │
│           → ReLU → Linear (64→1) → Output                                  │
│                                                                             │
│  Opção C: Bidirecional                                                      │
│  Input (1) → BiLSTM (64, 2 camadas) → Dropout (0.2) → Linear (128→1)      │
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  ONDE NO CÓDIGO (model.py):                                                 │
│                                                                             │
│  class StockLSTM(nn.Module):                                                │
│      def __init__(self, ...):                                               │
│                                                                             │
│          # VOCÊ ESCOLHEU essa estrutura:                                   │
│          self.lstm = nn.LSTM(...)      ← Sua escolha de usar LSTM         │
│          self.dropout = nn.Dropout(...)← Sua escolha de usar dropout      │
│          self.linear = nn.Linear(...)  ← Sua escolha de camada final      │
│                                                                             │
│          # VOCÊ PODERIA adicionar:                                         │
│          self.linear2 = nn.Linear(64, 32)  # Mais camadas                 │
│          self.relu = nn.ReLU()             # Ativações extras             │
│          self.batch_norm = nn.BatchNorm1d(50)  # Normalização             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.5 Decisões de Treinamento (VOCÊ escolhe!)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              TREINAMENTO - VOCÊ ESCOLHEU E PODE MUDAR                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  DECISÃO              │ SEU VALOR    │ ONDE              │ ALTERNATIVAS    │
│  ═══════════════════════════════════════════════════════════════════════   │
│                                                                             │
│  Função de perda      │ MSELoss      │ train.py, L72     │ MAELoss, L1Loss│
│                       │              │                    │ HuberLoss      │
│                                                                             │
│  Otimizador           │ Adam         │ train.py, L75     │ SGD, RMSprop,  │
│                       │              │                    │ AdamW          │
│                                                                             │
│  Scheduler de LR      │ Nenhum       │ (não implementado)│ StepLR,        │
│                       │              │                    │ ReduceLROnPlat.│
│                                                                             │
│  Early stopping       │ Não          │ (não implementado)│ Sim, com       │
│                       │              │                    │ patience=10    │
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  ONDE NO CÓDIGO (train.py):                                                 │
│                                                                             │
│  # VOCÊ ESCOLHEU MSE:                                                      │
│  criterion = nn.MSELoss()                                                  │
│                                                                             │
│  # VOCÊ PODERIA usar MAE:                                                  │
│  criterion = nn.L1Loss()  # MAE - menos sensível a outliers               │
│                                                                             │
│  # VOCÊ ESCOLHEU Adam:                                                     │
│  optimizer = optim.Adam(model.parameters(), lr=0.001)                      │
│                                                                             │
│  # VOCÊ PODERIA usar SGD com momentum:                                     │
│  optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. O Que o PyTorch Faz por Você (Caixa Preta)

Estas são coisas que você **USA**, mas **NÃO IMPLEMENTA**. O PyTorch cuida disso.

### 2.1 Implementação dos Portões LSTM

```
┌─────────────────────────────────────────────────────────────────────────────┐
│            PORTÕES LSTM - PyTorch IMPLEMENTA PRA VOCÊ                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  O QUE VOCÊ ESCREVE:                                                       │
│  ════════════════════                                                       │
│                                                                             │
│  self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=2)          │
│                                                                             │
│  # Uma linha só!                                                           │
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  O QUE O PyTorch FAZ INTERNAMENTE (você não vê):                           │
│  ═══════════════════════════════════════════════                            │
│                                                                             │
│  class LSTM_Interno:  # PSEUDOCÓDIGO - você NÃO implementa isso           │
│      def __init__(self, input_size, hidden_size):                          │
│          # Cria pesos para TODOS os 4 portões automaticamente             │
│          self.W_ii = Parameter(...)  # Pesos input gate                   │
│          self.W_if = Parameter(...)  # Pesos forget gate                  │
│          self.W_ig = Parameter(...)  # Pesos cell gate                    │
│          self.W_io = Parameter(...)  # Pesos output gate                  │
│          self.W_hi = Parameter(...)  # Pesos hidden→input                 │
│          self.W_hf = Parameter(...)  # Pesos hidden→forget                │
│          self.W_hg = Parameter(...)  # Pesos hidden→cell                  │
│          self.W_ho = Parameter(...)  # Pesos hidden→output                │
│          # + todos os biases...                                            │
│                                                                             │
│      def forward(self, x, h_prev, c_prev):                                 │
│          # Implementa as fórmulas dos portões                              │
│          f = sigmoid(self.W_if @ x + self.W_hf @ h_prev + self.b_f)       │
│          i = sigmoid(self.W_ii @ x + self.W_hi @ h_prev + self.b_i)       │
│          g = tanh(self.W_ig @ x + self.W_hg @ h_prev + self.b_g)          │
│          o = sigmoid(self.W_io @ x + self.W_ho @ h_prev + self.b_o)       │
│                                                                             │
│          c = f * c_prev + i * g  # Nova célula de memória                 │
│          h = o * tanh(c)         # Nova saída                             │
│                                                                             │
│          return h, c                                                        │
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  VOCÊ TEM CONTROLE SOBRE OS PORTÕES?                                       │
│  ════════════════════════════════════                                       │
│                                                                             │
│  ❌ NÃO pode mudar a FÓRMULA dos portões                                  │
│     (sigmoid, tanh, como combinam - é definição do LSTM)                   │
│                                                                             │
│  ✅ SIM pode mudar o TAMANHO dos portões                                  │
│     (hidden_size=50 → cada portão tem 50 neurônios)                       │
│                                                                             │
│  ✅ SIM pode mudar QUANTAS CAMADAS                                        │
│     (num_layers=2 → 2 conjuntos de portões empilhados)                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Backpropagation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│            BACKPROPAGATION - PyTorch IMPLEMENTA PRA VOCÊ                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  O QUE VOCÊ ESCREVE:                                                       │
│  ════════════════════                                                       │
│                                                                             │
│  loss.backward()  # Uma linha só!                                          │
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  O QUE O PyTorch FAZ INTERNAMENTE:                                         │
│  ═════════════════════════════════                                          │
│                                                                             │
│  1. Percorre o grafo computacional de trás pra frente                      │
│  2. Calcula a derivada parcial de cada operação                            │
│  3. Aplica a regra da cadeia para propagar gradientes                      │
│  4. Armazena gradiente em cada parâmetro (.grad)                           │
│                                                                             │
│  # PSEUDOCÓDIGO do que acontece (você NÃO implementa):                     │
│  for param in reversed(all_parameters):                                    │
│      param.grad = calcular_gradiente(loss, param)                          │
│      # Envolve derivadas parciais, regra da cadeia, etc.                  │
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  PARA LSTM É AINDA MAIS COMPLEXO:                                          │
│  ═════════════════════════════════                                          │
│                                                                             │
│  Backpropagation Through Time (BPTT):                                       │
│  • Desdobra a rede em 60 cópias (uma por dia)                             │
│  • Calcula gradientes para cada passo temporal                             │
│  • Acumula gradientes ao longo do tempo                                    │
│                                                                             │
│  VOCÊ TERIA QUE IMPLEMENTAR ~200 linhas de código.                        │
│  PyTorch faz em 1 linha: loss.backward()                                   │
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  VOCÊ TEM CONTROLE?                                                         │
│  ══════════════════                                                         │
│                                                                             │
│  ❌ NÃO pode mudar COMO backpropagation funciona                          │
│     (é matemática - derivadas parciais + regra da cadeia)                  │
│                                                                             │
│  ✅ SIM pode escolher SE fazer backpropagation                            │
│     (with torch.no_grad(): desabilita)                                     │
│                                                                             │
│  ✅ SIM pode controlar QUANDO                                             │
│     (só chama loss.backward() quando quiser)                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.3 Atualização de Pesos (Otimização)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│            OTIMIZAÇÃO - PyTorch IMPLEMENTA PRA VOCÊ                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  O QUE VOCÊ ESCREVE:                                                       │
│  ════════════════════                                                       │
│                                                                             │
│  optimizer = optim.Adam(model.parameters(), lr=0.001)  # Escolhe Adam     │
│  optimizer.step()  # Atualiza pesos                                        │
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  O QUE O PyTorch FAZ INTERNAMENTE (Adam):                                  │
│  ═════════════════════════════════════════                                  │
│                                                                             │
│  # PSEUDOCÓDIGO - você NÃO implementa:                                     │
│  class Adam:                                                                │
│      def step(self):                                                        │
│          for param in parameters:                                           │
│              # Atualiza médias móveis do gradiente                         │
│              m = beta1 * m + (1-beta1) * param.grad                        │
│              v = beta2 * v + (1-beta2) * param.grad²                       │
│                                                                             │
│              # Correção de bias                                             │
│              m_hat = m / (1 - beta1^t)                                     │
│              v_hat = v / (1 - beta2^t)                                     │
│                                                                             │
│              # Atualiza parâmetro                                          │
│              param = param - lr * m_hat / (sqrt(v_hat) + eps)             │
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  VOCÊ TEM CONTROLE?                                                         │
│  ══════════════════                                                         │
│                                                                             │
│  ✅ SIM escolhe QUAL otimizador (Adam, SGD, RMSprop...)                   │
│  ✅ SIM escolhe learning rate                                              │
│  ✅ SIM escolhe outros parâmetros (beta1, beta2, weight_decay...)         │
│  ❌ NÃO pode mudar as fórmulas internas do Adam                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.4 Operações com Tensores

```
┌─────────────────────────────────────────────────────────────────────────────┐
│          OPERAÇÕES COM TENSORES - PyTorch IMPLEMENTA PRA VOCÊ              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  O QUE VOCÊ ESCREVE:                                                       │
│  ════════════════════                                                       │
│                                                                             │
│  x = torch.FloatTensor(data)                                               │
│  y = model(x)  # Multiplicações de matrizes, ativações, etc.              │
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  O QUE O PyTorch FAZ INTERNAMENTE:                                         │
│  ═════════════════════════════════                                          │
│                                                                             │
│  • Aloca memória na GPU/CPU                                                │
│  • Faz multiplicação de matrizes otimizada (BLAS, cuBLAS)                 │
│  • Paraleliza operações                                                    │
│  • Gerencia memória (garbage collection)                                   │
│  • Constrói grafo computacional para autograd                             │
│                                                                             │
│  VOCÊ TEM CONTROLE?                                                         │
│  ══════════════════                                                         │
│                                                                             │
│  ✅ SIM escolhe device (CPU ou GPU): tensor.to('cuda')                    │
│  ✅ SIM escolhe dtype (float32, float64): torch.FloatTensor               │
│  ❌ NÃO implementa as operações de baixo nível                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. O Que é DEFINIÇÃO do LSTM (Não Pode Mudar)

Estas são características que **definem** o que é um LSTM. Se você mudar, não é mais LSTM.

### 3.1 Os Três Portões

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              PORTÕES - DEFINIÇÃO DO LSTM (NÃO PODE MUDAR)                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  LSTM TEM POR DEFINIÇÃO:                                                   │
│  ═══════════════════════                                                    │
│                                                                             │
│  1. FORGET GATE: Decide o que esquecer                                     │
│     f_t = sigmoid(W_f × [h_{t-1}, x_t] + b_f)                             │
│                                                                             │
│  2. INPUT GATE: Decide o que adicionar                                     │
│     i_t = sigmoid(W_i × [h_{t-1}, x_t] + b_i)                             │
│                                                                             │
│  3. OUTPUT GATE: Decide o que mostrar                                      │
│     o_t = sigmoid(W_o × [h_{t-1}, x_t] + b_o)                             │
│                                                                             │
│  ISSO NÃO É ESCOLHA SUA. É O QUE FAZ LSTM SER LSTM.                       │
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  SE VOCÊ MUDAR:                                                             │
│  ═══════════════                                                            │
│                                                                             │
│  ❌ Remover forget gate     → Não é mais LSTM (seria outra coisa)         │
│  ❌ Mudar sigmoid para ReLU → Não é mais LSTM padrão                      │
│  ❌ Mudar fórmula do gate   → Não é mais LSTM                             │
│                                                                             │
│  COMPARAÇÃO:                                                                │
│  ═══════════                                                                │
│                                                                             │
│  LSTM: 3 portões (forget, input, output) + cell state                     │
│  GRU:  2 portões (reset, update) → É DIFERENTE, não é LSTM                │
│  RNN:  0 portões → É DIFERENTE, não é LSTM                                │
│                                                                             │
│  É como perguntar: "Posso fazer um carro sem rodas?"                       │
│  Pode, mas não é mais um carro.                                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Cell State e Hidden State

```
┌─────────────────────────────────────────────────────────────────────────────┐
│           ESTADOS - DEFINIÇÃO DO LSTM (NÃO PODE MUDAR)                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  LSTM TEM POR DEFINIÇÃO:                                                   │
│  ═══════════════════════                                                    │
│                                                                             │
│  1. CELL STATE (c_t): Memória de longo prazo                              │
│     c_t = f_t × c_{t-1} + i_t × tanh(W_c × [h_{t-1}, x_t] + b_c)         │
│                                                                             │
│  2. HIDDEN STATE (h_t): Saída de curto prazo                              │
│     h_t = o_t × tanh(c_t)                                                  │
│                                                                             │
│  ISSO É A ESSÊNCIA DO LSTM. É O QUE RESOLVE O VANISHING GRADIENT.         │
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  POR QUE NÃO PODE MUDAR?                                                   │
│  ═══════════════════════                                                    │
│                                                                             │
│  A fórmula do cell state:                                                   │
│  c_t = f_t × c_{t-1} + i_t × c̃_t                                         │
│        ↑                                                                    │
│        └── SOMA! (não multiplicação)                                       │
│                                                                             │
│  Essa SOMA é o que permite gradientes fluírem sem desaparecer.            │
│  Se você mudar para multiplicação, perde a vantagem do LSTM.              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Processamento Sequencial

```
┌─────────────────────────────────────────────────────────────────────────────┐
│      PROCESSAMENTO SEQUENCIAL - DEFINIÇÃO DE RNNs (NÃO PODE MUDAR)         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  LSTM (como toda RNN) PROCESSA SEQUENCIALMENTE:                            │
│  ═══════════════════════════════════════════════                            │
│                                                                             │
│  x₁ → x₂ → x₃ → ... → x₆₀                                                 │
│   │     │     │           │                                                 │
│   ▼     ▼     ▼           ▼                                                 │
│  LSTM → LSTM → LSTM → ... → LSTM → Saída                                  │
│   │     ↑     ↑           ↑                                                 │
│   └─────┴─────┴───────────┘                                                │
│         h e c passam de um para o outro                                    │
│                                                                             │
│  ISSO É DEFINIÇÃO. NÃO É ESCOLHA.                                         │
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  SE VOCÊ NÃO QUISER PROCESSAMENTO SEQUENCIAL:                             │
│  ═════════════════════════════════════════════                              │
│                                                                             │
│  Use Transformer! Ele processa tudo em paralelo.                           │
│  Mas aí não é mais LSTM.                                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.4 Aprendizado por Gradiente

```
┌─────────────────────────────────────────────────────────────────────────────┐
│       APRENDIZADO POR GRADIENTE - DEFINIÇÃO DE REDES NEURAIS               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  TODA REDE NEURAL (incluindo LSTM) APRENDE ASSIM:                          │
│  ═════════════════════════════════════════════════                          │
│                                                                             │
│  1. Forward pass: Calcula previsão                                         │
│  2. Calcula erro (loss)                                                    │
│  3. Backward pass: Calcula gradientes                                      │
│  4. Atualiza pesos: peso_novo = peso_antigo - lr × gradiente              │
│                                                                             │
│  ISSO NÃO É DECISÃO. É COMO REDES NEURAIS FUNCIONAM.                      │
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  VOCÊ ESCOLHE:                        │  VOCÊ NÃO ESCOLHE:                 │
│  ═════════════                        │  ══════════════════                 │
│  • Qual loss usar (MSE, MAE)         │  • Que usa gradientes              │
│  • Qual otimizador (Adam, SGD)       │  • Que faz backpropagation         │
│  • Learning rate (0.001)             │  • Que atualiza pesos              │
│  • Quantas épocas (100)              │  • O processo iterativo            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Tabela Resumo Completa

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TABELA RESUMO: QUEM DECIDE O QUÊ?                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ╔════════════════════════════════════════════════════════════════════════╗│
│  ║ ASPECTO               │ QUEM DECIDE    │ PODE MUDAR? │ ONDE NO CÓDIGO ║│
│  ╠════════════════════════════════════════════════════════════════════════╣│
│  ║                                                                        ║│
│  ║ --- VOCÊ DECIDE (HIPERPARÂMETROS) ---                                 ║│
│  ║                                                                        ║│
│  ║ Qual ação usar        │ VOCÊ           │ ✅ Sim      │ data_collection║│
│  ║ Período dos dados     │ VOCÊ           │ ✅ Sim      │ data_collection║│
│  ║ Janela temporal (60)  │ VOCÊ           │ ✅ Sim      │ preprocessing  ║│
│  ║ hidden_size (50)      │ VOCÊ           │ ✅ Sim      │ model.py       ║│
│  ║ num_layers (2)        │ VOCÊ           │ ✅ Sim      │ model.py       ║│
│  ║ dropout (0.2)         │ VOCÊ           │ ✅ Sim      │ model.py       ║│
│  ║ epochs (100)          │ VOCÊ           │ ✅ Sim      │ train.py       ║│
│  ║ learning_rate (0.001) │ VOCÊ           │ ✅ Sim      │ train.py       ║│
│  ║ train_split (80%)     │ VOCÊ           │ ✅ Sim      │ preprocessing  ║│
│  ║ Tipo de normalização  │ VOCÊ           │ ✅ Sim      │ preprocessing  ║│
│  ║ Função de loss (MSE)  │ VOCÊ           │ ✅ Sim      │ train.py       ║│
│  ║ Otimizador (Adam)     │ VOCÊ           │ ✅ Sim      │ train.py       ║│
│  ║ Features (só Close)   │ VOCÊ           │ ✅ Sim      │ preprocessing  ║│
│  ║                                                                        ║│
│  ║ --- PYTORCH FAZ (IMPLEMENTAÇÃO) ---                                   ║│
│  ║                                                                        ║│
│  ║ Código dos portões    │ PyTorch        │ ❌ Não      │ nn.LSTM interno║│
│  ║ Backpropagation       │ PyTorch        │ ❌ Não      │ loss.backward()║│
│  ║ Cálculo de gradientes │ PyTorch        │ ❌ Não      │ autograd       ║│
│  ║ Atualização de pesos  │ PyTorch        │ ❌ Não      │ optimizer.step ║│
│  ║ Operações de tensor   │ PyTorch        │ ❌ Não      │ torch.*        ║│
│  ║ Gerenciamento de GPU  │ PyTorch        │ ❌ Não      │ .to('cuda')    ║│
│  ║                                                                        ║│
│  ║ --- DEFINIÇÃO DO LSTM (TEORIA) ---                                    ║│
│  ║                                                                        ║│
│  ║ Ter 3 portões         │ DEFINIÇÃO      │ ❌ Não*     │ É o que é LSTM ║│
│  ║ Ter cell state        │ DEFINIÇÃO      │ ❌ Não*     │ É o que é LSTM ║│
│  ║ Ter hidden state      │ DEFINIÇÃO      │ ❌ Não*     │ É o que é LSTM ║│
│  ║ Fórmulas dos portões  │ DEFINIÇÃO      │ ❌ Não*     │ É o que é LSTM ║│
│  ║ Processar sequencial  │ DEFINIÇÃO      │ ❌ Não*     │ É o que é RNN  ║│
│  ║ Aprender por gradient │ DEFINIÇÃO      │ ❌ Não*     │ É rede neural  ║│
│  ║                                                                        ║│
│  ║ * Se mudar, não é mais LSTM                                           ║│
│  ║                                                                        ║│
│  ╚════════════════════════════════════════════════════════════════════════╝│
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Analogia Final

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ANALOGIA: DIRIGIR UM CARRO                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ╔═══════════════════════════════════════════════════════════════════════╗ │
│  ║  VOCÊ (Motorista)                                                     ║ │
│  ╠═══════════════════════════════════════════════════════════════════════╣ │
│  ║                                                                       ║ │
│  ║  • Escolhe para onde ir (destino = problema a resolver)              ║ │
│  ║  • Escolhe qual caminho (arquitetura = como resolver)                ║ │
│  ║  • Escolhe a velocidade (learning rate)                              ║ │
│  ║  • Escolhe quando parar (epochs)                                     ║ │
│  ║  • Escolhe o combustível (dados)                                     ║ │
│  ║                                                                       ║ │
│  ╚═══════════════════════════════════════════════════════════════════════╝ │
│                                                                             │
│  ╔═══════════════════════════════════════════════════════════════════════╗ │
│  ║  PyTorch (O Carro)                                                    ║ │
│  ╠═══════════════════════════════════════════════════════════════════════╣ │
│  ║                                                                       ║ │
│  ║  • Implementa o motor (backpropagation)                              ║ │
│  ║  • Implementa a transmissão (otimizadores)                           ║ │
│  ║  • Implementa os sistemas (operações de tensor)                      ║ │
│  ║  • Você USA, mas não precisa CONSTRUIR o motor                       ║ │
│  ║                                                                       ║ │
│  ╚═══════════════════════════════════════════════════════════════════════╝ │
│                                                                             │
│  ╔═══════════════════════════════════════════════════════════════════════╗ │
│  ║  Física/Engenharia (Leis da Natureza)                                ║ │
│  ╠═══════════════════════════════════════════════════════════════════════╣ │
│  ║                                                                       ║ │
│  ║  • Carro precisa de rodas para andar (LSTM precisa de portões)       ║ │
│  ║  • Motor funciona por combustão (rede neural aprende por gradiente)  ║ │
│  ║  • Você não "escolhe" isso - é como as coisas funcionam              ║ │
│  ║                                                                       ║ │
│  ╚═══════════════════════════════════════════════════════════════════════╝ │
│                                                                             │
│  RESUMO:                                                                    │
│  • VOCÊ dirige (configura, escolhe, decide)                               │
│  • PYTORCH é o carro (implementa, executa)                                │
│  • FÍSICA/MATEMÁTICA são as leis (definem como funciona)                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Na Prática: O Que Você Pode Experimentar

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              EXPERIMENTOS QUE VOCÊ PODE FAZER                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  MUDANÇAS FÁCEIS (só alterar números):                                     │
│  ═════════════════════════════════════                                      │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ # model.py - testar diferentes tamanhos                             │   │
│  │ hidden_size: int = 100  # era 50                                   │   │
│  │ num_layers: int = 3     # era 2                                    │   │
│  │                                                                     │   │
│  │ # train.py - testar diferentes configurações                       │   │
│  │ EPOCHS = 200            # era 100                                  │   │
│  │ LEARNING_RATE = 0.0001  # era 0.001                               │   │
│  │                                                                     │   │
│  │ # preprocessing.py - testar diferentes janelas                     │   │
│  │ SEQ_LENGTH = 90         # era 60                                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  MUDANÇAS MÉDIAS (alterar componentes):                                    │
│  ══════════════════════════════════════                                     │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ # train.py - testar outro otimizador                               │   │
│  │ optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)   │   │
│  │                                                                     │   │
│  │ # train.py - testar outra loss                                     │   │
│  │ criterion = nn.L1Loss()  # MAE em vez de MSE                      │   │
│  │                                                                     │   │
│  │ # preprocessing.py - testar outra normalização                     │   │
│  │ from sklearn.preprocessing import StandardScaler                   │   │
│  │ scaler = StandardScaler()                                         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  MUDANÇAS AVANÇADAS (alterar arquitetura):                                 │
│  ══════════════════════════════════════════                                 │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ # model.py - usar GRU em vez de LSTM                               │   │
│  │ self.rnn = nn.GRU(...)  # em vez de nn.LSTM                       │   │
│  │                                                                     │   │
│  │ # model.py - usar LSTM bidirecional                                │   │
│  │ self.lstm = nn.LSTM(..., bidirectional=True)                      │   │
│  │                                                                     │   │
│  │ # model.py - adicionar mais camadas após LSTM                      │   │
│  │ self.fc1 = nn.Linear(50, 32)                                      │   │
│  │ self.fc2 = nn.Linear(32, 1)                                       │   │
│  │ self.relu = nn.ReLU()                                             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  O QUE VOCÊ NÃO PODE MUDAR:                                                │
│  ══════════════════════════                                                 │
│                                                                             │
│  ❌ Como os portões funcionam internamente (é nn.LSTM)                    │
│  ❌ Como backpropagation calcula gradientes (é loss.backward())           │
│  ❌ Como Adam atualiza pesos (é optimizer.step())                         │
│                                                                             │
│  Se precisar mudar isso, teria que implementar do zero (não recomendado). │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Resumo Final

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         RESUMO FINAL                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  VOCÊ É O ARQUITETO E CONFIGURADOR:                                        │
│  ═══════════════════════════════════                                        │
│  • Decide a estrutura (quantas camadas, quantos neurônios)                │
│  • Define os hiperparâmetros (learning rate, epochs, janela)              │
│  • Prepara os dados (normalização, features, split)                       │
│  • Escolhe os componentes (otimizador, loss function)                     │
│                                                                             │
│  PYTORCH É O CONSTRUTOR E EXECUTOR:                                        │
│  ═══════════════════════════════════                                        │
│  • Implementa os algoritmos (portões, backprop, otimização)              │
│  • Gerencia memória e GPU                                                  │
│  • Faz as operações matemáticas pesadas                                    │
│  • Você não precisa saber como funciona por dentro                         │
│                                                                             │
│  LSTM É A PLANTA/PROJETO:                                                  │
│  ═════════════════════════                                                  │
│  • Define que precisa ter 3 portões                                       │
│  • Define que precisa ter cell state e hidden state                       │
│  • Define as fórmulas matemáticas                                          │
│  • Se você mudar isso, não é mais LSTM                                    │
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  LINHA DO TEMPO DO PROJETO:                                                 │
│                                                                             │
│  1. VOCÊ leu o Tech Challenge e DECIDIU usar LSTM                         │
│  2. VOCÊ CONFIGUROU os hiperparâmetros (50 neurônios, 2 camadas, etc.)   │
│  3. VOCÊ PREPAROU os dados (normalização, janelas de 60 dias)             │
│  4. PyTorch IMPLEMENTOU os portões quando você escreveu nn.LSTM()         │
│  5. PyTorch TREINOU quando você chamou loss.backward() e optimizer.step() │
│  6. VOCÊ SALVOU os pesos em .pth                                          │
│  7. Na API, PyTorch USA os pesos para fazer previsões                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

*Documento criado para esclarecer a divisão de responsabilidades no projeto LSTM*
