# Resumo: Separação de Responsabilidades

---

## 1. O que VOCÊ (desenvolvedor) decide:

### Hiperparâmetros:
| Decisão | Valor | Por quê? | Onde no código |
|---------|-------|----------|----------------|
| `hidden_size=50` | 50 neurônios por camada | Balanceia capacidade de aprendizado vs. overfitting. Mais neurônios = mais padrões, mas mais risco de decorar dados | `src/model.py`, linha 49 |
| `num_layers=2` | 2 camadas LSTM empilhadas | Permite capturar padrões em diferentes níveis de abstração (curto e médio prazo) | `src/model.py`, linha 51 |
| `epochs=100` | 100 iterações de treino | Suficiente para convergir sem overtreinar. Mais epochs = mais tempo, risco de overfitting | `src/train.py`, linha 24 |
| `seq_length=60` | Janela de 60 dias | ~3 meses de histórico. Captura tendências de médio prazo sem ser muito longo | `src/preprocessing.py`, linha 23 |

### Componentes:
| Decisão | Escolha | Por quê? | Onde no código |
|---------|---------|----------|----------------|
| Otimizador | Adam | Converge rápido, ajusta learning rate automaticamente. Bom para a maioria dos casos | `src/train.py`, linha 75 |
| Loss function | MSELoss | Penaliza erros grandes quadraticamente. Bom para regressão de preços | `src/train.py`, linha 72 |
| Normalização | MinMaxScaler (0-1) | LSTM funciona melhor com valores pequenos. Mantém proporções dos dados | `src/preprocessing.py`, linha 45 |

### Dados:
| Decisão | Escolha | Por quê? | Onde no código |
|---------|---------|----------|----------------|
| Ação | PETR4.SA | Escolha do Tech Challenge. Ação líquida com histórico disponível | `src/data_collection.py`, linha 17 |
| Período | 2018-2024 | ~6 anos de dados. Suficiente para treinar sem pegar dados muito antigos | `src/data_collection.py`, linhas 18-19 |
| Feature | Só 'Close' | Simplifica o modelo. Preço de fechamento é o mais relevante para previsão | `src/preprocessing.py`, linha 195 |

### Arquitetura:
| Decisão | Escolha | Por quê? | Onde no código |
|---------|---------|----------|----------------|
| Dropout | 0.2 (20%) | Regularização para evitar overfitting. 20% é valor comum e conservador | `src/model.py`, linha 52 |
| Camada final | Linear(50→1) | Transforma 50 neurônios em 1 previsão (próximo preço) | `src/model.py`, linha 63 |
| Train/Test split | 80%/20% | Padrão da indústria. Suficiente dados para treino e validação | `src/preprocessing.py`, linha 26 |

---

## 2. O que o PyTorch faz por você:

| O que faz | Seu código | O que PyTorch faz internamente |
|-----------|------------|-------------------------------|
| Implementa os portões | `nn.LSTM()` | Cria todos os pesos (W, b), implementa sigmoid/tanh, faz as operações |
| Backpropagation | `loss.backward()` | Calcula derivadas parciais, aplica regra da cadeia, propaga gradientes |
| Atualiza pesos | `optimizer.step()` | Aplica fórmulas do Adam (médias móveis, correção de bias, atualização) |
| Operações de tensor | `torch.FloatTensor()` | Gerencia memória, paraleliza, otimiza multiplicações de matriz |

---

## 3. O que é DEFINIÇÃO do LSTM (não pode mudar):

### Ter e USAR os 3 portões (forget, input, output):

**Pergunta: É obrigatório usar os três?**

**Sim, é obrigatório.** Os três portões trabalham JUNTOS em cada passo temporal. Não dá para "desligar" um deles porque:

```
┌─────────────────────────────────────────────────────────────────┐
│  A CADA PASSO TEMPORAL, OS 3 PORTÕES EXECUTAM EM SEQUÊNCIA:    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. FORGET GATE → Decide o que APAGAR da memória anterior      │
│     f_t = sigmoid(W_f × [h_{t-1}, x_t])                        │
│     Se não tiver: memória acumula infinitamente (estoura)      │
│                                                                 │
│  2. INPUT GATE → Decide o que ADICIONAR de novo                │
│     i_t = sigmoid(W_i × [h_{t-1}, x_t])                        │
│     Se não tiver: não aprende informações novas                │
│                                                                 │
│  3. OUTPUT GATE → Decide o que MOSTRAR como saída              │
│     o_t = sigmoid(W_o × [h_{t-1}, x_t])                        │
│     Se não tiver: não produz resposta útil                     │
│                                                                 │
│  CONCLUSÃO: Tirar qualquer um quebra o modelo.                 │
│  Por isso não é "opção" - é DEFINIÇÃO do LSTM.                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Onde está no código:**
- Você NÃO vê os portões explicitamente no seu código
- Eles estão DENTRO do `nn.LSTM` do PyTorch
- Quando você escreve `self.lstm = nn.LSTM(...)` em `src/model.py` linha 54-60, o PyTorch cria os 3 portões automaticamente
- Na execução (`lstm_out, (h_n, c_n) = self.lstm(x)` - linha 70), os 3 portões processam cada um dos 60 dias da sequência

---

### Ter cell state e hidden state:

**O que são e por que são obrigatórios:**

```
┌─────────────────────────────────────────────────────────────────┐
│  CELL STATE (c_t) - Memória de longo prazo                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  • É o "HD" do LSTM - armazena informações importantes         │
│  • Fórmula: c_t = f_t × c_{t-1} + i_t × g_t                   │
│  • A SOMA (não multiplicação) é o que resolve vanishing grad.  │
│  • Se não tiver: LSTM vira RNN comum (perde memória longa)    │
│                                                                 │
│  Onde no código:                                                │
│  • Retornado como c_n em: lstm_out, (h_n, c_n) = self.lstm(x) │
│  • src/model.py, linha 70                                      │
│  • Você não usa c_n diretamente, mas ele EXISTE internamente  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  HIDDEN STATE (h_t) - Saída de curto prazo                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  • É a "resposta" do LSTM em cada passo                        │
│  • Fórmula: h_t = o_t × tanh(c_t)                             │
│  • Passa para o próximo passo E é a saída da camada           │
│  • Se não tiver: não tem saída nem conexão temporal           │
│                                                                 │
│  Onde no código:                                                │
│  • lstm_out contém h_t de TODOS os 60 passos                  │
│  • Você usa o ÚLTIMO: last_output = lstm_out[:, -1, :]        │
│  • src/model.py, linha 71                                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

### As fórmulas matemáticas específicas:

**São fixas e definem o LSTM:**

```
┌─────────────────────────────────────────────────────────────────┐
│  FÓRMULAS DO LSTM (não pode mudar)                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Forget gate:  f_t = σ(W_f × [h_{t-1}, x_t] + b_f)            │
│  Input gate:   i_t = σ(W_i × [h_{t-1}, x_t] + b_i)            │
│  Cell cand.:   g_t = tanh(W_g × [h_{t-1}, x_t] + b_g)         │
│  Output gate:  o_t = σ(W_o × [h_{t-1}, x_t] + b_o)            │
│  Cell state:   c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t               │
│  Hidden state: h_t = o_t ⊙ tanh(c_t)                          │
│                                                                 │
│  σ = sigmoid (sempre entre 0 e 1 - funciona como "quanto")    │
│  tanh = tangente hiperbólica (entre -1 e 1 - valor candidato) │
│  ⊙ = multiplicação elemento a elemento                         │
│                                                                 │
│  POR QUE NÃO PODE MUDAR:                                       │
│  • Se trocar sigmoid por ReLU → portões não funcionam (0-1)   │
│  • Se trocar + por × no cell state → vanishing gradient volta │
│  • Se mudar ordem → modelo não converge                        │
│                                                                 │
│  Onde no código:                                                │
│  • DENTRO do nn.LSTM do PyTorch (você não vê)                 │
│  • Implementado em C++/CUDA para performance                   │
│  • Quando você usa nn.LSTM(), essas fórmulas são aplicadas    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

### Processar sequencialmente:

**LSTM processa um elemento por vez, em ordem:**

```
┌─────────────────────────────────────────────────────────────────┐
│  PROCESSAMENTO SEQUENCIAL (obrigatório para RNNs)              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Dia 1 → Dia 2 → Dia 3 → ... → Dia 60                         │
│    │       │       │             │                              │
│    ▼       ▼       ▼             ▼                              │
│  LSTM → LSTM → LSTM → ... → LSTM → Previsão                   │
│    │     ↑ │     ↑ │           ↑ │                              │
│    └─────┘ └─────┘ └───────────┘                               │
│       h,c passam de um para outro                              │
│                                                                 │
│  POR QUE É OBRIGATÓRIO:                                        │
│  • h_t depende de h_{t-1} (hidden anterior)                   │
│  • c_t depende de c_{t-1} (cell anterior)                     │
│  • Não dá para calcular dia 30 sem calcular dias 1-29 antes   │
│                                                                 │
│  Onde no código:                                                │
│  • Quando você passa x com shape (batch, 60, 1) para o LSTM   │
│  • O PyTorch processa os 60 passos internamente em sequência  │
│  • src/model.py, linha 70: self.lstm(x)                       │
│  • O loop sequencial está DENTRO do nn.LSTM                   │
│                                                                 │
│  SE QUISESSE PARALELO:                                         │
│  • Usaria Transformer (attention) em vez de LSTM              │
│  • Mas aí não seria mais LSTM                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Resumo Visual:

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                                    │
│  VOCÊ CONTROLA (pode mudar):                                                                      │
│  ══════════════════════════                                                                        │
│                                                                                                    │
│  O QUÊ                          │ ONDE              │ POR QUÊ                                     │
│  ───────────────────────────────┼───────────────────┼──────────────────────────────────────────── │
│  QUANTOS neurônios (50)         │ model.py L49      │ Balanceia aprendizado vs overfitting       │
│  QUANTAS camadas (2)            │ model.py L51      │ Captura padrões curto e médio prazo        │
│  QUANTO treina (100 epochs)     │ train.py L24      │ Suficiente p/ convergir sem overtreinar    │
│  QUANTOS dias olha (60)         │ preprocessing L23 │ ~3 meses captura tendências médio prazo    │
│  QUAL otimizador (Adam)         │ train.py L75      │ Converge rápido, ajusta LR automaticamente │
│  QUAL loss (MSE)                │ train.py L72      │ Penaliza erros grandes, bom p/ regressão   │
│  QUAL normalização (MinMax)     │ preprocessing L45 │ LSTM funciona melhor com valores 0-1       │
│  QUANTO dropout (0.2)           │ model.py L52      │ 20% regularização p/ evitar overfitting    │
│                                                                                                    │
│  ──────────────────────────────────────────────────────────────────────────────────────────────── │
│                                                                                                    │
│  PyTorch IMPLEMENTA (você usa):                                                                   │
│  ═══════════════════════════════                                                                   │
│  • COMO os portões funcionam              → dentro nn.LSTM                                        │
│  • COMO backprop calcula                  → loss.backward()                                       │
│  • COMO pesos atualizam                   → optimizer.step()                                      │
│                                                                                                    │
│  ──────────────────────────────────────────────────────────────────────────────────────────────── │
│                                                                                                    │
│  LSTM É DEFINIDO POR (não pode mudar):                                                           │
│  ═════════════════════════════════════                                                             │
│  • TER E USAR 3 portões                   → É o que é LSTM                                       │
│  • TER cell state + hidden state          → É o que é LSTM                                       │
│  • USAR fórmulas específicas              → É o que é LSTM                                       │
│  • PROCESSAR sequencialmente              → É o que é RNN                                        │
│                                                                                                    │
└────────────────────────────────────────────────────────────────────────────────────────────────────┘
```
