# Entendendo o Modelo LSTM - Guia Completo

Este documento explica em detalhes o funcionamento do modelo LSTM (Long Short-Term Memory), desde sua anatomia interna até como ele aprende e faz previsões.

O documento inclui:
1 - O que é LSTM e por que usar
2 - Anatomia da Célula LSTM
3 - Os Três Portões (Gates)
4 - Cell State vs Hidden State
5 - O que são Pesos e Tensores
6 - Como o LSTM Aprende
7 - Ciclo de Vida da Memória
8 - LSTM vs RNN Simples
9 - Perguntas Frequentes

---

## 1. O que é LSTM e Por Que Usar

### 1.1 Definição

**LSTM (Long Short-Term Memory)** é um tipo especial de Rede Neural Recorrente (RNN) projetado para aprender dependências de longo prazo em dados sequenciais.

### 1.2 Por que LSTM para Séries Temporais?

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    O PROBLEMA DAS SÉRIES TEMPORAIS                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Preços de ações são SEQUENCIAIS - a ordem importa!                        │
│                                                                             │
│   Dia 1    Dia 2    Dia 3    Dia 4    Dia 5         Dia ?                  │
│   R$30  →  R$32  →  R$35  →  R$33  →  R$34    →    ???                     │
│                                                                             │
│   O preço de amanhã DEPENDE dos padrões dos dias anteriores.               │
│   Um modelo precisa "lembrar" o que viu antes para prever bem.             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.3 Comparativo: Por que LSTM e não RNN Simples?

| Característica | RNN Simples | LSTM |
|----------------|-------------|------|
| Memória de longo prazo | Ruim (esquece rápido) | Boa (pode lembrar 100+ passos) |
| Vanishing gradient | Sofre muito | Resolve o problema |
| Complexidade | Menor | Maior (4x mais parâmetros) |
| Ideal para | Sequências curtas (<20 passos) | Sequências longas (60+ passos) |

---

## 2. Anatomia da Célula LSTM

### 2.1 Visão Geral

Uma célula LSTM é composta por:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CÉLULA LSTM - COMPONENTES                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                                                                     │   │
│   │   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐           │   │
│   │   │ FORGET GATE  │   │  INPUT GATE  │   │ OUTPUT GATE  │           │   │
│   │   │              │   │              │   │              │           │   │
│   │   │ "O que       │   │ "O que       │   │ "O que       │           │   │
│   │   │  esquecer?"  │   │  adicionar?" │   │  mostrar?"   │           │   │
│   │   └──────────────┘   └──────────────┘   └──────────────┘           │   │
│   │          │                  │                  │                    │   │
│   │          └──────────────────┼──────────────────┘                    │   │
│   │                             │                                       │   │
│   │                             ▼                                       │   │
│   │                    ┌──────────────┐                                 │   │
│   │                    │ CELL STATE   │  ← Memória de longo prazo       │   │
│   │                    │    (c_t)     │                                 │   │
│   │                    └──────────────┘                                 │   │
│   │                             │                                       │   │
│   │                             ▼                                       │   │
│   │                    ┌──────────────┐                                 │   │
│   │                    │HIDDEN STATE  │  ← Saída de curto prazo         │   │
│   │                    │    (h_t)     │                                 │   │
│   │                    └──────────────┘                                 │   │
│   │                                                                     │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Os Dois Estados

| Estado | Nome | Função | Analogia |
|--------|------|--------|----------|
| **c_t** | Cell State | Memória de longo prazo | "Lembro que o protagonista traiu o vilão no episódio 3" |
| **h_t** | Hidden State | Saída de curto prazo | "O que aconteceu na cena anterior?" |

---

## 3. Os Três Portões (Gates)

### 3.1 Como os Portões Funcionam

Cada portão é uma operação matemática que produz valores entre 0 e 1:
- **0 = bloqueado completamente**
- **1 = passa completamente**
- **0.5 = passa metade**

### 3.2 Forget Gate (Portão de Esquecimento)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FORGET GATE                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PERGUNTA: "Quanto da memória anterior devo manter?"                        │
│                                                                             │
│  FÓRMULA:  f_t = sigmoid(W_f × [h_{t-1}, x_t] + b_f)                       │
│                                                                             │
│  EXEMPLO:                                                                   │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  Memória anterior (c_{t-1}): "preço subiu nos últimos 5 dias"              │
│  Nova entrada (x_t): preço CAIU hoje                                        │
│                                                                             │
│  f_t = 0.3 → "Esquecer 70% da memória de alta"                             │
│              (a queda invalida parte da tendência anterior)                 │
│                                                                             │
│  c_t = 0.3 × c_{t-1} + ...                                                 │
│        ↑                                                                    │
│        └── Multiplica a memória anterior pelo fator 0.3                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Input Gate (Portão de Entrada)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           INPUT GATE                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PERGUNTA: "Quanto da nova informação devo adicionar à memória?"           │
│                                                                             │
│  FÓRMULAS:                                                                  │
│  i_t = sigmoid(W_i × [h_{t-1}, x_t] + b_i)    ← Quanto adicionar          │
│  c̃_t = tanh(W_c × [h_{t-1}, x_t] + b_c)      ← O que adicionar           │
│                                                                             │
│  EXEMPLO:                                                                   │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  Nova entrada (x_t): preço subiu 15% (evento significativo!)               │
│                                                                             │
│  i_t = 0.9 → "Prestar MUITA atenção nessa informação"                      │
│  c̃_t = [0.5, 0.3, 0.6] → "Informação codificada: alta forte"             │
│                                                                             │
│  Contribuição = i_t × c̃_t = 0.9 × [0.5, 0.3, 0.6]                        │
│                           = [0.45, 0.27, 0.54]  ← Adicionado à memória    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.4 Output Gate (Portão de Saída)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           OUTPUT GATE                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PERGUNTA: "Quanto da memória atual devo expor como saída?"                │
│                                                                             │
│  FÓRMULAS:                                                                  │
│  o_t = sigmoid(W_o × [h_{t-1}, x_t] + b_o)    ← Quanto mostrar            │
│  h_t = o_t × tanh(c_t)                         ← Saída filtrada           │
│                                                                             │
│  EXEMPLO:                                                                   │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  Memória atual (c_t): contém muita informação acumulada                    │
│                                                                             │
│  o_t = 0.7 → "Mostrar 70% da memória como saída"                           │
│                                                                             │
│  h_t é usado para:                                                          │
│  • Alimentar a próxima célula temporal                                     │
│  • No último passo: fazer a previsão final                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.5 Resumo Visual dos Portões

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     FLUXO COMPLETO DOS PORTÕES                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                      c_{t-1} (memória anterior)                             │
│                           │                                                 │
│            ┌──────────────┼──────────────┐                                 │
│            │              ▼              │                                 │
│            │    ┌─────┐  ─×─  ┌─────┐   │                                 │
│            │    │FORGET│      │INPUT│   │                                 │
│            │    │ f_t │       │ i_t │   │                                 │
│            │    └──┬──┘       └──┬──┘   │                                 │
│            │       │      +      │      │                                 │
│            │       └──────┬──────┘      │                                 │
│            │              │             │                                 │
│            │              ▼             │                                 │
│            │           c_t ─────────────┼─► c_t (nova memória)           │
│            │              │             │                                 │
│            │         ┌────┴────┐        │                                 │
│            │         │ OUTPUT  │        │                                 │
│            │         │  o_t    │        │                                 │
│            │         └────┬────┘        │                                 │
│            │              │             │                                 │
│            └──────────────▼─────────────┘                                 │
│                          h_t ───────────────► h_t (saída)                 │
│                                                                             │
│   RESUMO DAS OPERAÇÕES:                                                     │
│   ─────────────────────                                                     │
│   1. f_t = sigmoid(W_f × [h_{t-1}, x_t] + b_f)     ← Forget                │
│   2. i_t = sigmoid(W_i × [h_{t-1}, x_t] + b_i)     ← Input (quanto)       │
│   3. c̃_t = tanh(W_c × [h_{t-1}, x_t] + b_c)       ← Input (o que)        │
│   4. c_t = f_t × c_{t-1} + i_t × c̃_t              ← Nova memória         │
│   5. o_t = sigmoid(W_o × [h_{t-1}, x_t] + b_o)     ← Output               │
│   6. h_t = o_t × tanh(c_t)                          ← Saída               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Cell State vs Hidden State

### 4.1 Diferenças Fundamentais

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                  CELL STATE vs HIDDEN STATE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  CELL STATE (c_t) - "MEMÓRIA DE LONGO PRAZO"                               │
│  ════════════════════════════════════════════                               │
│                                                                             │
│  • Transporta informação através de MUITOS passos temporais                │
│  • Modificado pelos portões forget e input                                 │
│  • Não é exposto diretamente (uso interno)                                 │
│                                                                             │
│       c₀ ───► c₁ ───► c₂ ───► ... ───► c₆₀                                │
│        │       │       │                 │                                  │
│     (zeros)  (atualizado)            (acumulou contexto)                   │
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  HIDDEN STATE (h_t) - "SAÍDA DE CURTO PRAZO"                               │
│  ════════════════════════════════════════════                               │
│                                                                             │
│  • Saída da célula (vai para a próxima camada ou célula)                  │
│  • Recalculado a cada passo temporal                                       │
│  • O ÚLTIMO h_t é usado para fazer a previsão final                       │
│                                                                             │
│       h₁ ───► h₂ ───► h₃ ───► ... ───► h₆₀ ───► Linear ───► R$           │
│        │       │       │                 │                                  │
│    (saída)  (saída)  (saída)        (PREVISÃO!)                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Analogia: Você Fazendo uma Prova

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ANALOGIA                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  CELL STATE = Seu rascunho durante a prova                                 │
│  ─────────────────────────────────────────                                  │
│  • "Anotei que a questão 1 falou sobre empresa X"                          │
│  • "O enunciado disse que o preço subiu 10%"                               │
│  • Você acumula informações para usar depois                               │
│  • Descarta quando termina a prova                                         │
│                                                                             │
│  HIDDEN STATE = Sua resposta em cada questão                               │
│  ────────────────────────────────────────────                               │
│  • O que você escreve na folha de resposta                                 │
│  • Baseado no seu rascunho + conhecimento                                  │
│  • Cada questão tem uma resposta diferente                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.3 No Código

```python
# forward() em model.py
lstm_out, (h_n, c_n) = self.lstm(x)
#   ↑         ↑   ↑
#   │         │   └── c_n: cell state final (não usamos diretamente)
#   │         └────── h_n: hidden state final
#   └──────────────── lstm_out: TODAS as h_t de cada passo temporal

# Pegamos o último hidden state da sequência
last_output = lstm_out[:, -1, :]  # h₆₀ (último passo)

# Usamos para prever
prediction = self.linear(last_output)  # h₆₀ → Preço previsto
```

---

## 5. O que são Pesos e Tensores

### 5.1 O que é um Tensor?

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        O QUE É UM TENSOR?                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  TENSOR = Array multidimensional de números                                 │
│                                                                             │
│  ESCALAR (0D):        42                                                    │
│                       └── Um único número                                   │
│                                                                             │
│  VETOR (1D):          [1.0, 2.0, 3.0, 4.0, 5.0]                            │
│                       └── Lista de números                                  │
│                                                                             │
│  MATRIZ (2D):         [[1, 2, 3],                                          │
│                        [4, 5, 6],                                           │
│                        [7, 8, 9]]                                           │
│                       └── Tabela de números                                 │
│                                                                             │
│  TENSOR 3D:           [[[1,2], [3,4]],                                     │
│                        [[5,6], [7,8]]]                                      │
│                       └── "Cubo" de números                                 │
│                                                                             │
│  EM PYTORCH:                                                                │
│  tensor = torch.tensor([1.0, 2.0, 3.0])                                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Pesos vs Estados

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      PESOS vs ESTADOS                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PESOS (W, b) - O QUE A REDE APRENDEU                                      │
│  ════════════════════════════════════                                       │
│  • São FIXOS após o treinamento                                            │
│  • São SALVOS no arquivo model.pth                                         │
│  • Determinam o COMPORTAMENTO da rede                                      │
│  • Exemplo: W_f (pesos do forget gate)                                     │
│                                                                             │
│  ANALOGIA: Conhecimento permanente do cérebro                              │
│            "Sei falar português" (não esquece)                             │
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  ESTADOS (h_t, c_t) - MEMÓRIA TEMPORÁRIA                                   │
│  ════════════════════════════════════════                                   │
│  • São RECALCULADOS a cada nova sequência                                  │
│  • NÃO são salvos (são efêmeros)                                           │
│  • Contêm informação sobre a sequência ATUAL                               │
│  • Começam zerados a cada previsão                                         │
│                                                                             │
│  ANALOGIA: Pensamentos momentâneos                                         │
│            "Estou pensando na frase que você disse"                        │
│            (esquece quando troca de assunto)                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.3 Todos os Tensores no Modelo LSTM

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                  TODOS OS TENSORES NO MODELO                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1️⃣ TENSORES DE ENTRADA/SAÍDA (temporários)                                │
│  ──────────────────────────────────────────                                 │
│  • x: entrada (batch_size, seq_length, features)                           │
│       Ex: (32, 60, 1) = 32 amostras de 60 dias                             │
│  • output: saída final (batch_size, 1)                                     │
│       Ex: (32, 1) = 32 previsões de preço                                  │
│                                                                             │
│  2️⃣ ESTADOS (temporários, recalculados)                                    │
│  ──────────────────────────────────────                                     │
│  • h_t: hidden state (num_layers, batch_size, hidden_size)                 │
│       Ex: (2, 32, 50) = 2 camadas, 32 amostras, 50 dimensões              │
│  • c_t: cell state (mesmo shape que h_t)                                   │
│                                                                             │
│  3️⃣ PESOS DA LSTM (aprendidos, salvos no .pth)                             │
│  ──────────────────────────────────────────────                             │
│  Para CADA camada LSTM:                                                     │
│  • W_ii, W_if, W_ig, W_io: pesos input-to-gates                           │
│  • W_hi, W_hf, W_hg, W_ho: pesos hidden-to-gates                          │
│  • b_i, b_f, b_g, b_o: biases para cada gate                              │
│                                                                             │
│  4️⃣ PESOS DA CAMADA LINEAR (aprendidos, salvos)                            │
│  ──────────────────────────────────────────────                             │
│  • W_linear: (hidden_size, 1) = (50, 1)                                    │
│  • b_linear: (1,)                                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Como o LSTM Aprende

### 6.1 Treinamento Supervisionado

O LSTM é treinado de forma **supervisionada**, ou seja, durante o treinamento você fornece:
- **Entrada (X)**: Sequência de dados históricos (ex: 60 dias de preços)
- **Saída esperada (y)**: O valor real que queremos prever (ex: preço do dia 61)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      TREINAMENTO SUPERVISIONADO                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   DADOS:  [dia1, dia2, ..., dia60] → preço_dia61 (conhecido)               │
│                    ↓                         ↓                              │
│                  INPUT                     LABEL                            │
│                    │                         │                              │
│                    ▼                         │                              │
│              ┌──────────┐                    │                              │
│              │  MODELO  │────────────────────┘                              │
│              │   LSTM   │                                                   │
│              └────┬─────┘                                                   │
│                   │                                                         │
│                   ▼                                                         │
│              PREVISÃO ──► COMPARA com LABEL ──► CALCULA ERRO               │
│                                                     │                       │
│                                                     ▼                       │
│                                           AJUSTA OS PESOS                   │
│                                          (Backpropagation)                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 O Processo de Aprendizado

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PROCESSO DE APRENDIZADO                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INÍCIO: Pesos dos portões são ALEATÓRIOS                                  │
│          W_f, W_i, W_o = números aleatórios                                │
│                                                                             │
│  ITERAÇÃO 1:                                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 1. Dados entram: [32.5, 33.1, 33.8, ...]                            │   │
│  │ 2. Portões processam (com pesos aleatórios)                          │   │
│  │ 3. Previsão: R$ 50.00 (péssima!)                                     │   │
│  │ 4. Valor real: R$ 35.20                                              │   │
│  │ 5. Erro: ENORME! (50.00 - 35.20 = 14.80)                            │   │
│  │ 6. Backpropagation: "Esses pesos estão ruins"                        │   │
│  │ 7. Ajusta W_f, W_i, W_o um pouquinho                                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ITERAÇÃO 1000:                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 1. Mesmos dados entram                                               │   │
│  │ 2. Portões processam (pesos ajustados)                               │   │
│  │ 3. Previsão: R$ 35.50 (boa!)                                         │   │
│  │ 4. Valor real: R$ 35.20                                              │   │
│  │ 5. Erro: pequeno (0.30)                                              │   │
│  │ 6. A rede APRENDEU que configuração de pesos funciona               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  O QUE FOI APRENDIDO (implicitamente):                                     │
│  • Forget gate deve "lembrar" ~80% quando preço sobe                       │
│  • Input gate deve "prestar atenção" em quedas bruscas                     │
│  • Output gate deve "focar" nas variações recentes                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.3 O que a Célula "Entende"?

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                  O QUE A CÉLULA REALMENTE "GUARDA"                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ❌ A LSTM NÃO GUARDA (literalmente):                                      │
│     • "Dia 1 teve preço 32.5"                                              │
│     • "Dia 2 teve preço 33.1"                                              │
│     • "Entre dia 1 e 4 subiu"                                              │
│                                                                             │
│  ✅ A LSTM GUARDA (implicitamente nos pesos):                              │
│     • "Quando vejo 4 valores crescentes, próximo tende ser maior"          │
│     • "Quando vejo queda após alta, próximo tende a cair mais"             │
│                                                                             │
│  É ESTATÍSTICA, não "entendimento" real!                                   │
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  O que você imagina:                    O que realmente está armazenado:   │
│  ┌──────────────────────┐              ┌──────────────────────┐            │
│  │ "Tendência de alta   │              │ [0.73, -0.21, 0.89,  │            │
│  │  entre dias 1-4"     │              │  0.02, -0.56, 0.34]  │            │
│  └──────────────────────┘              └──────────────────────┘            │
│         ↑                                       ↑                          │
│   Interpretação humana                  Tensor numérico real               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Ciclo de Vida da Memória

### 7.1 Durante uma Previsão

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CICLO DE VIDA DA MEMÓRIA                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  REQUISIÇÃO: Usuário pede previsão                                         │
│                                                                             │
│  1. API recebe: {"prices": [32.5, 33.1, ..., 42.8]}                        │
│                                                                             │
│  2. CRIA cell state zerado:                                                 │
│     h = torch.zeros(2, 1, 50)  ← NASCE AQUI                                │
│     c = torch.zeros(2, 1, 50)  ← NASCE AQUI                                │
│                                                                             │
│  3. Processa os 60 dias:                                                    │
│     ┌─────────────────────────────────────────────────────────────────┐    │
│     │ for dia in range(60):                                           │    │
│     │     h, c = lstm(x[dia], h, c)  ← ATUALIZA a cada passo         │    │
│     └─────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  4. Faz previsão:                                                           │
│     previsao = linear(h)  → R$ 43.29                                       │
│                                                                             │
│  5. Retorna resposta:                                                       │
│     {"predicted_price": 43.29}                                              │
│                                                                             │
│  6. FUNÇÃO TERMINA → h e c são DESTRUÍDOS (garbage collected)              │
│                      ↑                                                      │
│                      └── MORRE AQUI! Não existe mais.                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Memória NÃO Persiste Entre Requisições

```
┌─────────────────────────────────────────────────────────────────────────────┐
│            CADA REQUISIÇÃO COMEÇA DO ZERO                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  TEMPO ──────────────────────────────────────────────────────────────►     │
│                                                                             │
│  REQUISIÇÃO 1 (PETR4)              REQUISIÇÃO 2 (VALE3)                    │
│  ┌─────────────────┐              ┌─────────────────┐                      │
│  │ h, c criados    │              │ h, c criados    │                      │
│  │ h, c usados     │              │ h, c usados     │                      │
│  │ h, c DESTRUÍDOS │              │ h, c DESTRUÍDOS │                      │
│  └────────┬────────┘              └────────┬────────┘                      │
│           │                                │                                │
│           ▼                                ▼                                │
│      MEMÓRIA VAZIA                    MEMÓRIA VAZIA                        │
│      (não existe h, c)                (não existe h, c)                    │
│                                                                             │
│                                                                             │
│  PESOS (W, b) ─────────────────────────────────────────────────────────    │
│                SEMPRE EXISTEM (carregados do .pth)                         │
│                Mesmos valores para TODAS as requisições                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.3 Por que Funciona Assim?

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                POR QUE CELL STATE NÃO É SALVO?                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Faz sentido! Pense:                                                        │
│                                                                             │
│  Se eu pedi previsão de PETR4 ontem...                                     │
│  ...e hoje peço previsão de VALE3...                                       │
│  ...por que a memória de PETR4 seria útil para VALE3?                      │
│                                                                             │
│  CADA SEQUÊNCIA É INDEPENDENTE!                                             │
│                                                                             │
│  O que importa para prever VALE3:                                          │
│  ✅ Os PESOS (padrões gerais que a rede aprendeu)                          │
│  ✅ Os 60 dias de VALE3 que eu enviei                                      │
│  ❌ A memória de processamento de PETR4 (irrelevante)                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 8. LSTM vs RNN Simples

### 8.1 Por que RNN Simples tem Memória Curta?

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                POR QUE RNN TEM MEMÓRIA CURTA?                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  O PROBLEMA: VANISHING GRADIENT                                             │
│  ─────────────────────────────                                              │
│                                                                             │
│  Durante o BACKPROPAGATION, o gradiente é multiplicado                     │
│  repetidamente ao voltar no tempo:                                          │
│                                                                             │
│  Passo 100 ← Passo 99 ← Passo 98 ← ... ← Passo 1                          │
│      │          │          │                │                               │
│      ×0.8      ×0.8       ×0.8            ×0.8                             │
│                                                                             │
│  Gradiente no passo 1 = 0.8^100 = 0.0000000000000002                       │
│                                   ↑                                         │
│                              PRATICAMENTE ZERO!                             │
│                                                                             │
│  RESULTADO:                                                                 │
│  • Os pesos que processam passos antigos NÃO SÃO ATUALIZADOS              │
│  • A rede "desiste" de aprender padrões de longo prazo                    │
│  • Só aprende padrões dos últimos 10-20 passos                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 8.2 Como LSTM Resolve

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    COMO LSTM RESOLVE                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  RNN:  h_t = tanh(W × [h_{t-1}, x_t])                                      │
│              ↑                                                              │
│              └── Multiplicação a cada passo (gradiente some)               │
│                                                                             │
│  LSTM: c_t = f_t × c_{t-1} + i_t × c̃_t                                    │
│              ↑                                                              │
│              └── SOMA! O gradiente pode fluir inalterado                   │
│                                                                             │
│  O cell state é um "highway" onde a informação passa                       │
│  sem ser multiplicada por pesos pequenos.                                  │
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  ILUSTRAÇÃO:                                                                │
│                                                                             │
│  RNN (informação se PERDE):                                                │
│  Dia 1:  ████████████  (100% da informação)                                │
│  Dia 5:  ████████      (80%)                                               │
│  Dia 10: ████          (40%)                                               │
│  Dia 20: █             (10%)                                               │
│  Dia 50:               (~0%)                                               │
│                                                                             │
│  LSTM (informação PERSISTE):                                               │
│  Dia 1:  ████████████  (100%)                                              │
│  Dia 5:  ███████████   (95% - forget gate removeu pouco)                   │
│  Dia 10: ██████████    (85%)                                               │
│  Dia 20: █████████     (75%)                                               │
│  Dia 50: ███████       (60% - ainda significativo!)                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 8.3 Diferenças Estruturais

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     RNN SIMPLES vs LSTM                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  RNN SIMPLES:                                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │   x_t ──► [  tanh  ] ──► h_t ──►                                   │   │
│  │               ↑                 │                                   │   │
│  │               └─────────────────┘                                   │   │
│  │                                                                     │   │
│  │   Apenas UMA operação:                                              │   │
│  │   h_t = tanh(W_x · x_t + W_h · h_{t-1} + b)                        │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  LSTM:                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │   x_t ──► [Forget] ──► [Input] ──► [Cell] ──► [Output] ──► h_t    │   │
│  │               │            │          │           │                │   │
│  │               └────────────┴──────────┴───────────┘                │   │
│  │                           c_t (memória)                             │   │
│  │                                                                     │   │
│  │   QUATRO operações + dois estados (h e c)                          │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

| Aspecto | RNN Simples | LSTM |
|---------|-------------|------|
| Estados | Só h_t | h_t + c_t |
| Operações | 1 (tanh) | 4 (3 portões + 1 candidato) |
| Parâmetros | Menos (~4x menos) | Mais (4x mais) |
| Memória longa | Ruim | Boa |
| Vanishing gradient | Sofre muito | Resolve |

### 8.4 LSTM É uma RNN

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                  HIERARQUIA DE REDES RECORRENTES                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                         RNN (conceito geral)                                │
│                         ════════════════════                                │
│                                │                                            │
│             ┌──────────────────┼──────────────────┐                        │
│             │                  │                  │                         │
│             ▼                  ▼                  ▼                         │
│       ┌──────────┐      ┌──────────┐      ┌──────────┐                    │
│       │   RNN    │      │   LSTM   │      │   GRU    │                    │
│       │ Simples  │      │          │      │          │                    │
│       │(Vanilla) │      │          │      │          │                    │
│       └──────────┘      └──────────┘      └──────────┘                    │
│                                                                             │
│  TODOS são RNNs porque:                                                     │
│  • Processam sequências                                                     │
│  • Têm conexões recorrentes (saída → entrada do próximo)                   │
│  • Mantêm alguma forma de estado/memória                                   │
│                                                                             │
│  LSTM É uma RNN, só que com célula mais sofisticada.                       │
│                                                                             │
│  É como perguntar: "Um carro elétrico usa carro?"                          │
│  Não, ele É um carro, só que com motor diferente.                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 9. Perguntas Frequentes

### 9.1 Sobre Treinamento

**P: LSTM é treinado de forma supervisionada ou não supervisionada?**

R: **Supervisionada.** Você fornece X (entrada) e y (valor esperado). O modelo aprende comparando sua previsão com o valor real.

**P: Os portões são executados no treinamento e na inferência?**

R: **Sim, em ambos.** O forward pass (que usa os portões) acontece sempre. A diferença é:
- Treinamento: Forward + Backward (atualiza pesos)
- Inferência: Só Forward (usa pesos fixos)

### 9.2 Sobre a Arquitetura

**P: O que são as "células de memória" do LSTM?**

R: São os estados (h_t e c_t) que guardam informação temporária durante o processamento de uma sequência. Não são os padrões aprendidos (esses são os pesos).

**P: Por que múltiplas camadas LSTM?**

R: Cada camada captura diferentes níveis de abstração:
- Camada 1: padrões simples ("preço subiu hoje")
- Camada 2: padrões complexos ("tendência de alta nas últimas semanas")

### 9.3 Sobre Implementação

**P: Os portões são importados de biblioteca ou implementados manualmente?**

R: **Importados do PyTorch.** O `nn.LSTM` implementa internamente toda a lógica dos portões. Você só define a arquitetura.

**P: Onde ficam salvos os pesos do modelo?**

R: No arquivo `.pth` (ex: `models/model_lstm.pth`). São carregados uma vez quando a API inicia.

**P: O cell state é salvo em algum lugar?**

R: **Não.** É criado do zero a cada previsão e destruído quando a função termina.

---

## 10. Resumo Executivo

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          RESUMO: LSTM                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  O QUE É:        Tipo especial de RNN com memória de longo prazo           │
│                                                                             │
│  COMPONENTES:    3 portões (forget, input, output) + 2 estados (h, c)      │
│                                                                             │
│  PESOS (W, b):   Padrões APRENDIDOS, salvos em .pth, FIXOS após treino    │
│                                                                             │
│  ESTADOS (h, c): Memória TEMPORÁRIA, recriada a cada previsão             │
│                                                                             │
│  TREINAMENTO:    Supervisionado (X + y esperado → aprende por erro)       │
│                                                                             │
│  VANTAGEM:       Resolve vanishing gradient (memória de longo prazo)       │
│                                                                             │
│  IMPLEMENTAÇÃO:  Importado do PyTorch (nn.LSTM)                            │
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  FLUXO DE UMA PREVISÃO:                                                     │
│                                                                             │
│  1. Recebe 60 dias de preços                                               │
│  2. Cria h=0, c=0 (memória zerada)                                         │
│  3. Processa dia a dia (portões atualizam h e c)                           │
│  4. Último h vai para Linear → Previsão                                    │
│  5. h e c são destruídos (efêmeros)                                        │
│  6. Pesos permanecem (permanentes)                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

*Documento gerado com base em sessão de estudo sobre LSTM - Tech Challenge Fase 4*
