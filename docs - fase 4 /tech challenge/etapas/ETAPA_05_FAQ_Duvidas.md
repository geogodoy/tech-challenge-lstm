# ❓ FAQ - Dúvidas da Etapa 5: Treinamento

> Documento complementar à [ETAPA_05_Treinamento.md](./ETAPA_05_Treinamento.md)

---

## 📚 Índice

1. [Como funciona o treinamento de forma geral?](#1-como-funciona-o-treinamento-de-forma-geral)
2. [O treinamento é executável manualmente?](#2-o-treinamento-é-executável-manualmente)
3. [Quem avalia o treinamento: humano ou máquina?](#3-quem-avalia-o-treinamento-humano-ou-máquina)
4. [O que é avaliado na etapa de treinamento?](#4-o-que-é-avaliado-na-etapa-de-treinamento)
5. [Como funciona o ciclo de treinamento na prática?](#5-como-funciona-o-ciclo-de-treinamento-na-prática)
6. [O que é Forward Pass?](#6-o-que-é-forward-pass)
7. [O que é Loss (Função de Perda)?](#7-o-que-é-loss-função-de-perda)
8. [O que é Backward Pass (Backpropagation)?](#8-o-que-é-backward-pass-backpropagation)
9. [O que faz o optimizer.step()?](#9-o-que-faz-o-optimizerstep)
10. [Por que usar optimizer.zero_grad()?](#10-por-que-usar-optimizerzero_grad)
11. [Qual a diferença entre model.train() e model.eval()?](#11-qual-a-diferença-entre-modeltrain-e-modeleval)
12. [O que é torch.no_grad() e por que usar?](#12-o-que-é-torchno_grad-e-por-que-usar)
13. [O que são Épocas (Epochs)?](#13-o-que-são-épocas-epochs)
14. [O que é Learning Rate (Taxa de Aprendizado)?](#14-o-que-é-learning-rate-taxa-de-aprendizado)
15. [Por que escolhemos o otimizador Adam?](#15-por-que-escolhemos-o-otimizador-adam)
16. [Por que escolhemos MSELoss?](#16-por-que-escolhemos-mseloss)
17. [O que é Overfitting e Underfitting?](#17-o-que-é-overfitting-e-underfitting)
18. [Como interpretar os valores de Loss?](#18-como-interpretar-os-valores-de-loss)
19. [GPU vs CPU: qual usar?](#19-gpu-vs-cpu-qual-usar)
20. [Por que salvar o modelo em formato .pth?](#20-por-que-salvar-o-modelo-em-formato-pth)
21. [O modelo "aprende" sozinho?](#21-o-modelo-aprende-sozinho)
22. [O treinamento é determinístico?](#22-o-treinamento-é-determinístico)
23. [Posso mudar a arquitetura depois de treinar?](#23-posso-mudar-a-arquitetura-depois-de-treinar)
24. [O que é Early Stopping?](#24-o-que-é-early-stopping)
25. [Outras dúvidas comuns](#25-outras-dúvidas-comuns)

---

## 1. Como funciona o treinamento de forma geral?

**Referência:** Documento principal, seção "Fluxo do Treinamento"

### Analogia: Aprender a jogar dardos

Imagine que você está aprendendo a acertar o centro de um alvo de dardos:

```
┌────────────────────────────────────────────────────────────────┐
│                    APRENDENDO DARDOS                           │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  1. JOGAR      →   Você joga o dardo (Forward Pass)            │
│  2. VER ERRO   →   Mede distância do centro (Loss)             │
│  3. ENTENDER   →   Analisa o que fez errar (Backward Pass)     │
│  4. AJUSTAR    →   Corrige a mira (Atualizar Pesos)            │
│  5. REPETIR    →   Joga novamente, agora melhor                │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### No contexto do modelo LSTM

| Passo | Analogia Dardos | No Código | O que acontece |
|-------|-----------------|-----------|----------------|
| 1 | Jogar o dardo | `model(X_train)` | Dados entram, previsão sai |
| 2 | Medir distância | `criterion(outputs, y)` | Compara previsão com realidade |
| 3 | Entender o erro | `loss.backward()` | Calcula contribuição de cada peso |
| 4 | Ajustar mira | `optimizer.step()` | Atualiza pesos para errar menos |
| 5 | Repetir | Loop `for epoch` | Faz isso 100 vezes (épocas) |

---

## 2. O treinamento é executável manualmente?

**Resposta: SIM!**

### Forma 1: Executar o script diretamente

```bash
cd tech-challenge-lstm
python src/train.py
```

### Forma 2: Importar a função em outro código

```python
from train import train_model
from model import create_model
from preprocessing import preprocess_data

# Carregar dados
X_train, X_test, y_train, y_test, scaler = preprocess_data()

# Criar modelo
model = create_model()

# Treinar (você controla os parâmetros!)
model, train_losses, val_losses = train_model(
    model=model,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    epochs=100,           # ← Configurável!
    learning_rate=0.001   # ← Configurável!
)
```

### Parâmetros que você pode configurar

| Parâmetro | Valor Padrão | O que faz | Quando alterar |
|-----------|--------------|-----------|----------------|
| `epochs` | 100 | Quantas vezes ver todos os dados | Aumentar se loss ainda está caindo |
| `learning_rate` | 0.001 | Velocidade de aprendizado | Diminuir se loss oscila muito |
| `device` | Auto | GPU ou CPU | Forçar 'cpu' se GPU der erro |
| `verbose` | True | Imprimir progresso | False para rodar silenciosamente |

---

## 3. Quem avalia o treinamento: humano ou máquina?

**Resposta: OS DOIS!**

### O que a máquina faz automaticamente

```
┌─────────────────────────────────────────────────────────────┐
│                   AVALIAÇÃO AUTOMÁTICA                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  A cada época, o PyTorch automaticamente:                   │
│                                                             │
│  ✓ Calcula Train Loss (erro nos dados de treino)            │
│  ✓ Calcula Val Loss (erro nos dados de validação)           │
│  ✓ Registra o histórico de perdas                           │
│  ✓ Identifica o melhor modelo (menor val_loss)              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### O que o humano (você) precisa fazer

```
┌─────────────────────────────────────────────────────────────┐
│                   AVALIAÇÃO HUMANA                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Você precisa interpretar os resultados:                    │
│                                                             │
│  ❓ O loss está diminuindo? (modelo aprendendo)             │
│  ❓ Val loss está subindo enquanto train desce? (overfitting)│
│  ❓ Os valores fazem sentido para o problema?               │
│  ❓ Preciso ajustar hiperparâmetros?                        │
│  ❓ O modelo está bom o suficiente para usar?               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Resumo

| Tarefa | Quem faz | Como |
|--------|----------|------|
| Calcular métricas | Máquina | Automaticamente no loop |
| Atualizar pesos | Máquina | `optimizer.step()` |
| Interpretar resultados | Humano | Olhar gráficos e valores |
| Decidir se está bom | Humano | Baseado na experiência |
| Ajustar hiperparâmetros | Humano | Tentativa e erro informada |

---

## 4. O que é avaliado na etapa de treinamento?

**Referência:** Linhas 186-229 do documento principal

### Métricas Monitoradas

| Métrica | O que mede | Fórmula | Interpretação |
|---------|------------|---------|---------------|
| **Train Loss** | Erro nos dados de treino | MSE(previsões, reais) | Quão bem o modelo "decorou" os dados |
| **Val Loss** | Erro nos dados de validação | MSE(previsões, reais) | Quão bem o modelo generaliza |

### Por que duas métricas?

```
┌───────────────────────────────────────────────────────────────┐
│              ANALOGIA: PROVA NA ESCOLA                        │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│  Train Loss = Nota nos exercícios que você estudou            │
│  Val Loss   = Nota na prova com questões novas                │
│                                                               │
│  ┌─────────────────┬─────────────────────────────────────┐    │
│  │ Situação        │ O que significa                     │    │
│  ├─────────────────┼─────────────────────────────────────┤    │
│  │ Train ↓ Val ↓   │ ✅ Aprendendo e generalizando      │    │
│  │ Train ↓ Val →   │ ⚠️ Começando a decorar demais      │    │
│  │ Train ↓ Val ↑   │ ❌ Overfitting (decorou, não aprendeu) │ │
│  │ Train → Val →   │ ⚠️ Modelo estagnado                │    │
│  └─────────────────┴─────────────────────────────────────┘    │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

### Diagnóstico Visual

```
Loss
  │
  │    ╲
  │     ╲  Val Loss
  │      ╲ 
  │       ╲_______________  ← Ideal: ambos diminuem e estabilizam
  │        ╲
  │         ╲ Train Loss
  │          ╲____________
  │
  └─────────────────────────► Época
```

---

## 5. Como funciona o ciclo de treinamento na prática?

**Referência:** Linhas 106-113 do código `train.py`

### O código real

```python
# FASE DE TREINO - Uma iteração
model.train()                              # 1. Ativa modo treino
outputs = model(X_train)                   # 2. Forward Pass
loss = criterion(outputs, y_train)         # 3. Calcular Loss

optimizer.zero_grad()                      # 4. Limpar gradientes anteriores
loss.backward()                            # 5. Backward Pass (Backpropagation)
optimizer.step()                           # 6. Atualizar pesos
```

### Fluxo detalhado com exemplo numérico

```
┌─────────────────────────────────────────────────────────────────┐
│                    EXEMPLO PRÁTICO                              │
│              (Previsão de preço de ação)                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ENTRADA: 60 dias de preços normalizados [0.2, 0.3, 0.25, ...]  │
│                                                                 │
│  ═══════════════════════════════════════════════════════════    │
│                                                                 │
│  1️⃣ FORWARD PASS                                               │
│     model(X_train) executa:                                     │
│     - Dados passam pela camada LSTM 1                           │
│     - Dados passam pela camada LSTM 2                           │
│     - Dados passam pela camada Linear (fully connected)         │
│     - Resultado: previsão = 0.45 (normalizado)                  │
│                                                                 │
│  2️⃣ CALCULAR LOSS                                              │
│     Valor real (y_train) = 0.42                                 │
│     loss = (0.45 - 0.42)² = 0.0009                              │
│     → O modelo errou um pouquinho                               │
│                                                                 │
│  3️⃣ BACKWARD PASS                                              │
│     loss.backward() calcula:                                    │
│     - Quanto o peso W1 contribuiu pro erro? → gradiente de W1   │
│     - Quanto o peso W2 contribuiu? → gradiente de W2            │
│     - ... para TODOS os milhares de pesos                       │
│                                                                 │
│  4️⃣ ATUALIZAR PESOS                                            │
│     optimizer.step() faz:                                       │
│     - W1_novo = W1_antigo - 0.001 × gradiente_W1                │
│     - W2_novo = W2_antigo - 0.001 × gradiente_W2                │
│     - ... para todos os pesos                                   │
│                                                                 │
│  5️⃣ REPETIR                                                    │
│     Fazer tudo de novo com os pesos atualizados                 │
│     → Na próxima vez, a previsão será mais próxima de 0.42      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Quais funções/APIs são chamadas?

| Etapa | Função PyTorch | O que faz internamente |
|-------|----------------|------------------------|
| Forward | `model(X)` | Chama `model.forward(X)` |
| Loss | `criterion(pred, real)` | Calcula `(pred - real)²` e faz média |
| Limpar | `optimizer.zero_grad()` | Zera `.grad` de todos os parâmetros |
| Backward | `loss.backward()` | Usa autograd para calcular gradientes |
| Atualizar | `optimizer.step()` | Aplica Adam: `w = w - lr * grad` |

---

## 6. O que é Forward Pass?

**Referência:** Linha 107 do código

### Definição

**Forward Pass** é quando os dados de entrada **atravessam** todas as camadas do modelo da esquerda para a direita, produzindo uma previsão.

### Visualização

```
┌─────────────────────────────────────────────────────────────────┐
│                       FORWARD PASS                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Entrada          Camadas do Modelo            Saída            │
│  (60 dias)                                     (previsão)       │
│                                                                 │
│  [0.2]  ─┐                                                      │
│  [0.3]  ─┼──► [LSTM 1] ──► [LSTM 2] ──► [Linear] ──► [0.45]    │
│  [0.25] ─┤        ↓           ↓            ↓                    │
│  [...]  ─┘     h₁, c₁      h₂, c₂       output                  │
│                                                                 │
│  ════════════════════════════════════════════════════════════   │
│  Direção do fluxo: ENTRADA → SAÍDA (forward = para frente)      │
└─────────────────────────────────────────────────────────────────┘
```

### No código

```python
outputs = model(X_train)  # Forward pass acontece aqui
```

Quando você chama `model(X_train)`, o PyTorch automaticamente executa o método `forward()` da classe `StockLSTM`.

---

## 7. O que é Loss (Função de Perda)?

**Referência:** Linhas 72, 108 do código

### Definição

**Loss** (ou função de perda) é um **número que mede quão errado** o modelo está. Quanto menor o loss, melhor o modelo.

### MSE (Mean Squared Error)

```
MSE = (1/n) × Σ(previsão - real)²

Exemplo com 3 previsões:
┌──────────┬──────────┬───────────┬───────────────┐
│ Previsão │ Real     │ Erro      │ Erro²         │
├──────────┼──────────┼───────────┼───────────────┤
│ 0.45     │ 0.42     │ 0.03      │ 0.0009        │
│ 0.38     │ 0.40     │ -0.02     │ 0.0004        │
│ 0.50     │ 0.48     │ 0.02      │ 0.0004        │
├──────────┴──────────┴───────────┼───────────────┤
│                          Soma   │ 0.0017        │
│                          Média  │ 0.000567      │
└─────────────────────────────────┴───────────────┘

Loss = 0.000567
```

### Por que elevar ao quadrado?

| Motivo | Explicação |
|--------|------------|
| **Penaliza erros grandes** | Erro de 2 vira 4, mas erro de 0.1 vira 0.01 |
| **Sempre positivo** | Erro -0.5 vira 0.25, não cancela com erro +0.5 |
| **Derivada suave** | Facilita o cálculo do gradiente |

### No código

```python
criterion = nn.MSELoss()                    # Define a função de perda
loss = criterion(outputs, y_train)          # Calcula o loss
print(f"Loss: {loss.item():.6f}")           # Ex: Loss: 0.000567
```

---

## 8. O que é Backward Pass (Backpropagation)?

**Referência:** Linha 112 do código

### Definição

**Backward Pass** (ou Backpropagation) é o processo de **calcular quanto cada peso contribuiu para o erro**. Funciona "de trás para frente" - do erro de volta até a entrada.

### Analogia: Descobrir quem errou

```
┌─────────────────────────────────────────────────────────────────┐
│              ANALOGIA: FÁBRICA COM PROBLEMA                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Produto defeituoso detectado no final da linha!                │
│                                                                 │
│  [Matéria Prima] → [Setor A] → [Setor B] → [Setor C] → [DEFEITO]│
│                                                                 │
│  Backpropagation = Investigar de trás pra frente:               │
│  - Setor C contribuiu 40% pro defeito                           │
│  - Setor B contribuiu 35% pro defeito                           │
│  - Setor A contribuiu 25% pro defeito                           │
│                                                                 │
│  Agora sabemos onde ajustar mais!                               │
└─────────────────────────────────────────────────────────────────┘
```

### No contexto de redes neurais

```
┌─────────────────────────────────────────────────────────────────┐
│                     BACKPROPAGATION                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Forward:  Entrada → LSTM1 → LSTM2 → Linear → Previsão          │
│                                                     ↓           │
│                                                   LOSS          │
│                                                     ↓           │
│  Backward: Entrada ← LSTM1 ← LSTM2 ← Linear ← gradientes        │
│               ↑         ↑       ↑       ↑                       │
│            ∂L/∂w₁   ∂L/∂w₂  ∂L/∂w₃  ∂L/∂w₄                      │
│                                                                 │
│  Cada peso agora "sabe" quanto contribuiu para o erro           │
└─────────────────────────────────────────────────────────────────┘
```

### Matematicamente (simplificado)

Para cada peso `w`, calcula-se a **derivada parcial** do loss em relação a esse peso:

```
∂Loss/∂w = quanto o loss muda se w mudar um pouquinho
```

### No código

```python
loss.backward()  # Calcula ∂L/∂w para TODOS os pesos automaticamente
```

O PyTorch usa **autograd** (diferenciação automática) - você não precisa calcular as derivadas manualmente!

---

## 9. O que faz o optimizer.step()?

**Referência:** Linha 113 do código

### Definição

`optimizer.step()` **atualiza todos os pesos** do modelo usando os gradientes calculados no backward pass.

### Fórmula básica do gradiente descendente

```
peso_novo = peso_antigo - learning_rate × gradiente

Exemplo:
peso_antigo = 0.5
gradiente = 0.1 (calculado no backward)
learning_rate = 0.001

peso_novo = 0.5 - 0.001 × 0.1 = 0.4999
```

### Por que subtrair?

```
┌─────────────────────────────────────────────────────────────────┐
│               GRADIENTE DESCENDENTE VISUAL                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Loss │                                                         │
│       │    ╲                                                    │
│       │     ╲   ← Queremos ir DESCENDO o "morro" do loss        │
│       │      ╲                                                  │
│       │       ╲                                                 │
│       │        ●  ← Começamos aqui (loss alto)                  │
│       │         ╲                                               │
│       │          ╲                                              │
│       │           ● ← Depois de step() (loss menor)             │
│       │            ╲____                                        │
│       │                 ╲____●  ← Mínimo (objetivo)             │
│       └──────────────────────────────────────────► peso         │
│                                                                 │
│  O gradiente aponta para CIMA (onde loss aumenta)               │
│  Por isso SUBTRAÍMOS: andamos na direção oposta (descida)       │
└─────────────────────────────────────────────────────────────────┘
```

### Adam vs Gradiente Descendente Simples

| Aspecto | SGD Simples | Adam |
|---------|-------------|------|
| Fórmula | `w = w - lr × grad` | Mais complexa (adaptativa) |
| Learning rate | Fixo para todos | Ajustado por parâmetro |
| Momento | Não tem | Usa momento (histórico) |
| Performance | OK | Melhor para LSTMs |

### No código

```python
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Configura o Adam
# ... (forward e backward)
optimizer.step()  # Atualiza TODOS os pesos do modelo
```

---

## 10. Por que usar optimizer.zero_grad()?

**Referência:** Linha 111 do código

### O problema: gradientes acumulam

Por padrão, o PyTorch **soma** os gradientes a cada `backward()`. Isso é útil em alguns casos, mas geralmente queremos gradientes "limpos".

### Exemplo do problema

```python
# SEM zero_grad()
loss.backward()      # gradiente = 0.1
loss.backward()      # gradiente = 0.1 + 0.1 = 0.2  ← Errado!
loss.backward()      # gradiente = 0.2 + 0.1 = 0.3  ← Mais errado!

# COM zero_grad()
optimizer.zero_grad()
loss.backward()      # gradiente = 0.1  ← Correto!
```

### Visualização

```
┌─────────────────────────────────────────────────────────────────┐
│                SEM zero_grad() (ERRADO)                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Época 1: gradiente = 0.1                                       │
│  Época 2: gradiente = 0.1 + 0.1 = 0.2   ← Acumulou!             │
│  Época 3: gradiente = 0.2 + 0.1 = 0.3   ← Cada vez pior         │
│                                                                 │
│  Os pesos vão "explodir" ou oscilar loucamente                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                COM zero_grad() (CORRETO)                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Época 1: zero_grad() → gradiente = 0.1                         │
│  Época 2: zero_grad() → gradiente = 0.1   ← Limpo!              │
│  Época 3: zero_grad() → gradiente = 0.1   ← Sempre limpo        │
│                                                                 │
│  Os pesos atualizam corretamente a cada época                   │
└─────────────────────────────────────────────────────────────────┘
```

### No código

```python
optimizer.zero_grad()  # SEMPRE antes de backward()
loss.backward()
optimizer.step()
```

---

## 11. Qual a diferença entre model.train() e model.eval()?

**Referência:** Linhas 104, 118 do código

### Diferença principal

| Modo | Dropout | BatchNorm | Quando usar |
|------|---------|-----------|-------------|
| `model.train()` | ATIVO (desliga neurônios aleatórios) | Atualiza estatísticas | Durante treino |
| `model.eval()` | DESATIVO (usa todos os neurônios) | Usa estatísticas fixas | Durante validação/inferência |

### Por que isso importa?

```
┌─────────────────────────────────────────────────────────────────┐
│                      model.train()                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Neurônios: [●] [○] [●] [●] [○] [●]   ← Alguns desligados       │
│                 ↑           ↑                                   │
│              Dropout=0.2 desliga 20% aleatoriamente             │
│                                                                 │
│  Por quê? Força o modelo a não depender demais de um neurônio   │
│           Ajuda a prevenir overfitting                          │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      model.eval()                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Neurônios: [●] [●] [●] [●] [●] [●]   ← Todos ligados           │
│                                                                 │
│  Por quê? Na hora de prever "de verdade", queremos usar         │
│           toda a capacidade do modelo                           │
└─────────────────────────────────────────────────────────────────┘
```

### No código

```python
# TREINO
model.train()                    # Ativa dropout
outputs = model(X_train)
loss = criterion(outputs, y_train)
loss.backward()
optimizer.step()

# VALIDAÇÃO
model.eval()                     # Desativa dropout
with torch.no_grad():
    val_outputs = model(X_test)
    val_loss = criterion(val_outputs, y_test)
```

---

## 12. O que é torch.no_grad() e por que usar?

**Referência:** Linha 119 do código

### Definição

`torch.no_grad()` **desativa o cálculo de gradientes** temporariamente.

### Por que usar na validação?

| Com gradientes | Sem gradientes (no_grad) |
|----------------|--------------------------|
| Consome memória para guardar operações | Não guarda nada |
| Mais lento | Mais rápido |
| Necessário para backward() | Não precisa de backward() |
| Usar no treino | Usar na validação/inferência |

### Exemplo de economia

```
┌─────────────────────────────────────────────────────────────────┐
│              ECONOMIA COM torch.no_grad()                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  SEM no_grad():                                                 │
│  - PyTorch guarda todas as operações intermediárias             │
│  - Usa ~2GB de memória GPU para nosso modelo                    │
│  - Mais lento                                                   │
│                                                                 │
│  COM no_grad():                                                 │
│  - PyTorch só calcula o resultado final                         │
│  - Usa ~200MB de memória GPU                                    │
│  - ~2x mais rápido                                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### No código

```python
# ERRADO (funciona, mas desperdiça recursos)
model.eval()
val_outputs = model(X_test)

# CORRETO
model.eval()
with torch.no_grad():              # ← Importante!
    val_outputs = model(X_test)
    val_loss = criterion(val_outputs, y_test)
```

---

## 13. O que são Épocas (Epochs)?

**Referência:** Linha 24 do código (`EPOCHS = 100`)

### Definição

Uma **época** é **uma passagem completa** por todos os dados de treinamento.

### Analogia: Estudar para prova

```
┌─────────────────────────────────────────────────────────────────┐
│                     ANALOGIA: ESTUDAR                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Você tem um livro com 1000 páginas para estudar.               │
│                                                                 │
│  1 Época = Ler o livro inteiro uma vez                          │
│                                                                 │
│  - Época 1: Primeira leitura (entende pouco)                    │
│  - Época 2: Segunda leitura (entende mais)                      │
│  - Época 10: Décima leitura (domina o conteúdo)                 │
│  - Época 100: Centésima leitura (expert, mas cansado)           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Quantas épocas usar?

| Épocas | Resultado típico |
|--------|------------------|
| Poucas (10-20) | Modelo não aprende o suficiente (underfitting) |
| Médio (50-100) | Geralmente bom equilíbrio |
| Muitas (500+) | Risco de decorar os dados (overfitting) |

### Como saber se precisa de mais?

```
Se no final do treino:
- Loss ainda está caindo → Talvez precise de mais épocas
- Loss estabilizou → Épocas suficientes
- Val Loss subindo → PARE! Está overfitando
```

---

## 14. O que é Learning Rate (Taxa de Aprendizado)?

**Referência:** Linha 25 do código (`LEARNING_RATE = 0.001`)

### Definição

**Learning Rate** (η ou lr) controla **o tamanho do passo** que o modelo dá a cada atualização de peso.

### Analogia: Descendo uma montanha com neblina

```
┌─────────────────────────────────────────────────────────────────┐
│          LEARNING RATE: TAMANHO DO PASSO                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Você está no topo de uma montanha, na neblina, tentando        │
│  chegar ao vale (menor loss). Você só sente a inclinação        │
│  do chão sob seus pés (gradiente).                              │
│                                                                 │
│  Learning Rate ALTO (0.1):                                      │
│  ┌────────────────────────────┐                                 │
│  │    ●                       │  Passos grandes                 │
│  │     ╲                      │  → Pode pular o vale            │
│  │      ╲    ●                │  → Pode oscilar dos dois lados  │
│  │       ╲  ╱ ╲               │                                 │
│  │        ╲╱   ╲   ●          │                                 │
│  └────────────────────────────┘                                 │
│                                                                 │
│  Learning Rate BAIXO (0.0001):                                  │
│  ┌────────────────────────────┐                                 │
│  │    ●                       │  Passos minúsculos              │
│  │    ●                       │  → Muito lento                  │
│  │     ●                      │  → Pode demorar 10000 épocas    │
│  │     ●                      │                                 │
│  │      ●                     │                                 │
│  └────────────────────────────┘                                 │
│                                                                 │
│  Learning Rate BOM (0.001):                                     │
│  ┌────────────────────────────┐                                 │
│  │    ●                       │  Passos equilibrados            │
│  │     ●                      │  → Converge em tempo razoável   │
│  │       ●                    │  → Chega perto do mínimo        │
│  │         ●                  │                                 │
│  │           ●____            │                                 │
│  └────────────────────────────┘                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Valores típicos

| Valor | Quando usar |
|-------|-------------|
| 0.1 | Quase nunca (muito alto) |
| 0.01 | Às vezes para SGD |
| 0.001 | Padrão para Adam (nosso caso) |
| 0.0001 | Fine-tuning de modelos pré-treinados |

---

## 15. Por que escolhemos o otimizador Adam?

**Referência:** Linha 75 do código, Seção "Adam - Por que escolhemos?" do documento principal

### Comparação de otimizadores

| Otimizador | Vantagens | Desvantagens | Quando usar |
|------------|-----------|--------------|-------------|
| **SGD** | Simples, teórico | Lento, sensível a lr | Modelos simples |
| **SGD + Momentum** | Mais rápido que SGD | Ainda sensível | CNNs |
| **RMSprop** | Adaptativo | Pode ser instável | RNNs |
| **Adam** | Adaptativo, robusto | Usa mais memória | **LSTMs, padrão geral** |

### O que Adam faz de especial?

```
┌─────────────────────────────────────────────────────────────────┐
│                    ADAM: O OTIMIZADOR                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Adam = Adaptive Moment Estimation                              │
│                                                                 │
│  Combina duas ideias:                                           │
│                                                                 │
│  1. MOMENTO (do SGD+Momentum)                                   │
│     → "Lembra" a direção que estava indo                        │
│     → Não muda de direção bruscamente                           │
│                                                                 │
│  2. TAXA ADAPTATIVA (do RMSprop)                                │
│     → Pesos que mudam pouco recebem lr maior                    │
│     → Pesos que mudam muito recebem lr menor                    │
│                                                                 │
│  Resultado: Converge rápido e estável                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### No código

```python
optimizer = optim.Adam(model.parameters(), lr=0.001)
# model.parameters() = todos os pesos do modelo
# lr=0.001 = learning rate inicial (Adam adapta depois)
```

---

## 16. Por que escolhemos MSELoss?

**Referência:** Linha 72 do código, Seção "MSELoss - Por que escolhemos?" do documento principal

### Tipos de problemas vs funções de perda

| Tipo de problema | Função de perda | Exemplo |
|------------------|-----------------|---------|
| **Regressão** (nosso caso) | MSELoss, MAELoss | Prever preço: R$ 27.50 |
| Classificação binária | BCELoss | Spam ou não spam |
| Classificação multi-classe | CrossEntropyLoss | Gato, cachorro, ou pássaro |

### Por que MSE para regressão?

```
┌─────────────────────────────────────────────────────────────────┐
│                MSE vs MAE para Regressão                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  MSE = Mean Squared Error = Média dos erros ao quadrado         │
│  MAE = Mean Absolute Error = Média dos erros absolutos          │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Erro │   MSE        │   MAE        │ Observação         │    │
│  │──────│──────────────│──────────────│────────────────────│    │
│  │ 0.1  │ 0.01         │ 0.1          │ MSE penaliza menos │    │
│  │ 1.0  │ 1.0          │ 1.0          │ Igual              │    │
│  │ 10.0 │ 100.0        │ 10.0         │ MSE penaliza MUITO │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
│  MSE penaliza mais erros grandes → modelo evita erros grosseiros│
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### No código

```python
criterion = nn.MSELoss()
loss = criterion(outputs, y_train)  # loss = (1/n) * Σ(pred - real)²
```

---

## 17. O que é Overfitting e Underfitting?

**Referência:** Linhas 224-228 do documento principal

### Definições

| Termo | Significado | Analogia |
|-------|-------------|----------|
| **Underfitting** | Modelo muito simples, não aprendeu | Estudou pouco, não sabe nada |
| **Bom ajuste** | Modelo equilibrado | Estudou bem, sabe aplicar |
| **Overfitting** | Modelo decorou os dados | Decorou o livro, não entende |

### Como detectar

```
┌─────────────────────────────────────────────────────────────────┐
│                    DIAGNÓSTICO VISUAL                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  UNDERFITTING              BOM AJUSTE            OVERFITTING    │
│                                                                 │
│  Loss│                     Loss│                 Loss│          │
│      │_____ train              │╲                    │╲         │
│      │───── val                │ ╲                   │ ╲ val    │
│      │                         │  ╲___               │  ╲___    │
│      │                         │   ╲__               │     ╲    │
│      └──────► época            └──────► época        │    train │
│                                                      └──────►   │
│                                                                 │
│  Train alto                 Ambos baixos         Train baixo    │
│  Val alto                   Gap pequeno          Val SOBE       │
│  Gap pequeno                                     Gap AUMENTA    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### No nosso resultado (bom!)

```
Train Loss final: 0.000699
Val Loss final:   0.002358
Razão: 3.37×

✅ Ambos diminuíram ao longo do treino
✅ Gap é aceitável para séries temporais
✅ Não há sinais de overfitting grave
```

### Como corrigir

| Problema | Soluções |
|----------|----------|
| **Underfitting** | Mais épocas, modelo maior, menos regularização |
| **Overfitting** | Early stopping, mais dados, mais dropout, menos épocas |

---

## 18. Como interpretar os valores de Loss?

**Referência:** Linhas 186-229 do documento principal

### Valores absolutos vs relativos

```
┌─────────────────────────────────────────────────────────────────┐
│              INTERPRETANDO VALORES DE LOSS                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ❌ ERRADO: "Meu loss é 0.001, isso é bom?"                     │
│     → Depende! Loss de 0.001 em dados de 0-1 é ótimo            │
│     → Loss de 0.001 em dados de 0-1000000 é péssimo             │
│                                                                 │
│  ✅ CORRETO: Compare com o início do treino                     │
│     → Loss inicial: 0.01 → Loss final: 0.001 = 10x melhor!      │
│                                                                 │
│  ✅ CORRETO: Compare train vs val loss                          │
│     → Se val loss é 3x train loss = aceitável                   │
│     → Se val loss é 100x train loss = overfitting               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### No nosso caso

| Métrica | Época 10 | Época 100 | Melhoria |
|---------|----------|-----------|----------|
| Train Loss | 0.002840 | 0.000699 | 4x melhor |
| Val Loss | 0.008114 | 0.002358 | 3.4x melhor |

### Convertendo loss para erro real

Como nossos dados estão normalizados (0-1), podemos estimar o erro em preço:

```
Loss = 0.002358
RMSE = √0.002358 ≈ 0.0486 (em escala normalizada)

Faixa de preço original: R$ 3.24 a R$ 27.38
Faixa normalizada: 0 a 1

Erro estimado em R$ ≈ 0.0486 × (27.38 - 3.24) ≈ R$ 1.17

O modelo erra, em média, cerca de R$ 1.17 nos dados de validação.
```

---

## 19. GPU vs CPU: qual usar?

**Referência:** Linhas 59-60 do código

### Detecção automática

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

### Comparação

| Aspecto | CPU | GPU (CUDA) |
|---------|-----|------------|
| **Disponibilidade** | Todo computador | Precisa de placa NVIDIA |
| **Velocidade (LSTM pequeno)** | 9 segundos | ~3 segundos |
| **Velocidade (LSTM grande)** | 10+ minutos | ~30 segundos |
| **Configuração** | Nenhuma | Instalar CUDA/cuDNN |
| **Custo** | Incluído | GPU boa custa caro |

### Para o nosso projeto

```
┌─────────────────────────────────────────────────────────────────┐
│               RECOMENDAÇÃO PARA ESTE PROJETO                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Dataset pequeno (~1400 amostras) + Modelo pequeno (2 camadas)  │
│                                                                 │
│  → CPU é suficiente! Treino completo em ~9 segundos             │
│  → GPU não vai fazer diferença significativa                    │
│  → Use GPU se tiver, mas não precisa comprar uma                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Como forçar CPU (se GPU der erro)

```python
model, train_losses, val_losses = train_model(
    model=model,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    device='cpu'  # ← Força CPU
)
```

---

## 20. Por que salvar o modelo em formato .pth?

**Referência:** Linhas 259-277 do documento principal

### O que é um arquivo .pth?

É o formato padrão do PyTorch para salvar modelos. Contém:

```python
torch.save({
    'model_state_dict': model.state_dict(),  # Os pesos treinados
    'model_config': {...},                   # Configuração da arquitetura
    'train_losses': [...],                   # Histórico de treino
    'val_losses': [...],                     # Histórico de validação
}, 'models/model_lstm.pth')
```

### Por que não salvar só os pesos?

| O que salvar | Vantagem | Desvantagem |
|--------------|----------|-------------|
| Só `state_dict` | Arquivo menor | Precisa lembrar a arquitetura |
| `state_dict` + config | Reprodutível | Arquivo maior |
| Modelo inteiro (`torch.save(model)`) | Mais simples | Pode quebrar entre versões |

### Como usar o modelo salvo

```python
# Carregar
checkpoint = torch.load('models/model_lstm.pth')

# Recriar modelo
model = StockLSTM(**checkpoint['model_config'])

# Carregar pesos treinados
model.load_state_dict(checkpoint['model_state_dict'])

# Modo de previsão
model.eval()

# Usar para prever
with torch.no_grad():
    previsao = model(novos_dados)
```

---

## 21. O modelo "aprende" sozinho?

**Resposta: Não exatamente.**

### O que o modelo realmente faz

```
┌─────────────────────────────────────────────────────────────────┐
│              O QUE O MODELO FAZ vs NÃO FAZ                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ❌ NÃO "pensa" como humano                                     │
│  ❌ NÃO "entende" o mercado de ações                            │
│  ❌ NÃO "sabe" que está prevendo preços                         │
│                                                                 │
│  ✅ FAZ ajustes matemáticos baseados em regras definidas        │
│  ✅ FAZ reconhecimento de padrões estatísticos                  │
│  ✅ FAZ otimização de uma função objetivo (minimizar loss)      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Analogia: Termostato vs Humano

```
Termostato: "Está frio, ligo o aquecedor. Está quente, desligo."
            → Segue regras fixas, não "sabe" o que é temperatura

Modelo LSTM: "Esse padrão de entrada dá esse padrão de saída."
             → Segue regras matemáticas, não "sabe" o que é preço
```

O "aprendizado" é só um ajuste iterativo de números para minimizar erros.

---

## 22. O treinamento é determinístico?

**Resposta: NÃO!**

### Fontes de aleatoriedade

| Fonte | O que faz | Impacto |
|-------|-----------|---------|
| **Pesos iniciais** | Inicializados aleatoriamente | Cada treino começa diferente |
| **Dropout** | Desliga neurônios aleatoriamente | Cada época é diferente |
| **Ordem dos dados** | Se usar shuffle (não no nosso caso) | Pode variar |

### Como tornar reprodutível

```python
import torch
import numpy as np
import random

# Fixar todas as sementes
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Para GPU (se usar)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
```

### Na prática

```
┌─────────────────────────────────────────────────────────────────┐
│             RESULTADOS PODEM VARIAR ENTRE EXECUÇÕES             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Execução 1: Val Loss final = 0.002358                          │
│  Execução 2: Val Loss final = 0.002412                          │
│  Execução 3: Val Loss final = 0.002301                          │
│                                                                 │
│  → Variação de ~5% é normal                                     │
│  → Se variar muito (>20%), algo pode estar errado               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 23. Posso mudar a arquitetura depois de treinar?

**Resposta: Sim, mas precisará treinar novamente.**

### O que pode mudar

| Mudança | Precisa re-treinar? | Por quê |
|---------|---------------------|---------|
| Hiperparâmetros (epochs, lr) | Sim | Afeta o processo de aprendizado |
| Arquitetura (hidden_size, layers) | Sim | Pesos antigos não se encaixam |
| Dados de entrada | Sim | Modelo aprendeu padrões diferentes |

### Por que os pesos não servem?

```
Modelo antigo: hidden_size=50 → 50 neurônios → X pesos
Modelo novo:   hidden_size=100 → 100 neurônios → 2X pesos

Os pesos extras não existem no modelo antigo!
```

### Transfer Learning (avançado)

Em alguns casos, você pode reutilizar PARTE dos pesos de um modelo treinado. Mas isso é mais avançado e não aplicamos neste projeto.

---

## 24. O que é Early Stopping?

**Referência:** Linhas 92-94 do código

### Definição

**Early Stopping** é uma técnica para **parar o treino automaticamente** quando o modelo começa a overfitar.

### Como funciona

```
┌─────────────────────────────────────────────────────────────────┐
│                     EARLY STOPPING                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Loss│                                                          │
│      │╲                                                         │
│      │ ╲                                                        │
│      │  ╲   Val Loss                                            │
│      │   ╲                                                      │
│      │    ╲________                                             │
│      │              ╲                                           │
│      │               ╲──────── ← Melhor ponto (salvar!)         │
│      │                    ╱                                     │
│      │                   ╱  ← Val Loss começa a SUBIR           │
│      │                  ╱                                       │
│      │                                                          │
│      └──────────────────────────────────────────────► Época     │
│               ↑                                                 │
│         PARAR AQUI! (antes de overfitar)                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### No nosso código (versão simplificada)

```python
# Monitoramos o melhor val_loss
best_val_loss = float('inf')
best_epoch = 0

for epoch in range(epochs):
    # ... treino ...
    
    if val_loss.item() < best_val_loss:
        best_val_loss = val_loss.item()
        best_epoch = epoch + 1
        # Aqui poderíamos salvar o melhor modelo

# No final, reportamos qual foi a melhor época
print(f"Melhor Val Loss: {best_val_loss:.6f} (época {best_epoch})")
```

### Implementação completa de Early Stopping (opcional)

```python
patience = 10  # Quantas épocas esperar antes de parar
counter = 0

for epoch in range(epochs):
    # ... treino ...
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping na época {epoch}")
            break
```

---

## 25. Outras dúvidas comuns

### Sobre performance

| Dúvida | Resposta |
|--------|----------|
| **"Quanto tempo demora o treino?"** | Depende do hardware e dados. No nosso caso, ~9 segundos para 100 épocas. |
| **"Posso treinar enquanto uso o computador?"** | Sim, mas pode ficar lento se usar muita CPU/GPU. |
| **"O treino pode ser interrompido?"** | Sim, mas você perde o progresso a menos que salve checkpoints. |

### Sobre os dados

| Dúvida | Resposta |
|--------|----------|
| **"Por que dividir em treino e validação?"** | Para saber se o modelo generaliza ou só decorou. |
| **"Posso usar 100% dos dados pra treinar?"** | Tecnicamente sim, mas não saberá se está overfitando. |
| **"E se meus dados tiverem erros?"** | O modelo vai aprender os erros também! Limpe antes. |

### Sobre o modelo

| Dúvida | Resposta |
|--------|----------|
| **"Por que LSTM e não transformer?"** | LSTMs são mais simples e suficientes para séries temporais curtas. Transformers são melhores para sequências longas. |
| **"Quantos pesos o modelo tem?"** | `sum(p.numel() for p in model.parameters())` - No nosso caso, ~31.000 pesos. |
| **"O modelo pode prever qualquer ação?"** | Ele foi treinado em PETR4. Para outras ações, precisaria re-treinar. |

### Sobre a saída do modelo

| Dúvida | Resposta |
|--------|----------|
| **"A previsão é garantida?"** | Não! É uma estimativa baseada em padrões passados. |
| **"Posso usar para investir?"** | Use com muito cuidado. Modelos não preveem eventos inesperados (COVID, guerras, etc). |
| **"Por que a previsão está normalizada?"** | Porque treinamos com dados normalizados. Na Etapa 6 revertemos para R$. |

---

## 🔗 Navegação

| Anterior | Próximo |
|----------|---------|
| [ETAPA 04 - Modelo LSTM](./ETAPA_04_Modelo_LSTM.md) | [ETAPA 06 - Avaliação](./ETAPA_06_Avaliacao.md) |

---

*Documento criado para esclarecer dúvidas comuns sobre a Etapa 5 do projeto LSTM.*
