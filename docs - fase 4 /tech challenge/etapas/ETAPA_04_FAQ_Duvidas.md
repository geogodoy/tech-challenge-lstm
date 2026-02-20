# ❓ FAQ - Dúvidas da Etapa 4: Modelo LSTM

> Documento complementar à [ETAPA_04_Modelo_LSTM.md](./ETAPA_04_Modelo_LSTM.md)

---

## 📚 Índice

1. [O que são Hiperparâmetros e qual sua finalidade?](#1-o-que-são-hiperparâmetros-e-qual-sua-finalidade)
2. [O que é Dropout e Regularização?](#2-o-que-é-dropout-e-regularização)
3. [O que é X Shape no contexto do modelo?](#3-o-que-é-x-shape-no-contexto-do-modelo)
4. [Como funciona o Treinamento de uma Rede Neural?](#4-como-funciona-o-treinamento-de-uma-rede-neural)
5. [Quais ações de Tuning posso fazer durante o treinamento?](#5-quais-ações-de-tuning-posso-fazer-durante-o-treinamento)
6. [O que é nn.Module e por que herdar dele?](#6-o-que-é-nnmodule-e-por-que-herdar-dele)
7. [O que significa batch_first=True?](#7-o-que-significa-batch_firsttrue)
8. [Por que hidden_size=50 e não outro valor?](#8-por-que-hidden_size50-e-não-outro-valor)
9. [O que são h_n e c_n retornados pela LSTM?](#9-o-que-são-h_n-e-c_n-retornados-pela-lstm)
10. [Por que pegar apenas lstm_out[:, -1, :]?](#10-por-que-pegar-apenas-lstm_out-1-)
11. [O que é a camada Linear e por que ela existe?](#11-o-que-é-a-camada-linear-e-por-que-ela-existe)
12. [O que significa "31.051 parâmetros treináveis"?](#12-o-que-significa-31051-parâmetros-treináveis)
13. [Qual a diferença entre Parâmetros e Hiperparâmetros?](#13-qual-a-diferença-entre-parâmetros-e-hiperparâmetros)
14. [O que é Forward Pass?](#14-o-que-é-forward-pass)
15. [Por que LSTM e não uma RNN comum?](#15-por-que-lstm-e-não-uma-rnn-comum)

---

## 1. O que são Hiperparâmetros e qual sua finalidade?

**Referência:** Linhas 123-134 do documento principal

### Definição Simples

> **Hiperparâmetros** são configurações que **você define ANTES** do treinamento e que controlam **como** o modelo vai aprender.

Pense neles como os "ajustes do carro" antes de uma corrida: você escolhe a pressão dos pneus, a altura da suspensão, o tipo de combustível - tudo ANTES de começar a correr.

### São baseados em Regra de Negócio + Arquitetura?

**Sim!** A escolha de hiperparâmetros depende de dois fatores:

```
┌─────────────────────────────────────────────────────────────┐
│           COMO ESCOLHER HIPERPARÂMETROS                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  REGRA DE NEGÓCIO (seu problema específico):               │
│  ├─ Quantidade de dados disponíveis                        │
│  ├─ Complexidade do padrão a aprender                      │
│  ├─ Tolerância a erro                                      │
│  └─ Tempo disponível para treinar                          │
│                                                             │
│  +                                                          │
│                                                             │
│  ARQUITETURA (características da LSTM):                    │
│  ├─ LSTMs precisam de mais hidden_size para sequências     │
│  │   longas                                                 │
│  ├─ num_layers > 1 para padrões mais complexos             │
│  └─ dropout necessário para evitar overfitting             │
│                                                             │
│  =                                                          │
│                                                             │
│  HIPERPARÂMETROS ESCOLHIDOS                                │
│  (hidden_size=50, num_layers=2, dropout=0.2)               │
└─────────────────────────────────────────────────────────────┘
```

### Tabela de Hiperparâmetros do Projeto

| Hiperparâmetro | Valor | Justificativa de Negócio | Justificativa Técnica |
|----------------|-------|--------------------------|----------------------|
| `input_size` | 1 | Usando apenas preço Close | Feature única simplifica |
| `hidden_size` | 50 | Preços têm padrões moderados | Capacidade de memória da LSTM |
| `num_layers` | 2 | Queremos capturar tendências | Deep LSTM para padrões hierárquicos |
| `dropout` | 0.2 | Dados financeiros são ruidosos | Evitar decorar ruído |
| `seq_length` | 60 | 60 dias úteis ≈ 3 meses | Janela temporal razoável |

### Analogia

Hiperparâmetros são como a **receita de um bolo**:
- Você define os ingredientes e quantidades ANTES de fazer o bolo
- O bolo "aprende" a forma final no forno (treinamento)
- Se o bolo não ficou bom, você ajusta a receita e faz outro

---

## 2. O que é Dropout e Regularização?

**Referência:** Linhas 135-146 do documento principal

### O que é Regularização?

**Regularização** = técnicas para evitar **overfitting** (quando o modelo "decora" os dados ao invés de aprender padrões).

### O que é Dropout?

**Dropout** é uma técnica de regularização que **desliga neurônios aleatoriamente** durante o treinamento.

### Analogia: A Sala de Aula

Imagine uma sala de aula com 10 alunos:
- **SEM Dropout**: A professora sempre pergunta para os mesmos 2-3 alunos "gênios". Os outros não aprendem.
- **COM Dropout**: A professora ALEATORIAMENTE escolhe quais alunos vão responder. Todos precisam aprender!

### Como funciona tecnicamente

```
DURANTE O TREINAMENTO (dropout=0.2 = 20%)
────────────────────────────────────────

Neurônios ANTES do dropout:
[N1] [N2] [N3] [N4] [N5] [N6] [N7] [N8] [N9] [N10]
  ●    ●    ●    ●    ●    ●    ●    ●    ●    ●

Neurônios DURANTE o dropout (20% desligados aleatoriamente):
[N1] [N2] [N3] [N4] [N5] [N6] [N7] [N8] [N9] [N10]
  ●    ✖    ●    ●    ✖    ●    ●    ●    ●    ●
       ↑              ↑
    desligados aleatoriamente

Na PRÓXIMA iteração, OUTROS 20% são desligados:
[N1] [N2] [N3] [N4] [N5] [N6] [N7] [N8] [N9] [N10]
  ●    ●    ●    ✖    ●    ●    ✖    ●    ●    ●

DURANTE INFERÊNCIA/TESTE:
Todos os neurônios são usados (100%) - sem dropout!
```

### No código do projeto

```python
# Dropout ENTRE as camadas LSTM (dentro do nn.LSTM)
self.lstm = nn.LSTM(..., dropout=0.2)  # desliga conexões entre camadas

# Dropout APÓS a LSTM (antes da previsão final)
self.dropout = nn.Dropout(0.2)  # desliga neurônios antes da saída
```

### Por que isso funciona?

| Problema | Sintoma | Como Dropout ajuda |
|----------|---------|-------------------|
| **Overfitting** | Modelo acerta treino mas erra teste | Força a rede a não depender de neurônios específicos |
| **Co-adaptação** | Neurônios "combinam" demais entre si | Quebra a dependência entre neurônios |

### Valores comuns de Dropout

| Valor | Quando usar |
|-------|-------------|
| 0.1 - 0.2 | Poucos dados ou modelo pequeno |
| 0.2 - 0.3 | Caso padrão (como nosso projeto) |
| 0.4 - 0.5 | Muitos dados, modelo grande, muito overfitting |
| > 0.5 | Raramente usado (pode prejudicar aprendizado) |

---

## 3. O que é X Shape no contexto do modelo?

**Referência:** Linhas 83-84, 154-155 do documento principal

### Definição

> **X shape** (formato de X) descreve as **dimensões** do tensor de entrada que o modelo espera receber.

### No modelo LSTM do projeto

```python
x shape: (batch_size, seq_length, input_size)
Exemplo: (32, 60, 1)
```

### O que cada dimensão significa

```
┌─────────────────────────────────────────────────────────────┐
│                     X SHAPE: (32, 60, 1)                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  batch_size = 32                                           │
│  └─ Quantas amostras são processadas DE UMA VEZ           │
│     (32 "janelas" de 60 dias cada)                         │
│                                                             │
│  seq_length = 60                                           │
│  └─ Quantos "passos de tempo" cada amostra tem            │
│     (60 dias de histórico)                                 │
│                                                             │
│  input_size = 1                                            │
│  └─ Quantas features por passo de tempo                   │
│     (1 = apenas preço de fechamento)                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Visualização prática

```
Uma ÚNICA amostra (shape: 1, 60, 1):
┌────────────────────────────────────────────────────────┐
│ Dia 1 │ Dia 2 │ Dia 3 │ ... │ Dia 59 │ Dia 60 │       │
│ $100  │ $102  │ $101  │ ... │ $115   │ $118   │→ $??? │
└────────────────────────────────────────────────────────┘
  └──────────────── 60 valores ────────────────┘   ↑
                                            Previsão dia 61

Um BATCH de 32 amostras (shape: 32, 60, 1):
┌───────────────────────────────────────────────────────┐
│ Amostra 1:  [Dia1, Dia2, ..., Dia60] → Previsão 1    │
│ Amostra 2:  [Dia1, Dia2, ..., Dia60] → Previsão 2    │
│ Amostra 3:  [Dia1, Dia2, ..., Dia60] → Previsão 3    │
│ ...                                                   │
│ Amostra 32: [Dia1, Dia2, ..., Dia60] → Previsão 32   │
└───────────────────────────────────────────────────────┘
```

### Por que processar em batches?

| Razão | Explicação |
|-------|------------|
| **Eficiência** | GPU processa 32 amostras quase tão rápido quanto 1 |
| **Estabilidade** | Média de 32 gradientes é mais estável que 1 |
| **Memória** | Limita uso de RAM/VRAM |

### Resumo visual

```
x = torch.randn(32, 60, 1)
                │   │   │
                │   │   └─ 1 feature (preço Close)
                │   └───── 60 dias de histórico
                └───────── 32 amostras no batch
```

---

## 4. Como funciona o Treinamento de uma Rede Neural?

**Referência:** Mencionado nas aulas e próxima etapa (ETAPA 5)

### Analogia: Aprendendo a jogar dardos

Pense no treinamento como **ensinar uma criança a jogar dardos**:

1. Criança joga o dardo (faz previsão)
2. Vê onde acertou vs onde deveria (calcula erro)
3. Entende o que fez errado (backpropagation)
4. Ajusta a mira (atualiza pesos)
5. Repete até ficar bom!

### O Ciclo de Treinamento

```
┌─────────────────────────────────────────────────────────────┐
│              CICLO DE TREINAMENTO (1 iteração)              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1️⃣ FORWARD PASS - "Jogar o dardo"                         │
│     ┌─────────┐                                            │
│     │ Entrada │ → [MODELO] → Previsão: $120                │
│     │ 60 dias │              Realidade: $115               │
│     └─────────┘                                            │
│                                                             │
│  2️⃣ CALCULAR ERRO (Loss) - "Medir distância do alvo"      │
│     Loss = (120 - 115)² = 25                               │
│     (Errou por $5, loss = 25)                              │
│                                                             │
│  3️⃣ BACKWARD PASS - "Entender o que fez errar"            │
│     Backpropagation: calcula quanto CADA peso              │
│     contribuiu para o erro (gradientes)                    │
│                                                             │
│  4️⃣ ATUALIZAR PESOS - "Ajustar a mira"                    │
│     Otimizador (Adam): ajusta os pesos para                │
│     errar MENOS na próxima vez                             │
│                                                             │
│  5️⃣ REPETIR com próximo batch de dados                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Código simplificado

```python
for epoch in range(100):  # Repetir 100 vezes todo o dataset
    for batch in data_loader:  # Para cada grupo de 32 amostras
        
        # 1️⃣ Forward: modelo faz previsão
        previsao = modelo(batch_entrada)
        
        # 2️⃣ Calcula erro (loss)
        erro = funcao_perda(previsao, valor_real)
        
        # 3️⃣ Backward: calcula gradientes
        erro.backward()
        
        # 4️⃣ Atualiza pesos
        otimizador.step()
        
        # Limpa gradientes para próxima iteração
        otimizador.zero_grad()
```

### Termos importantes

| Termo | O que é | Analogia |
|-------|---------|----------|
| **Época (Epoch)** | Uma passada completa por todos os dados | Uma rodada completa de treino |
| **Batch** | Grupo de amostras processadas juntas | Jogar 32 dardos de uma vez |
| **Loss** | Medida do erro da previsão | Distância do dardo ao alvo |
| **Gradiente** | Direção para melhorar | "Mire mais para a esquerda" |
| **Learning Rate** | Tamanho do ajuste | Quanto ajustar a mira |

### Visualização do aprendizado

```
Loss
  │
  │ ●
  │  ●
  │   ●●
  │     ●●●
  │        ●●●●●
  │             ●●●●●●●●●●●●●
  └────────────────────────────────→ Épocas
    1   10   20   30   40   50

A loss DEVE diminuir ao longo das épocas!
```

---

## 5. Quais ações de Tuning posso fazer durante o treinamento?

**Referência:** Próxima etapa (ETAPA 5)

### Ações DURANTE o treinamento (sem reiniciar)

| Ação | O que fazer | Quando aplicar |
|------|-------------|----------------|
| **Early Stopping** | Parar quando `val_loss` para de melhorar | Quando val_loss começa a subir |
| **Learning Rate Scheduling** | Reduzir LR gradualmente | Quando a loss "empaca" |
| **Checkpointing** | Salvar modelo nos melhores momentos | Sempre - guarda melhor versão |
| **Monitoramento** | Observar train_loss vs val_loss | Durante todo o treinamento |

### Ações que EXIGEM reiniciar o treinamento

| Ação | O que mudar | Impacto |
|------|-------------|---------|
| **Alterar hiperparâmetros** | `hidden_size`, `num_layers`, `dropout` | Treinar do zero |
| **Mudar arquitetura** | Adicionar/remover camadas | Treinar do zero |
| **Ajustar batch_size** | Tamanho do lote | Reiniciar |
| **Mudar otimizador** | Adam → SGD, RMSprop | Reiniciar |

### Fluxo prático de tuning

```
1. Treina modelo inicial
       ↓
2. Analisa curvas de loss (train_loss vs val_loss)
       ↓
3. DIAGNÓSTICO:
   ┌─────────────────────────────────────────────────────┐
   │ val_loss >> train_loss → OVERFITTING               │
   │   → Aumentar dropout, reduzir modelo               │
   ├─────────────────────────────────────────────────────┤
   │ Ambos altos → UNDERFITTING                         │
   │   → Aumentar modelo, mais épocas                   │
   ├─────────────────────────────────────────────────────┤
   │ val_loss oscila muito → LEARNING RATE alto         │
   │   → Reduzir LR                                     │
   └─────────────────────────────────────────────────────┘
       ↓
4. Ajusta e treina novamente
       ↓
5. Repete até satisfatório
```

### Diagnóstico pelas curvas de loss

```
CENÁRIO 1: OVERFITTING (decoreba) ❌
─────────────────────────────────
Loss │     train_loss (desce e fica baixo)
     │ ╲
     │  ╲_______________
     │
     │        val_loss (desce mas depois SOBE)
     │       ╱╲
     │      ╱  ╲____╱╲___
     └──────────────────────────→ Épocas

SOLUÇÃO: ↑ dropout, ↓ hidden_size, ↓ num_layers, early stopping


CENÁRIO 2: UNDERFITTING (não aprende) ❌
─────────────────────────────────────
Loss │ ─────────────── train_loss (alto, não desce)
     │ ─────────────── val_loss (alto também)
     │
     └──────────────────────────→ Épocas

SOLUÇÃO: ↑ hidden_size, ↑ num_layers, ↑ épocas, ↓ learning_rate


CENÁRIO 3: BOM TREINAMENTO (ideal) ✅
──────────────────────────────────
Loss │
     │ ╲
     │  ╲  train_loss
     │   ╲_______________
     │    ╲ val_loss
     │     ╲______________
     └──────────────────────────→ Épocas

Ambas descem juntas e estabilizam próximas!
```

### Ordem sugerida para tuning

```
1. Learning Rate (maior impacto)
   └─ Teste: 0.01, 0.001, 0.0001

2. Hidden Size (capacidade do modelo)
   └─ Teste: 32, 50, 100, 128

3. Número de Layers
   └─ Teste: 1, 2, 3

4. Dropout (regularização)
   └─ Teste: 0.1, 0.2, 0.3, 0.5

5. Batch Size (estabilidade)
   └─ Teste: 16, 32, 64
```

---

## 6. O que é nn.Module e por que herdar dele?

**Referência:** Linha 53 do documento principal

### O que é nn.Module?

`nn.Module` é a **classe base do PyTorch** para criar redes neurais. É como um "molde" que já vem com funcionalidades prontas.

### Por que herdar de nn.Module?

```python
class StockLSTM(nn.Module):  # ← Herda de nn.Module
    def __init__(self, ...):
        super(StockLSTM, self).__init__()  # ← Inicializa a classe pai
```

| O que você ganha | Explicação |
|-----------------|------------|
| **Gerenciamento de parâmetros** | Rastreia automaticamente todos os pesos |
| **Método `.to(device)`** | Facilita mover modelo para GPU |
| **Método `.train()/.eval()`** | Alterna entre modo treino/teste |
| **Serialização** | Salvar/carregar modelo facilmente |
| **Gradientes automáticos** | Backpropagation funciona "magicamente" |

### Analogia

É como herdar de uma receita básica de bolo:
- `nn.Module` = receita base (já tem forno, forma, etc.)
- `StockLSTM` = sua versão customizada (adiciona sabor, cobertura)

Você não precisa reinventar a roda - só customizar o que precisa!

---

## 7. O que significa batch_first=True?

**Referência:** Linha 63 do documento principal

### O problema

PyTorch LSTM pode receber dados em duas ordens diferentes:

```python
# batch_first=False (padrão do PyTorch)
x.shape = (seq_length, batch_size, input_size)
Exemplo:  (60, 32, 1)

# batch_first=True (mais intuitivo)
x.shape = (batch_size, seq_length, input_size)
Exemplo:  (32, 60, 1)
```

### Por que usamos batch_first=True?

| Motivo | Explicação |
|--------|------------|
| **Mais intuitivo** | "32 amostras de 60 dias cada" faz mais sentido |
| **Compatibilidade** | Outros frameworks (TensorFlow, etc.) usam batch primeiro |
| **DataLoader** | O DataLoader do PyTorch retorna batch na primeira dimensão |

### Visualização

```
batch_first=True (nosso caso):
┌─────────────────────────────────────┐
│ Amostra 1: [dia1, dia2, ..., dia60] │ ← batch_size é a PRIMEIRA dimensão
│ Amostra 2: [dia1, dia2, ..., dia60] │
│ ...                                 │
│ Amostra 32: [dia1, dia2, ..., dia60]│
└─────────────────────────────────────┘

batch_first=False (padrão PyTorch):
┌─────────────────────────────────────────────────────────┐
│ Dia 1:  [amostra1, amostra2, ..., amostra32]           │
│ Dia 2:  [amostra1, amostra2, ..., amostra32]           │ ← seq_length primeiro
│ ...                                                     │
│ Dia 60: [amostra1, amostra2, ..., amostra32]           │
└─────────────────────────────────────────────────────────┘
```

---

## 8. Por que hidden_size=50 e não outro valor?

**Referência:** Linha 130 do documento principal

### O que é hidden_size?

É a **dimensão do vetor de memória** da LSTM - quantas "células de memória" ela tem para guardar informações.

### Analogia

Pense em hidden_size como o **tamanho do cérebro** da rede:
- Muito pequeno (10): Não consegue lembrar padrões complexos
- Muito grande (500): Demora para treinar, pode decorar (overfitting)
- Adequado (50): Equilíbrio entre capacidade e eficiência

### Por que 50 especificamente?

| Fator | Análise | Impacto na escolha |
|-------|---------|-------------------|
| **Tamanho dos dados** | ~1.500 amostras | Modelo não pode ser muito grande |
| **Complexidade** | Preços são moderadamente complexos | Não precisa de 500 neurônios |
| **Tempo de treino** | Queremos treinar rápido | 50 é eficiente |
| **Referência** | Aula usou 128 para problema mais complexo | 50 é proporcional |

### Valores comuns

| hidden_size | Quando usar |
|-------------|-------------|
| 16-32 | Problemas simples, poucos dados |
| 50-100 | Problemas médios (nosso caso) |
| 128-256 | Problemas complexos, muitos dados |
| 512+ | NLP, problemas muito complexos |

### Se não estiver funcionando bem?

É um dos primeiros hiperparâmetros a ajustar no tuning:
- Performance ruim? Tente `hidden_size=100`
- Overfitting? Tente `hidden_size=32`

---

## 9. O que são h_n e c_n retornados pela LSTM?

**Referência:** Linhas 158-161 do documento principal

### Contexto

Quando passamos dados pela LSTM, ela retorna 3 coisas:

```python
lstm_out, (h_n, c_n) = self.lstm(x)
```

### O que cada um significa

| Retorno | Nome | Shape | O que é |
|---------|------|-------|---------|
| `lstm_out` | Output | (32, 60, 50) | Saída de CADA passo temporal |
| `h_n` | Hidden State | (2, 32, 50) | Estado oculto FINAL (memória de curto prazo) |
| `c_n` | Cell State | (2, 32, 50) | Estado da célula FINAL (memória de longo prazo) |

### Visualização

```
                    LSTM processando 60 dias
    ┌─────┐   ┌─────┐   ┌─────┐         ┌─────┐
x → │Dia 1│ → │Dia 2│ → │Dia 3│ → ... → │Dia60│ → h_n (estado final)
    └──┬──┘   └──┬──┘   └──┬──┘         └──┬──┘
       ↓         ↓         ↓               ↓
     out[0]    out[1]    out[2]         out[59]
    
    └──────────────── lstm_out ────────────────┘
```

### Por que existem dois estados (h_n e c_n)?

É o que torna LSTM especial! É o "segredo" de como ela lembra coisas por muito tempo:

| Estado | Função | Analogia |
|--------|--------|----------|
| `h_n` (hidden) | Memória de trabalho | O que você está pensando AGORA |
| `c_n` (cell) | Memória de longo prazo | O que você aprendeu e GUARDA |

### No nosso código, usamos qual?

Usamos `lstm_out[:, -1, :]` que é equivalente a `h_n[-1]` da última camada. Ambos contêm a "memória" após processar toda a sequência.

---

## 10. Por que pegar apenas lstm_out[:, -1, :]?

**Referência:** Linhas 163-165, 177-180 do documento principal

### O problema

A LSTM retorna uma saída para CADA dia dos 60:

```python
lstm_out.shape = (32, 60, 50)
#                 │   │   └─ 50 features de saída
#                 │   └───── 60 dias
#                 └───────── 32 amostras

# Mas queremos apenas UMA previsão por amostra!
```

### A solução: pegar o último passo

```python
last_output = lstm_out[:, -1, :]
# Shape: (32, 50)
#         │   └─ 50 features
#         └───── 32 amostras (uma por amostra!)
```

### O que significa `[:, -1, :]`?

```python
lstm_out[:, -1, :]
         │   │   │
         │   │   └─ : = todas as features (0 a 49)
         │   └───── -1 = apenas o ÚLTIMO dia (dia 60)
         └───────── : = todas as amostras (0 a 31)
```

### Por que o ÚLTIMO dia?

O último hidden state contém a **informação acumulada** de todos os 60 dias anteriores!

```
Dia 1: LSTM vê preço do dia 1
       ↓
Dia 2: LSTM vê dia 2 + LEMBRA do dia 1
       ↓
Dia 3: LSTM vê dia 3 + LEMBRA dos dias 1-2
       ↓
       ...
       ↓
Dia 60: LSTM vê dia 60 + LEMBRA dos dias 1-59 ← ESTE USAMOS!
        └─ Contém o "resumo" de toda a história
```

### Analogia

É como ler um livro de 60 páginas:
- `lstm_out[:, 0, :]` = Sua opinião após ler só a página 1
- `lstm_out[:, 30, :]` = Sua opinião após ler até a página 30
- `lstm_out[:, -1, :]` = Sua opinião após ler o livro INTEIRO ← Mais completa!

---

## 11. O que é a camada Linear e por que ela existe?

**Referência:** Linha 71 do documento principal

### O que é?

```python
self.linear = nn.Linear(hidden_size, 1)  # 50 → 1
```

É uma camada que transforma 50 números em 1 número (o preço previsto).

### Por que é necessária?

A LSTM retorna um vetor de 50 valores (hidden_size), mas queremos **1 único número** (o preço):

```
LSTM output: [0.23, -0.15, 0.89, ..., 0.45]  ← 50 números
                           │
                     nn.Linear(50, 1)
                           │
                           ▼
Previsão:                $118.50              ← 1 número
```

### Como funciona matematicamente?

```
preço = w₁×v₁ + w₂×v₂ + ... + w₅₀×v₅₀ + bias

Onde:
- v₁...v₅₀ = saída da LSTM (50 valores)
- w₁...w₅₀ = pesos aprendidos (50 pesos)
- bias = termo de ajuste (1 valor)

Total de parâmetros: 50 + 1 = 51
```

### Analogia

A camada Linear é como um **tradutor**:
- LSTM fala em "linguagem interna" (50 dimensões)
- Linear traduz para "linguagem humana" (1 preço)

---

## 12. O que significa "31.051 parâmetros treináveis"?

**Referência:** Linhas 193-205 do documento principal

### O que são parâmetros?

**Parâmetros** são os "pesos" e "bias" que o modelo **aprende** durante o treinamento.

### De onde vêm os 31.051?

```
┌─────────────────────────────────────────────────────────────┐
│              CONTAGEM DE PARÂMETROS                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  CAMADA LSTM (2 layers):                                   │
│  ├─ Layer 1: 4 × (1×50 + 50×50 + 50 + 50) = 10.600        │
│  │           └─ input→hidden + hidden→hidden + biases      │
│  │                                                         │
│  └─ Layer 2: 4 × (50×50 + 50×50 + 50 + 50) = 20.400       │
│              └─ hidden→hidden + hidden→hidden + biases     │
│                                                             │
│  Subtotal LSTM: ~31.000                                    │
│                                                             │
│  CAMADA LINEAR:                                            │
│  └─ 50 × 1 + 1 (bias) = 51                                │
│                                                             │
│  TOTAL: ~31.051 parâmetros                                 │
└─────────────────────────────────────────────────────────────┘
```

### Por que "4×" na LSTM?

A LSTM tem 4 "portões" (gates) que controlam o fluxo de informação:
1. **Forget gate** - O que esquecer
2. **Input gate** - O que adicionar
3. **Cell gate** - Candidato a nova memória
4. **Output gate** - O que expor como saída

Cada gate tem seus próprios pesos, por isso multiplicamos por 4.

### Isso é muito ou pouco?

| Modelo | Parâmetros | Classificação |
|--------|-----------|---------------|
| **StockLSTM** | 31K | Pequeno/Médio |
| GPT-2 | 124M | Grande |
| GPT-3 | 175B | Muito Grande |
| GPT-4 | ~1T+ | Gigante |

Para nosso problema de previsão de ações, 31K é adequado!

---

## 13. Qual a diferença entre Parâmetros e Hiperparâmetros?

**Referência:** Linhas 123-134 do documento principal

### Tabela comparativa

| Aspecto | Parâmetros | Hiperparâmetros |
|---------|-----------|-----------------|
| **Quem define** | O modelo aprende | Você define |
| **Quando** | Durante o treinamento | Antes de treinar |
| **Exemplo** | Pesos das conexões | `hidden_size=50` |
| **Quantidade** | 31.051 no nosso modelo | ~5-10 principais |
| **Como ajustar** | Backpropagation (automático) | Experimentação (manual) |

### Visualização

```
ANTES DO TREINAMENTO
────────────────────
Você define HIPERPARÂMETROS:
├─ hidden_size = 50
├─ num_layers = 2
├─ dropout = 0.2
└─ learning_rate = 0.001

DURANTE O TREINAMENTO
─────────────────────
Modelo aprende PARÂMETROS:
├─ peso_1 = 0.234  (era 0.001)
├─ peso_2 = -0.156 (era 0.002)
├─ ...
└─ peso_31051 = 0.089 (era -0.001)
```

### Analogia: Aprendendo a andar de bicicleta

- **Hiperparâmetros** = Configurações da bicicleta (altura do banco, pressão do pneu)
- **Parâmetros** = Seu equilíbrio e coordenação (aprendido com prática)

---

## 14. O que é Forward Pass?

**Referência:** Linhas 150-175 do documento principal

### Definição

**Forward Pass** (passagem direta) é quando os dados **entram** no modelo e **saem** como previsão.

### Visualização

```
FORWARD PASS
────────────

Entrada (x)              Processamento               Saída (previsão)
    │                          │                          │
    ▼                          ▼                          ▼

[60 dias de       →      [LSTM + Dropout      →      [Preço previsto
 preços]                  + Linear]                   para dia 61]

(32, 60, 1)                                          (32, 1)
```

### No código

```python
def forward(self, x):
    # x entra: (32, 60, 1)
    
    lstm_out, _ = self.lstm(x)      # Passa pela LSTM
    last = lstm_out[:, -1, :]       # Pega último estado
    out = self.dropout(last)        # Aplica dropout
    prediction = self.linear(out)   # Transforma em preço
    
    # prediction sai: (32, 1)
    return prediction
```

### Forward vs Backward

| Direção | O que acontece | Quando |
|---------|---------------|--------|
| **Forward** | Dados → Modelo → Previsão | Sempre |
| **Backward** | Gradientes calculados do erro para os pesos | Só no treino |

---

## 15. Por que LSTM e não uma RNN comum?

**Referência:** Linhas 24-32 do documento principal

### O problema das RNNs comuns

RNNs tradicionais sofrem de dois problemas graves:

```
PROBLEMA: Vanishing Gradient (Gradiente Desvanecente)
─────────────────────────────────────────────────────
Informação de dias antigos "some" ao longo da sequência:

Dia 1 → Dia 2 → Dia 3 → ... → Dia 58 → Dia 59 → Dia 60
  ●       ●       ●              ○       ○       ○
 100%    80%     60%            2%      1%      0.5%
  └────── Informação vai "desaparecendo" ──────────┘

Resultado: RNN "esquece" o que viu nos primeiros dias!
```

### Como LSTM resolve isso

A LSTM tem **portões (gates)** que controlam o que lembrar/esquecer:

```
┌─────────────────────────────────────────────────────────────┐
│                    CÉLULA LSTM                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐                │
│  │ FORGET   │   │  INPUT   │   │  OUTPUT  │                │
│  │  GATE    │   │   GATE   │   │   GATE   │                │
│  │          │   │          │   │          │                │
│  │ "O que   │   │ "O que   │   │ "O que   │                │
│  │ esquecer"│   │ guardar" │   │ mostrar" │                │
│  └────┬─────┘   └────┬─────┘   └────┬─────┘                │
│       │              │              │                       │
│       └──────────────┼──────────────┘                       │
│                      ▼                                      │
│              ┌───────────────┐                             │
│              │  CELL STATE   │                             │
│              │ (Memória de   │                             │
│              │ longo prazo)  │                             │
│              └───────────────┘                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Comparação

| Aspecto | RNN Comum | LSTM |
|---------|-----------|------|
| Memória de longo prazo | ❌ Fraca | ✅ Forte |
| Vanishing gradient | ❌ Sofre muito | ✅ Minimizado |
| Complexidade | Simples | Mais complexa |
| Parâmetros | Menos | ~4x mais |
| Performance em séries longas | ❌ Ruim | ✅ Boa |

### Por que isso importa para ações?

Preços de ações têm **dependências de longo prazo**:
- Um evento em janeiro pode afetar preços em dezembro
- Padrões sazonais se repetem ao longo de meses
- RNN comum "esqueceria" - LSTM lembra!

---

## 🔗 Navegação

| Anterior | Próximo |
|----------|---------|
| [ETAPA 03 - FAQ](./ETAPA_03_FAQ_Duvidas.md) | [ETAPA 05 - Treinamento](./ETAPA_05_Treinamento.md) |

---

*Documento criado para esclarecer dúvidas comuns sobre a Etapa 4 do projeto LSTM - Definição da Arquitetura do Modelo.*
