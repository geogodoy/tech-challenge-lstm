# ❓ FAQ - Dúvidas da Etapa 3: Pré-processamento

> Documento complementar à [ETAPA_03_Preprocessamento.md](./ETAPA_03_Preprocessamento.md)

---

## 📚 Índice

1. [O que ocorre em cada fase do processamento?](#1-o-que-ocorre-em-cada-fase-do-processamento)
2. [O que significa normalizar entre 0 e 1?](#2-o-que-significa-normalizar-entre-0-e-1)
3. [O que é Exploding Gradients?](#3-o-que-é-exploding-gradients)
4. [O que são Janelas Deslizantes?](#4-o-que-são-janelas-deslizantes)
5. [Quais são os tipos de normalização além do MinMaxScaler?](#5-quais-são-os-tipos-de-normalização-além-do-minmaxscaler)
6. [Onde e como são salvos os dados processados?](#6-onde-e-como-são-salvos-os-dados-processados)
7. [O que significa Inferência na prática?](#7-o-que-significa-inferência-na-prática)
8. [Janela Temporal, Sequência Temporal e Janela Deslizante são a mesma coisa?](#8-janela-temporal-sequência-temporal-e-janela-deslizante-são-a-mesma-coisa)
9. [O que são Tensores? (Explicação para leigos)](#9-o-que-são-tensores-explicação-para-leigos)
10. [Quais são os fundamentos de Deep Learning?](#10-quais-são-os-fundamentos-de-deep-learning)
11. [O que é a coluna Close?](#11-o-que-é-a-coluna-close)
12. [Como interpretar os shapes dos dados?](#12-como-interpretar-os-shapes-dos-dados)
13. [Por que não embaralhamos os dados?](#13-por-que-não-embaralhamos-os-dados)
14. [O que é o Scaler e por que precisa ser salvo?](#14-o-que-é-o-scaler-e-por-que-precisa-ser-salvo)
15. [Qual a diferença entre fit, transform e fit_transform?](#15-qual-a-diferença-entre-fit-transform-e-fit_transform)
16. [Por que 80% treino e 20% teste?](#16-por-que-80-treino-e-20-teste)
17. [O que acontece se eu mudar o seq_length?](#17-o-que-acontece-se-eu-mudar-o-seq_length)

---

## 1. O que ocorre em cada fase do processamento?

**Referência:** Função `preprocess_data()` no arquivo `src/preprocessing.py`

### Visão Geral do Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                   PIPELINE DE PRÉ-PROCESSAMENTO                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  CSV com preços → Filtrar Close → Normalizar → Criar sequências    │
│                         ↓                                           │
│                   Dividir treino/teste → Converter para tensores   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Fase 1: Normalização

**O que acontece:** Os dados brutos de preço (ex: R$ 3,24 a R$ 27,38) são transformados para valores entre 0 e 1.

**Por que é necessário:** Redes neurais funcionam melhor com valores pequenos e uniformes. Imagine treinar uma rede com preços de R$ 3 a R$ 30 - a diferença de escala causaria problemas no cálculo do erro.

**Decisão de uso:** Foi usado **MinMaxScaler** porque:
- É simples e eficiente
- Preserva a distribuição original dos dados
- É reversível (importante para "desnormalizar" a previsão depois)

```python
# Antes: [3.24, 15.31, 27.38]
# Depois: [0.0, 0.5, 1.0]
```

### Fase 2: Criação de Sequência Temporal

**Quem decide:** **VOCÊ** decide! Não é o modelo nem uma biblioteca que cria automaticamente.

**É implementado manualmente** - veja o código:

```python
def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])    # 60 dias de entrada
        y.append(data[i+seq_length])       # 1 dia de saída
    return np.array(X), np.array(y)
```

**Como é determinado o tamanho:** Você escolhe `seq_length=60` baseado em:
- Conhecimento do domínio (60 dias ≈ 3 meses de mercado)
- Experimentação (testar diferentes valores)
- Recomendação do guia do Tech Challenge

### Fase 3: Conversão para Tensores PyTorch

**O que significa:** Converter arrays NumPy em "tensores" - estruturas de dados especiais que o PyTorch entende e consegue processar de forma otimizada (especialmente em GPU).

**Analogia:** NumPy é como uma planilha Excel. Tensores são como essa planilha convertida para um formato especial que a GPU consegue processar milhões de vezes mais rápido.

---

## 2. O que significa normalizar entre 0 e 1?

**Referência:** Função `normalize_data()` no código

### Definição

Significa transformar qualquer valor para um número entre 0 e 1, proporcional à sua posição no intervalo original.

### Fórmula

```
valor_normalizado = (valor - min) / (max - min)
```

### Exemplo Prático

```
┌─────────────────────────────────────────────────────────────────────┐
│                    NORMALIZAÇÃO MIN-MAX                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Dados originais: Preços de R$ 3,24 a R$ 27,38                      │
│                                                                     │
│  Preço R$ 3,24  (mínimo)  → (3,24 - 3,24) / (27,38 - 3,24) = 0.00  │
│  Preço R$ 15,31 (médio)   → (15,31 - 3,24) / (27,38 - 3,24) = 0.50 │
│  Preço R$ 27,38 (máximo)  → (27,38 - 3,24) / (27,38 - 3,24) = 1.00 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Analogia

É como converter notas de 0-100 para 0-10. A proporção se mantém, mas a escala muda.

| Nota Original (0-100) | Nota Convertida (0-10) |
|----------------------|------------------------|
| 0 | 0 |
| 50 | 5 |
| 100 | 10 |

---

## 3. O que é Exploding Gradients?

**Referência:** Linha 29 do documento principal

### O Problema

Durante o treinamento, a rede ajusta seus pesos calculando "gradientes" (derivadas). Se os valores de entrada forem muito grandes, os gradientes também ficam muito grandes.

### Ilustração

```
┌─────────────────────────────────────────────────────────────────────┐
│                    EXPLODING GRADIENTS                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  SEM normalização (valores grandes):                                │
│  ───────────────────────────────────                                │
│  Gradiente = 1000 × 1000 × 1000 = 1.000.000.000 💥                 │
│                                                                     │
│  Resultado: Os pesos são atualizados com valores ABSURDOS           │
│             O modelo não aprende nada (diverge)                     │
│                                                                     │
│  ─────────────────────────────────────────────────────────────────  │
│                                                                     │
│  COM normalização (valores pequenos):                               │
│  ────────────────────────────────────                               │
│  Gradiente = 0.5 × 0.5 × 0.5 = 0.125 ✅                            │
│                                                                     │
│  Resultado: Os pesos são atualizados de forma controlada            │
│             O modelo aprende gradualmente                           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Relação com Vanishing Gradients

| Problema | O que acontece | Causa |
|----------|----------------|-------|
| **Exploding** | Gradientes muito grandes | Valores de entrada muito grandes |
| **Vanishing** | Gradientes muito pequenos | Muitas camadas, gradientes se multiplicam por valores < 1 |

Ambos impedem o modelo de aprender. A normalização resolve o Exploding; o LSTM resolve o Vanishing.

---

## 4. O que são Janelas Deslizantes?

**Referência:** Função `create_sequences()` no código

### Conceito

Imagine que você tem uma régua de 60cm passando por uma fita longa de dados. A cada passo, você move a régua 1 posição para a direita.

### Visualização

```
┌─────────────────────────────────────────────────────────────────────┐
│                      JANELA DESLIZANTE                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  DADOS: [P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, ...]             │
│                                                                     │
│             ┌─────────────────┐                                     │
│  Posição 1: │P1, P2, P3, P4, P5│ → Prever P6                       │
│             └─────────────────┘                                     │
│                ┌─────────────────┐                                  │
│  Posição 2:    │P2, P3, P4, P5, P6│ → Prever P7                    │
│                └─────────────────┘                                  │
│                   ┌─────────────────┐                               │
│  Posição 3:       │P3, P4, P5, P6, P7│ → Prever P8                 │
│                   └─────────────────┘                               │
│                                                                     │
│  A "janela" (régua) desliza 1 dia de cada vez                      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### No Código

```python
for i in range(len(data) - seq_length):
    X.append(data[i:i+seq_length])      # A janela (60 dias)
    y.append(data[i+seq_length])         # O próximo valor (dia 61)
```

### Finalidade

| Propósito | Explicação |
|-----------|------------|
| **Criar exemplos de treinamento** | De 1487 dias, você cria 1427 exemplos |
| **Dar contexto temporal** | O modelo vê 60 dias para prever o próximo |
| **Maximizar uso dos dados** | Cada dia participa de várias sequências |

---

## 5. Quais são os tipos de normalização além do MinMaxScaler?

**Referência:** Linha 79 do código

### Comparativo de Normalizadores

| Tipo | Fórmula | Range Resultado | Quando usar |
|------|---------|-----------------|-------------|
| **MinMaxScaler** | (x - min)/(max - min) | [0, 1] | Dados sem outliers extremos |
| **StandardScaler** | (x - média)/desvio_padrão | ~[-3, 3] | Dados com distribuição normal |
| **RobustScaler** | (x - mediana)/IQR | Variável | Dados com muitos outliers |
| **MaxAbsScaler** | x / max(\|x\|) | [-1, 1] | Dados já centralizados em 0 |

### Visualização

```
┌─────────────────────────────────────────────────────────────────────┐
│                 COMPARATIVO DE NORMALIZADORES                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  MINMAXSCALER (usado neste projeto)                                 │
│  ──────────────────────────────────                                 │
│  Dados: [10, 20, 30, 40, 50]                                        │
│  Resultado: [0.0, 0.25, 0.5, 0.75, 1.0]                            │
│  → Todos os valores ficam entre 0 e 1                               │
│                                                                     │
│  STANDARDSCALER                                                     │
│  ──────────────                                                     │
│  Dados: [10, 20, 30, 40, 50]                                        │
│  Resultado: [-1.41, -0.71, 0, 0.71, 1.41]                          │
│  → Média = 0, desvio padrão = 1                                     │
│                                                                     │
│  ROBUSTSCALER                                                       │
│  ────────────                                                       │
│  Dados: [10, 20, 30, 40, 1000] (com outlier)                       │
│  Resultado: [-1.0, -0.5, 0, 0.5, 48.5]                             │
│  → Usa mediana, ignora outliers                                     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Por que MinMaxScaler neste projeto?

- Preços de ações geralmente não têm outliers extremos
- Queremos valores estritamente entre 0 e 1
- É simples de reverter (importante para mostrar o preço real na previsão)

---

## 6. Onde e como são salvos os dados processados?

**Referência:** Linhas 201-204 do código

### Arquivos Salvos

Os dados processados são salvos em dois arquivos na pasta `models/`:

```
tech-challenge-lstm/
├── models/
│   ├── scaler.pkl      ← MinMaxScaler treinado
│   └── config.pkl      ← Configurações usadas
```

### Arquivo 1: `models/scaler.pkl`

```python
joblib.dump(scaler, 'models/scaler.pkl')
```

**Contém:** O objeto MinMaxScaler com min e max aprendidos

**Para que serve:** Quando você faz uma previsão, precisa:
1. Normalizar os novos dados (usando o mesmo scaler)
2. Desnormalizar a previsão (converter de 0-1 de volta para reais)

### Arquivo 2: `models/config.pkl`

```python
joblib.dump(config, 'models/config.pkl')
```

**Contém:** Dicionário com configurações usadas:
- `seq_length`: 60
- `ticker`: "PETR4.SA"
- outras configurações

### O que NÃO é salvo

```
┌─────────────────────────────────────────────────────────────────────┐
│                    O QUE É SALVO vs NÃO SALVO                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ✅ SALVO:                                                          │
│  • scaler.pkl (MinMaxScaler)                                        │
│  • config.pkl (configurações)                                       │
│  • model_lstm.pth (pesos do modelo - salvo na Etapa 4)             │
│                                                                     │
│  ❌ NÃO SALVO:                                                      │
│  • X_train, X_test (dados de treino/teste)                         │
│  • y_train, y_test (labels)                                         │
│  • Dados normalizados intermediários                                │
│                                                                     │
│  Por quê? São usados apenas durante o treinamento e depois         │
│  descartados. O modelo aprendeu os padrões - não precisa mais      │
│  dos dados brutos.                                                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Fluxo Completo

```
TREINAMENTO (uma vez):
─────────────────────
Dados brutos → Normalizar → Criar sequências → Treinar modelo
                   │                               │
                   ↓                               ↓
            scaler.pkl                     model_lstm.pth

INFERÊNCIA (muitas vezes):
──────────────────────────
1. Carrega scaler.pkl
2. Normaliza dados novos
3. Carrega model_lstm.pth
4. Faz previsão (valor entre 0-1)
5. Desnormaliza resultado (converte para R$)
```

---

## 7. O que significa Inferência na prática?

**Referência:** Conceito usado em Machine Learning

### Definição

**Inferência** = usar o modelo treinado para fazer previsões em dados novos.

### Comparação: Treinamento vs Inferência

```
┌─────────────────────────────────────────────────────────────────────┐
│                  TREINAMENTO vs INFERÊNCIA                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  TREINAMENTO (acontece UMA vez):                                    │
│  ────────────────────────────────                                   │
│  • Você dá dados + respostas certas                                 │
│  • Modelo aprende ajustando os pesos                                │
│  • Demora minutos/horas                                             │
│  • Usa GPU intensivamente                                           │
│  • Resultado: arquivo .pth com pesos aprendidos                     │
│                                                                     │
│  INFERÊNCIA (acontece MUITAS vezes):                                │
│  ─────────────────────────────────────                              │
│  • Você dá apenas dados novos (sem resposta)                        │
│  • Modelo usa os pesos FIXOS para prever                            │
│  • Demora milissegundos                                             │
│  • API retorna previsão                                             │
│  • Resultado: preço previsto                                        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Exemplo Prático

| Fase | Input | Output |
|------|-------|--------|
| **Treinamento** | 5 anos de preços + preços reais do dia seguinte | Modelo treinado (.pth) |
| **Inferência** | Últimos 60 dias de preços | Previsão do próximo dia |

### Analogia

- **Treinamento:** Estudar para uma prova (demora, requer esforço)
- **Inferência:** Fazer a prova (usa o que aprendeu, é rápido)

---

## 8. Janela Temporal, Sequência Temporal e Janela Deslizante são a mesma coisa?

**Referência:** Função `create_sequences()` no código

### Resposta: Sim, são termos intercambiáveis!

| Termo | Ênfase | Uso comum |
|-------|--------|-----------|
| **Sequência temporal** | O grupo de dados ordenados | "Criamos sequências de 60 dias" |
| **Janela temporal** | O recorte/tamanho | "A janela tem 60 dias" |
| **Janela deslizante** | O processo de criação | "A janela desliza 1 dia por vez" |

### Visualização

```
┌─────────────────────────────────────────────────────────────────────┐
│                   TERMOS EQUIVALENTES                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  "Criar sequências temporais de 60 dias"                            │
│                    =                                                │
│  "Usar janela temporal de 60 dias"                                  │
│                    =                                                │
│  "Aplicar janela deslizante de tamanho 60"                         │
│                                                                     │
│  TODOS significam: pegar 60 dias consecutivos para usar como input │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Por que 60 dias?

| Razão | Explicação |
|-------|------------|
| ~3 meses de histórico | Captura tendências de curto/médio prazo |
| Recomendação do guia | O Tech Challenge sugere esse valor |
| Padrão comum | Análise técnica frequentemente usa 60 dias |

**Nota:** Você poderia usar 30, 90 ou 120 dias - cada escolha capturaria padrões diferentes. O valor 60 é um bom ponto de partida.

---

## 9. O que são Tensores? (Explicação para leigos)

**Referência:** Função `to_tensors()` no código

### Analogia: Do Conhecido ao Novo

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TENSOR PARA LEIGOS                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  VOCÊ JÁ CONHECE:                                                   │
│  ────────────────                                                   │
│                                                                     │
│  Um número:           42                    (escalar - 0D)          │
│                       └── Um único número                           │
│                                                                     │
│  Uma lista:           [1, 2, 3, 4, 5]       (vetor - 1D)            │
│                       └── Números em fila                           │
│                                                                     │
│  Uma tabela Excel:    | A | B | C |         (matriz - 2D)           │
│                       | 1 | 2 | 3 |                                 │
│                       | 4 | 5 | 6 |                                 │
│                       └── Números em linhas e colunas               │
│                                                                     │
│  O QUE É TENSOR:                                                    │
│  ───────────────                                                    │
│                                                                     │
│  Tensor 3D:           (é como um livro de tabelas)                  │
│                       Página 1: [[1,2], [3,4]]                      │
│                       Página 2: [[5,6], [7,8]]                      │
│                       └── Múltiplas tabelas empilhadas              │
│                                                                     │
│  TENSOR = Array multidimensional otimizado para Deep Learning       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Por que não usar só NumPy?

| Aspecto | NumPy | PyTorch Tensor |
|---------|-------|----------------|
| Cálculo em GPU | ❌ Não | ✅ **Sim** |
| Gradientes automáticos | ❌ Não | ✅ **Sim** |
| Velocidade em Deep Learning | Lento | **Muito Rápido** |

### O que `torch.FloatTensor()` faz?

```python
# Seus dados em NumPy (formato Python comum)
X_train = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

# Convertendo para Tensor (formato que PyTorch entende)
X_train_t = torch.FloatTensor(X_train)
```

**Analogia:** É como salvar um documento Word como PDF - o conteúdo é o mesmo, mas o formato é diferente para um propósito específico.

### Por que Float (32 bits)?

| Tipo | Precisão | Memória | Uso |
|------|----------|---------|-----|
| Float16 | Baixa | 2 bytes | Inferência rápida |
| **Float32** | **Média** | **4 bytes** | **Treinamento padrão** |
| Float64 | Alta | 8 bytes | Cálculos científicos |

Float32 é o padrão porque oferece precisão suficiente com bom uso de memória.

---

## 10. Quais são os fundamentos de Deep Learning?

**Referência:** Conceitos das aulas de Redes Neurais

### Fundamentos Principais

| Fundamento | O que é | Explicado nas aulas? |
|------------|---------|---------------------|
| **Tensores** | Estrutura de dados para redes neurais | ✅ Sim |
| **Gradientes** | Como a rede aprende (direção do ajuste) | ✅ Sim |
| **Backpropagation** | Algoritmo para calcular gradientes | ✅ Sim |
| **Função de perda (Loss)** | Mede o erro do modelo | ✅ Sim |
| **Otimizador** | Ajusta os pesos (ex: Adam, SGD) | ✅ Sim |
| **Épocas/Batch** | Organização do treinamento | ✅ Sim |
| **Overfitting** | Modelo "decora" ao invés de aprender | ✅ Sim |
| **Regularização/Dropout** | Técnicas para evitar overfitting | ✅ Sim |

### Mapa Mental

```
┌─────────────────────────────────────────────────────────────────────┐
│                   FUNDAMENTOS DE DEEP LEARNING                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│                        ┌──────────────┐                             │
│                        │   DADOS      │                             │
│                        │  (Tensores)  │                             │
│                        └──────┬───────┘                             │
│                               │                                     │
│                               ▼                                     │
│                        ┌──────────────┐                             │
│                        │    MODELO    │                             │
│                        │ (Rede Neural)│                             │
│                        └──────┬───────┘                             │
│                               │                                     │
│         ┌─────────────────────┼─────────────────────┐               │
│         │                     │                     │               │
│         ▼                     ▼                     ▼               │
│  ┌────────────┐       ┌────────────┐       ┌────────────┐          │
│  │  FORWARD   │       │    LOSS    │       │ BACKWARD   │          │
│  │   PASS     │──────▶│  (Erro)    │──────▶│   PASS     │          │
│  │(Previsão)  │       │            │       │(Gradientes)│          │
│  └────────────┘       └────────────┘       └─────┬──────┘          │
│                                                  │                  │
│                                                  ▼                  │
│                                          ┌────────────┐            │
│                                          │ OTIMIZADOR │            │
│                                          │(Ajusta     │            │
│                                          │ pesos)     │            │
│                                          └────────────┘            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 11. O que é a coluna Close?

**Referência:** Linha 187 do código

### Definição

`Close` é o **preço de fechamento** da ação - o último preço negociado no dia (às 17h no Brasil).

```python
data = df['Close'].values.reshape(-1, 1)
```

### Todas as Colunas OHLCV

| Coluna | Nome | O que representa |
|--------|------|------------------|
| Open | Abertura | Primeiro preço do dia (às 10h) |
| High | Máxima | Maior preço atingido no dia |
| Low | Mínima | Menor preço atingido no dia |
| **Close** | **Fechamento** | **Último preço do dia (às 17h)** |
| Volume | Volume | Quantidade de ações negociadas |

### Por que usar só Close?

| Motivo | Explicação |
|--------|------------|
| **Representa o consenso** | É o preço que o mercado "concordou" no fim do dia |
| **Mais estável** | Menos sujeito a oscilações momentâneas |
| **Padrão da indústria** | Analistas usam Close como referência |
| **Simplifica o modelo** | 1 feature ao invés de 5 |

### O que significa `.reshape(-1, 1)`?

```python
# Antes do reshape:
data = [10.5, 11.2, 12.0, ...]  # Shape: (1487,) - vetor 1D

# Depois do reshape:
data = [[10.5],
        [11.2],
        [12.0],
        ...]                    # Shape: (1487, 1) - matriz 2D

# O -1 significa "calcule automaticamente essa dimensão"
```

O MinMaxScaler espera dados em 2D (linhas × colunas), por isso fazemos o reshape.

---

## 12. Como interpretar os shapes dos dados?

**Referência:** Linhas 210-229 do documento principal

### Explicação Visual Passo a Passo

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DO DADO BRUTO AO TENSOR                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1️⃣ DADOS ORIGINAIS (1487 dias de preço)                           │
│  ════════════════════════════════════════                           │
│                                                                     │
│  Imagine uma lista de preços:                                       │
│  [R$10, R$11, R$12, ..., R$45]  ← 1487 números em fila             │
│  Shape: (1487,)                                                     │
│                                                                     │
│  2️⃣ APÓS RESHAPE (para MinMaxScaler)                               │
│  ════════════════════════════════════                               │
│                                                                     │
│  [[R$10],                                                           │
│   [R$11],                                                           │
│   [R$12],                                                           │
│   ...]                                                              │
│  Shape: (1487, 1)  ← 1487 linhas, 1 coluna                         │
│                                                                     │
│  3️⃣ CRIAR SEQUÊNCIAS (janela de 60 dias)                           │
│  ════════════════════════════════════════                           │
│                                                                     │
│  Total: 1487 - 60 = 1427 sequências                                 │
│                                                                     │
│  Sequência 1: [Dia1, Dia2, ..., Dia60]  → Prever Dia61             │
│  Sequência 2: [Dia2, Dia3, ..., Dia61]  → Prever Dia62             │
│  ...                                                                │
│  Sequência 1427: [Dia1427, ..., Dia1486] → Prever Dia1487          │
│                                                                     │
│  Shape X: (1427, 60, 1)                                             │
│            │     │   └── 1 feature (só Close)                       │
│            │     └────── 60 dias por sequência                      │
│            └──────────── 1427 sequências                            │
│                                                                     │
│  Shape y: (1427, 1)                                                 │
│            │     └── 1 valor (preço a prever)                       │
│            └──────── 1427 respostas                                 │
│                                                                     │
│  4️⃣ DIVIDIR TREINO/TESTE (80%/20%)                                 │
│  ════════════════════════════════════                               │
│                                                                     │
│  X_train: (1141, 60, 1)  │  X_test: (286, 60, 1)                   │
│  y_train: (1141, 1)      │  y_test: (286, 1)                       │
│                                                                     │
│  1141 + 286 = 1427 ✅                                               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Visualização 3D do X_train

```
         Sequência 1        Sequência 2           Sequência 1141
        ┌─────────┐        ┌─────────┐           ┌─────────┐
  Dia 1 │  0.15   │        │  0.16   │           │  0.89   │
  Dia 2 │  0.18   │        │  0.17   │    ...    │  0.87   │
  Dia 3 │  0.20   │        │  0.19   │           │  0.85   │
   ...  │   ...   │        │   ...   │           │   ...   │
 Dia 60 │  0.35   │        │  0.36   │           │  0.92   │
        └─────────┘        └─────────┘           └─────────┘
             │                  │                     │
             ▼                  ▼                     ▼
  y_train:  0.36               0.37                  0.93
         (preço real       (preço real           (preço real
          do dia 61)        do dia 62)          do dia 1487)
```

---

## 13. Por que não embaralhamos os dados?

**Referência:** Linha 150 do documento principal

### O Motivo

Em **séries temporais**, a ordem dos dados importa! Se embaralharmos:

```
┌─────────────────────────────────────────────────────────────────────┐
│               POR QUE NÃO EMBARALHAR SÉRIES TEMPORAIS               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  DADOS CORRETOS (ordem preservada):                                 │
│  ─────────────────────────────────                                  │
│  Treino: Jan/2018 ──────────────────────► Dez/2022                 │
│  Teste:  Jan/2023 ──────────────────────► Dez/2023                 │
│                                                                     │
│  ✅ Modelo treina no passado, testa no futuro (realista!)          │
│                                                                     │
│  ─────────────────────────────────────────────────────────────────  │
│                                                                     │
│  DADOS EMBARALHADOS (ERRADO):                                       │
│  ────────────────────────────                                       │
│  Treino: Jul/2020, Fev/2018, Nov/2023, Mar/2019, ...               │
│  Teste:  Abr/2018, Set/2022, Jun/2021, ...                         │
│                                                                     │
│  ❌ Modelo "vê o futuro" durante o treino (data leakage!)          │
│  ❌ Avaliação não reflete uso real                                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Data Leakage (Vazamento de Dados)

Se você embaralhar, dados de 2023 podem aparecer no treino, e dados de 2019 no teste. O modelo "aprenderia o futuro" e teria resultados artificialmente bons que não se repetem na prática.

### Comparação: Imagens vs Séries Temporais

| Tipo de dado | Pode embaralhar? | Por quê? |
|--------------|------------------|----------|
| Imagens de gatos/cachorros | ✅ Sim | Uma foto não depende de outra |
| Preços de ações | ❌ Não | O preço de hoje depende de ontem |

---

## 14. O que é o Scaler e por que precisa ser salvo?

**Referência:** Linha 202 do código

### O que é

O **Scaler** (MinMaxScaler) é o objeto que "aprendeu" os valores mínimo e máximo dos dados de treino.

### Por que salvar?

```
┌─────────────────────────────────────────────────────────────────────┐
│                    POR QUE SALVAR O SCALER?                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  DURANTE O TREINAMENTO:                                             │
│  ──────────────────────                                             │
│  Scaler aprende: min = 3.24, max = 27.38                            │
│  Normaliza: 15.31 → 0.50                                            │
│                                                                     │
│  DEPOIS, NA INFERÊNCIA:                                             │
│  ───────────────────────                                            │
│  1. Usuário envia: [25.0, 26.0, 27.0, ...]  (60 dias)              │
│                                                                     │
│  2. Precisa normalizar com O MESMO scaler:                          │
│     25.0 → (25.0 - 3.24) / (27.38 - 3.24) = 0.90                   │
│                                                                     │
│  3. Modelo prevê: 0.92 (valor normalizado)                          │
│                                                                     │
│  4. Precisa DESNORMALIZAR para mostrar em R$:                       │
│     0.92 → 0.92 × (27.38 - 3.24) + 3.24 = R$ 25.45                 │
│                                                                     │
│  SE NÃO SALVAR o scaler, você não consegue fazer os passos 2 e 4!  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Código de Uso

```python
# Salvando (durante treinamento)
import joblib
joblib.dump(scaler, 'models/scaler.pkl')

# Carregando (durante inferência)
scaler = joblib.load('models/scaler.pkl')

# Normalizando novos dados
dados_normalizados = scaler.transform(novos_dados)

# Desnormalizando previsão
preco_real = scaler.inverse_transform(previsao_normalizada)
```

---

## 15. Qual a diferença entre fit, transform e fit_transform?

**Referência:** Linha 80 do código

### Os Três Métodos

```python
scaler = MinMaxScaler()

# fit: Aprende os parâmetros (min, max)
scaler.fit(dados_treino)

# transform: Aplica a transformação usando parâmetros aprendidos
dados_normalizados = scaler.transform(dados_treino)

# fit_transform: Faz os dois de uma vez (mais eficiente)
dados_normalizados = scaler.fit_transform(dados_treino)
```

### Quando usar cada um?

```
┌─────────────────────────────────────────────────────────────────────┐
│                   QUANDO USAR CADA MÉTODO                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  TREINAMENTO:                                                       │
│  ────────────                                                       │
│  dados_treino → fit_transform → dados_normalizados                  │
│                 (aprende E aplica)                                  │
│                                                                     │
│  INFERÊNCIA/TESTE:                                                  │
│  ─────────────────                                                  │
│  dados_novos → transform → dados_normalizados                       │
│                (só aplica, NÃO aprende novamente!)                  │
│                                                                     │
│  ⚠️ ERRO COMUM:                                                     │
│  Usar fit_transform em dados de teste → causa data leakage!        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Por que não fazer fit nos dados de teste?

Se você fizer `fit` nos dados de teste, o scaler vai aprender min/max diferentes, e os dados não serão comparáveis com o que o modelo aprendeu.

| Fase | Método correto |
|------|----------------|
| Treino | `fit_transform` |
| Teste/Inferência | `transform` (apenas) |

---

## 16. Por que 80% treino e 20% teste?

**Referência:** Linha 65 do código (`TRAIN_SPLIT = 0.8`)

### A Lógica

| Proporção | Para que serve |
|-----------|----------------|
| **80% Treino** | Dados para o modelo aprender padrões |
| **20% Teste** | Dados que o modelo NUNCA viu, para avaliar se generalizou |

### Por que essa divisão?

```
┌─────────────────────────────────────────────────────────────────────┐
│                     TRADE-OFF DA DIVISÃO                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  MUITO TREINO (ex: 95/5):                                           │
│  • ✅ Modelo aprende mais padrões                                   │
│  • ❌ Poucos dados para avaliar (teste não confiável)               │
│                                                                     │
│  POUCO TREINO (ex: 50/50):                                          │
│  • ✅ Avaliação mais confiável                                      │
│  • ❌ Modelo não aprende o suficiente                               │
│                                                                     │
│  80/20 é um EQUILÍBRIO comum na indústria.                         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Outras proporções comuns

| Divisão | Quando usar |
|---------|-------------|
| 80/20 | Padrão geral |
| 70/30 | Quando quer mais confiança no teste |
| 90/10 | Quando tem poucos dados |
| 70/15/15 | Com conjunto de validação separado |

---

## 17. O que acontece se eu mudar o seq_length?

**Referência:** Linha 64 do código (`SEQ_LENGTH = 60`)

### Impacto de Diferentes Valores

| seq_length | Prós | Contras |
|------------|------|---------|
| **30 dias** | Mais amostras de treino, treino mais rápido | Pode perder padrões de longo prazo |
| **60 dias** | Bom equilíbrio (padrão do projeto) | - |
| **120 dias** | Captura mais contexto | Menos amostras, modelo mais pesado |

### Cálculo do Impacto

```
Dados originais: 1487 dias

seq_length = 30: 1487 - 30 = 1457 sequências
seq_length = 60: 1487 - 60 = 1427 sequências
seq_length = 120: 1487 - 120 = 1367 sequências
```

### Visualização

```
┌─────────────────────────────────────────────────────────────────────┐
│                   IMPACTO DO SEQ_LENGTH                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  seq_length PEQUENO (30):                                           │
│  ────────────────────────                                           │
│  [Dia1...Dia30] → Prever Dia31                                     │
│  • Contexto curto (1 mês)                                           │
│  • Bom para padrões de curto prazo                                  │
│  • Modelo leve e rápido                                             │
│                                                                     │
│  seq_length MÉDIO (60):                                             │
│  ──────────────────────                                             │
│  [Dia1...Dia60] → Prever Dia61                                     │
│  • Contexto médio (3 meses)                                         │
│  • Equilíbrio entre curto e longo prazo                             │
│  • Padrão recomendado                                               │
│                                                                     │
│  seq_length GRANDE (120):                                           │
│  ───────────────────────                                            │
│  [Dia1...Dia120] → Prever Dia121                                   │
│  • Contexto longo (6 meses)                                         │
│  • Captura tendências sazonais                                      │
│  • Modelo mais pesado e lento                                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Dica Prática

Não existe "valor perfeito" - o ideal é **experimentar** diferentes valores e ver qual dá melhor resultado no seu caso específico.

---

## 🔗 Navegação

| Anterior | Próximo |
|----------|---------|
| [ETAPA 02 - FAQ](./ETAPA_02_FAQ_Duvidas.md) | [ETAPA 04 - Modelo LSTM](./ETAPA_04_Modelo_LSTM.md) |

---

*Documento criado para esclarecer dúvidas comuns sobre a Etapa 3 do projeto LSTM.*
