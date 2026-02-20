# Modelos de Machine Learning - Visão Geral

Este documento apresenta uma visão geral sobre modelos de Machine Learning, tipos de dados, arquiteturas neurais e quando usar cada abordagem.

O documento inclui:
1 - Famílias de Modelos
2 - Tipos de Dados e Arquiteturas Recomendadas
3 - Redes Neurais Recorrentes (RNNs)
4 - O que são Séries Temporais
5 - Comparativo de Arquiteturas
6 - Quando Usar Cada Modelo
7 - Glossário de Termos

---

## 1. Famílias de Modelos

### 1.1 Panorama Geral

Existem milhares de arquiteturas de modelos, mas podem ser organizadas em famílias principais:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        FAMÍLIAS DE MODELOS                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  REDES FEEDFORWARD (MLPs)                                                   │
│  └── Perceptron, MLP (Multi-Layer Perceptron)                              │
│      • Dados fluem em uma direção (entrada → saída)                        │
│      • Boas para dados tabulares                                           │
│                                                                             │
│  REDES CONVOLUCIONAIS (CNNs)                                                │
│  └── LeNet, AlexNet, VGG, ResNet, EfficientNet, YOLO...                    │
│      • Especializadas em padrões espaciais                                 │
│      • Dominam visão computacional (imagens, vídeos)                       │
│                                                                             │
│  REDES RECORRENTES (RNNs)                                                   │
│  └── RNN Simples (Vanilla), LSTM, GRU, Bidirectional                       │
│      • Processam sequências (ordem importa)                                │
│      • Séries temporais, texto, áudio                                      │
│                                                                             │
│  TRANSFORMERS                                                               │
│  └── BERT, GPT, T5, ViT, LLaMA, Claude, Gemini...                         │
│      • Atenção paralela (não sequencial)                                   │
│      • Dominam NLP e estão expandindo para tudo                           │
│                                                                             │
│  AUTOENCODERS                                                               │
│  └── AE, VAE (Variational), Denoising AE                                   │
│      • Comprimem e reconstroem dados                                       │
│      • Redução de dimensionalidade, detecção de anomalias                 │
│                                                                             │
│  REDES GENERATIVAS                                                          │
│  └── GAN, Diffusion Models (Stable Diffusion), Flow-based                  │
│      • Geram dados novos (imagens, texto, áudio)                          │
│      • Criação de conteúdo, data augmentation                             │
│                                                                             │
│  GRAPH NEURAL NETWORKS (GNNs)                                               │
│  └── GCN, GAT, GraphSAGE                                                   │
│      • Processam dados em formato de grafo                                 │
│      • Redes sociais, moléculas, recomendação                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Modelos Clássicos (Não Deep Learning)

Além de redes neurais profundas, existem modelos clássicos muito usados:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      MODELOS CLÁSSICOS                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ÁRVORES DE DECISÃO                                                         │
│  └── Decision Tree, Random Forest, XGBoost, LightGBM, CatBoost             │
│      • Muito usados em dados tabulares                                     │
│      • Interpretáveis, rápidos                                             │
│      • Ganham muitas competições Kaggle                                    │
│                                                                             │
│  MODELOS LINEARES                                                           │
│  └── Regressão Linear, Logística, Ridge, Lasso, ElasticNet                 │
│      • Simples, interpretáveis                                             │
│      • Baseline para comparação                                            │
│                                                                             │
│  SUPPORT VECTOR MACHINES (SVM)                                              │
│  └── SVC, SVR, Kernel SVM                                                  │
│      • Bons com poucos dados                                               │
│      • Classificação e regressão                                           │
│                                                                             │
│  CLUSTERING                                                                 │
│  └── K-Means, DBSCAN, Hierarchical                                         │
│      • Agrupamento não supervisionado                                      │
│      • Descobrir estruturas nos dados                                      │
│                                                                             │
│  SÉRIES TEMPORAIS CLÁSSICAS                                                 │
│  └── ARIMA, SARIMA, Prophet, Exponential Smoothing                         │
│      • Modelos estatísticos para previsão                                  │
│      • Não são redes neurais                                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Tipos de Dados e Arquiteturas Recomendadas

### 2.1 Tipos de Dados em ML

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        TIPOS DE DADOS                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. DADOS TABULARES                                                         │
│     ┌────────┬────────┬────────┬────────┐                                  │
│     │ Nome   │ Idade  │ Salário│ Cidade │                                  │
│     ├────────┼────────┼────────┼────────┤                                  │
│     │ João   │ 25     │ 5000   │ SP     │                                  │
│     │ Maria  │ 30     │ 7000   │ RJ     │                                  │
│     └────────┴────────┴────────┴────────┘                                  │
│     Exemplo: Planilhas, bancos de dados                                    │
│                                                                             │
│  2. SÉRIES TEMPORAIS                                                        │
│     [t1: 32.5] → [t2: 33.1] → [t3: 33.8] → [t4: 34.2] → ...              │
│     Exemplo: Preços de ações, temperatura, vendas diárias                  │
│                                                                             │
│  3. IMAGENS                                                                 │
│     ┌─────────────┐                                                        │
│     │ ░░▓▓▓▓▓░░  │  Matriz de pixels (altura × largura × canais)         │
│     │ ░▓▓███▓▓░  │  RGB: 3 canais, Grayscale: 1 canal                    │
│     │ ▓▓█████▓▓  │                                                        │
│     └─────────────┘                                                        │
│     Exemplo: Fotos, raio-X, satélite                                       │
│                                                                             │
│  4. TEXTO / NLP                                                             │
│     "O gato sentou no tapete"                                              │
│     → Sequência de tokens/palavras                                         │
│     Exemplo: Reviews, documentos, chat                                      │
│                                                                             │
│  5. ÁUDIO                                                                   │
│     ∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿                                                      │
│     → Sinal no tempo (waveform) ou espectrograma                          │
│     Exemplo: Música, voz, sons ambientes                                   │
│                                                                             │
│  6. VÍDEO                                                                   │
│     [Frame1] → [Frame2] → [Frame3] → ...                                   │
│     → Sequência de imagens no tempo                                        │
│     Exemplo: Filmes, câmeras de segurança                                  │
│                                                                             │
│  7. GRAFOS                                                                  │
│        A ─── B                                                             │
│       /│\    │                                                             │
│      C │ D   E                                                             │
│     → Nós e arestas (relações)                                            │
│     Exemplo: Redes sociais, moléculas, mapas                               │
│                                                                             │
│  8. DADOS 3D                                                                │
│     Nuvens de pontos, meshes, voxels                                       │
│     Exemplo: LIDAR, modelos CAD, escaneamento 3D                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Arquitetura Recomendada por Tipo de Dado

| Tipo de Dado | Arquitetura Principal | Por quê? | Alternativas |
|--------------|----------------------|----------|--------------|
| **Tabular** | XGBoost, Random Forest | Rápidos, interpretáveis | MLP, TabNet |
| **Série Temporal** | LSTM, GRU | Capturam dependências temporais | Transformer, ARIMA |
| **Imagem** | CNN (ResNet, EfficientNet) | Capturam padrões espaciais | ViT (Vision Transformer) |
| **Texto** | Transformer (BERT, GPT) | Atenção paralela, contexto longo | LSTM, RNN |
| **Áudio** | CNN 1D, Wav2Vec | Padrões frequenciais | LSTM, Transformer |
| **Vídeo** | CNN + LSTM, 3D CNN | Espacial + temporal | Video Transformer |
| **Grafo** | GNN (GCN, GAT) | Processam relações | Node2Vec + MLP |

---

## 3. Redes Neurais Recorrentes (RNNs)

### 3.1 O que são RNNs?

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    O QUE SÃO RNNs?                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  RNN = Rede Neural Recorrente                                               │
│                                                                             │
│  É uma rede que processa dados SEQUENCIAIS, onde a saída de um            │
│  passo alimenta a entrada do próximo:                                       │
│                                                                             │
│     x₁        x₂        x₃        x₄               x_n                     │
│      │         │         │         │                 │                      │
│      ▼         ▼         ▼         ▼                 ▼                      │
│    ┌───┐     ┌───┐     ┌───┐     ┌───┐           ┌───┐                    │
│    │RNN│────►│RNN│────►│RNN│────►│RNN│──► ... ──►│RNN│                    │
│    └───┘     └───┘     └───┘     └───┘           └───┘                    │
│      │         │         │         │                 │                      │
│      ▼         ▼         ▼         ▼                 ▼                      │
│     h₁        h₂        h₃        h₄               h_n                     │
│                                                      │                      │
│                                               SAÍDA FINAL                   │
│                                                                             │
│  A mesma célula é reutilizada, mas ACUMULA informação                      │
│  dos passos anteriores (memória).                                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Família RNN

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    FAMÍLIA DE RNNs                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                           RNN (conceito geral)                              │
│                           ════════════════════                              │
│                                   │                                         │
│            ┌──────────────────────┼──────────────────────┐                 │
│            │                      │                      │                  │
│            ▼                      ▼                      ▼                  │
│      ┌──────────┐          ┌──────────┐          ┌──────────┐             │
│      │   RNN    │          │   LSTM   │          │   GRU    │             │
│      │ Simples  │          │          │          │          │             │
│      │(Vanilla) │          │          │          │          │             │
│      └──────────┘          └──────────┘          └──────────┘             │
│                                                                             │
│  RNN Simples:                                                               │
│  • 1 operação (tanh)                                                       │
│  • Memória curta (vanishing gradient)                                      │
│  • Rápida, poucos parâmetros                                               │
│                                                                             │
│  LSTM (Long Short-Term Memory):                                             │
│  • 3 portões + 2 estados                                                   │
│  • Memória longa                                                           │
│  • 4x mais parâmetros que RNN simples                                      │
│                                                                             │
│  GRU (Gated Recurrent Unit):                                                │
│  • 2 portões + 1 estado                                                    │
│  • Meio-termo entre RNN e LSTM                                             │
│  • ~3x mais parâmetros que RNN simples                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Comparativo: RNN vs LSTM vs GRU

| Aspecto | RNN Simples | LSTM | GRU |
|---------|-------------|------|-----|
| Estados | 1 (h) | 2 (h, c) | 1 (h) |
| Portões | 0 | 3 | 2 |
| Parâmetros | 1x | 4x | 3x |
| Memória longa | Ruim | Boa | Boa |
| Velocidade | Rápida | Lenta | Média |
| Quando usar | Sequências curtas | Sequências longas | Quando LSTM é lento demais |

### 3.4 RNN para Séries Temporais

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                  RNN PARA SÉRIES TEMPORAIS                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Quando aplicada a séries temporais (como preços), a RNN:                  │
│                                                                             │
│  1. RECEBE um ponto de cada vez:                                           │
│     preço do dia 1, dia 2, dia 3...                                        │
│                                                                             │
│  2. ACUMULA contexto:                                                       │
│     "o preço subiu nos últimos 5 dias"                                     │
│                                                                             │
│  3. USA o contexto para prever:                                            │
│     "provavelmente vai continuar subindo"                                  │
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  Série Temporal:     [32.5, 33.1, 33.8, 34.2, 35.0, ...]                   │
│                          │      │      │      │      │                      │
│                          ▼      ▼      ▼      ▼      ▼                      │
│                      ┌─────┬─────┬─────┬─────┬─────┐                       │
│  RNN processando:    │ RNN │ RNN │ RNN │ RNN │ RNN │──► Previsão           │
│                      └─────┴─────┴─────┴─────┴─────┘                       │
│                          ──────────────────────────►                        │
│                                fluxo de memória                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. O que são Séries Temporais

### 4.1 Definição

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    O QUE É SÉRIE TEMPORAL?                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Série Temporal = Sequência de dados ordenados no TEMPO                    │
│                                                                             │
│  CARACTERÍSTICA PRINCIPAL: A ORDEM IMPORTA!                                │
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  EXEMPLOS DE SÉRIES TEMPORAIS:                                             │
│                                                                             │
│  • Preços de ações:                                                         │
│    [R$32, R$33, R$35, R$34, R$36, ...]                                     │
│                                                                             │
│  • Temperatura diária:                                                      │
│    [25°C, 26°C, 24°C, 23°C, 27°C, ...]                                     │
│                                                                             │
│  • Vendas mensais:                                                          │
│    [1000, 1200, 900, 1500, 1300, ...]                                      │
│                                                                             │
│  • Batimentos cardíacos:                                                    │
│    [72, 75, 71, 80, 78, ...]                                               │
│                                                                             │
│  • Consumo de energia:                                                      │
│    [150kWh, 160kWh, 145kWh, 170kWh, ...]                                   │
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  POR QUE A ORDEM IMPORTA?                                                   │
│                                                                             │
│  [32, 33, 35, 34, 36] → Tendência de ALTA                                  │
│  [36, 34, 35, 33, 32] → Tendência de BAIXA                                 │
│                                                                             │
│  Mesmos números, ordens diferentes, significados opostos!                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Série Temporal vs Outros Dados

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                SÉRIE TEMPORAL vs DADOS TABULARES                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  DADOS TABULARES (ordem NÃO importa):                                      │
│  ┌────────┬────────┬────────┐                                              │
│  │ Nome   │ Idade  │ Salário│                                              │
│  ├────────┼────────┼────────┤                                              │
│  │ João   │ 25     │ 5000   │  ← Pode embaralhar as linhas               │
│  │ Maria  │ 30     │ 7000   │    que o modelo não é afetado              │
│  │ Pedro  │ 28     │ 6000   │                                              │
│  └────────┴────────┴────────┘                                              │
│                                                                             │
│  SÉRIE TEMPORAL (ordem IMPORTA):                                           │
│  ┌────────┬────────┬────────┬────────┬────────┐                           │
│  │ Dia 1  │ Dia 2  │ Dia 3  │ Dia 4  │ Dia 5  │                           │
│  ├────────┼────────┼────────┼────────┼────────┤                           │
│  │ R$32   │ R$33   │ R$35   │ R$34   │ R$36   │  ← NÃO pode embaralhar!  │
│  └────────┴────────┴────────┴────────┴────────┘    A sequência é crucial  │
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  MODELOS RECOMENDADOS:                                                      │
│                                                                             │
│  Dados Tabulares:                                                           │
│  • XGBoost, Random Forest (tratam linhas independentemente)                │
│  • MLP (não considera ordem)                                               │
│                                                                             │
│  Série Temporal:                                                            │
│  • LSTM, GRU (processam sequencialmente, mantêm memória)                  │
│  • Transformer (atenção entre todos os pontos)                             │
│  • ARIMA, Prophet (modelos estatísticos)                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.3 Componentes de uma Série Temporal

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              COMPONENTES DE SÉRIE TEMPORAL                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Uma série temporal pode ser decomposta em:                                 │
│                                                                             │
│  1. TENDÊNCIA (Trend)                                                       │
│     Direção geral de longo prazo                                           │
│                                                                             │
│     ───────────────────────►                                               │
│         crescimento ao longo do tempo                                       │
│                                                                             │
│  2. SAZONALIDADE (Seasonality)                                              │
│     Padrões que se repetem em intervalos fixos                             │
│                                                                             │
│     ∿∿∿∿∿∿∿∿∿∿∿∿                                                          │
│     vendas aumentam todo dezembro                                          │
│                                                                             │
│  3. CICLO (Cycle)                                                           │
│     Flutuações não fixas (econômicas, etc)                                 │
│                                                                             │
│     ∼∼∼∼∼∼∼∼∼∼                                                             │
│     ciclos econômicos de 7-10 anos                                         │
│                                                                             │
│  4. RUÍDO (Noise)                                                           │
│     Variação aleatória, imprevisível                                       │
│                                                                             │
│     .:·:.':.':.':.                                                         │
│     flutuações diárias aleatórias                                          │
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  SÉRIE = TENDÊNCIA + SAZONALIDADE + CICLO + RUÍDO                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Comparativo de Arquiteturas

### 5.1 Quando Usar Cada Arquitetura

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              QUANDO USAR CADA ARQUITETURA                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  MLP (Multi-Layer Perceptron)                                               │
│  ════════════════════════════                                               │
│  ✅ Dados tabulares simples                                                │
│  ✅ Quando interpretabilidade não é crucial                                │
│  ❌ Imagens, texto, séries temporais                                       │
│                                                                             │
│  CNN (Convolutional Neural Network)                                         │
│  ═══════════════════════════════════                                        │
│  ✅ Imagens (classificação, detecção)                                      │
│  ✅ Áudio (como espectrograma)                                             │
│  ✅ Séries temporais 1D (com CNN 1D)                                       │
│  ❌ Texto longo, grafos                                                    │
│                                                                             │
│  RNN / LSTM / GRU                                                           │
│  ════════════════                                                           │
│  ✅ Séries temporais                                                       │
│  ✅ Texto curto a médio                                                    │
│  ✅ Sequências onde ordem importa                                          │
│  ❌ Sequências muito longas (>1000 passos)                                 │
│  ❌ Quando paralelização é crucial                                         │
│                                                                             │
│  Transformer                                                                │
│  ═══════════                                                                │
│  ✅ Texto (NLP) - dominante atualmente                                     │
│  ✅ Sequências longas                                                      │
│  ✅ Quando tem MUITOS dados                                                │
│  ❌ Dados pequenos (overfitting)                                           │
│  ❌ Quando recursos computacionais são limitados                           │
│                                                                             │
│  XGBoost / Random Forest                                                    │
│  ═══════════════════════                                                    │
│  ✅ Dados tabulares (MELHOR escolha geralmente)                            │
│  ✅ Quando interpretabilidade importa                                      │
│  ✅ Datasets médios                                                        │
│  ❌ Imagens, texto, séries temporais                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Tabela de Decisão Rápida

| Seu Dado | Primeira Escolha | Segunda Escolha | Evitar |
|----------|-----------------|-----------------|--------|
| Planilha/CSV | XGBoost | Random Forest | LSTM |
| Preços diários | LSTM | Transformer | Random Forest |
| Foto de produto | ResNet/EfficientNet | ViT | LSTM |
| Review de cliente | BERT/GPT | LSTM | CNN |
| Áudio de voz | Wav2Vec | CNN + LSTM | MLP |
| Rede social (grafo) | GNN | Node2Vec + MLP | CNN |

### 5.3 Trade-offs

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TRADE-OFFS                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                    SIMPLES ◄─────────────────────────► COMPLEXO            │
│                                                                             │
│  Regressão    MLP    Random    XGBoost    CNN    LSTM    Transformer       │
│  Linear              Forest                                                 │
│     │          │        │         │        │       │          │            │
│     ├──────────┴────────┴─────────┴────────┴───────┴──────────┤            │
│     │                                                          │            │
│     │  Mais interpretável              Menos interpretável     │            │
│     │  Menos dados necessários         Mais dados necessários  │            │
│     │  Treino rápido                   Treino lento            │            │
│     │  Menos capacidade                Mais capacidade         │            │
│     │                                                          │            │
│     └──────────────────────────────────────────────────────────┘            │
│                                                                             │
│  REGRA DE OURO:                                                             │
│  Comece simples. Só aumente complexidade se precisar.                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Quando Usar Cada Modelo

### 6.1 Fluxograma de Decisão

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    FLUXOGRAMA DE DECISÃO                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                         ┌─────────────────┐                                │
│                         │ Qual é seu dado?│                                │
│                         └────────┬────────┘                                │
│                                  │                                          │
│         ┌────────────────────────┼────────────────────────┐                │
│         │                        │                        │                 │
│         ▼                        ▼                        ▼                 │
│   ┌──────────┐            ┌──────────┐            ┌──────────┐            │
│   │ Tabular  │            │Sequencial│            │  Imagem  │            │
│   │(planilha)│            │ (tempo)  │            │          │            │
│   └────┬─────┘            └────┬─────┘            └────┬─────┘            │
│        │                       │                       │                   │
│        ▼                       ▼                       ▼                   │
│   ┌──────────┐            ┌──────────┐            ┌──────────┐            │
│   │ XGBoost  │            │  LSTM /  │            │   CNN    │            │
│   │ ou RF    │            │   GRU    │            │ (ResNet) │            │
│   └──────────┘            └────┬─────┘            └──────────┘            │
│                                │                                           │
│                    ┌───────────┴───────────┐                              │
│                    │                       │                               │
│                    ▼                       ▼                               │
│             ┌──────────┐            ┌──────────┐                          │
│             │Seq curta │            │Seq longa │                          │
│             │ (<100)   │            │ (>1000)  │                          │
│             └────┬─────┘            └────┬─────┘                          │
│                  │                       │                                 │
│                  ▼                       ▼                                 │
│             ┌──────────┐            ┌──────────┐                          │
│             │   LSTM   │            │Transformer│                         │
│             └──────────┘            └──────────┘                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 RNNs: Quando São Recomendáveis?

| Situação | RNN/LSTM é boa escolha? | Alternativa melhor |
|----------|------------------------|-------------------|
| Séries temporais curtas (<100 passos) | Sim (LSTM/GRU) | - |
| Séries temporais longas (>1000 passos) | Não | Transformer |
| Texto curto (tweets, reviews) | Sim | Transformer |
| Texto longo (documentos) | Não | Transformer |
| Previsão de ações (60 dias) | Sim (LSTM) | - |
| Dados em tempo real | Sim | - |
| Tradução de idiomas | Não | Transformer |

### 6.3 Por Que LSTM para Este Projeto?

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              POR QUE LSTM PARA PREVISÃO DE AÇÕES?                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. DADOS SÃO SÉRIES TEMPORAIS                                             │
│     • Preços ordenados no tempo                                            │
│     • A ordem dos dias importa                                             │
│     • Padrões de dias anteriores influenciam previsão                      │
│                                                                             │
│  2. DEPENDÊNCIAS DE MÉDIO PRAZO                                            │
│     • Usamos 60 dias (não é muito longo)                                   │
│     • LSTM lida bem com essa escala                                        │
│     • Transformer seria "overkill"                                         │
│                                                                             │
│  3. DATASET DE TAMANHO MÉDIO                                               │
│     • ~1500 dias de dados                                                  │
│     • Suficiente para LSTM                                                 │
│     • Transformer precisaria de mais dados                                 │
│                                                                             │
│  4. REQUISITO DO TECH CHALLENGE                                            │
│     • Especificado no enunciado                                            │
│     • Foco em aprender LSTM especificamente                                │
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  ALTERNATIVAS QUE TAMBÉM FUNCIONARIAM:                                     │
│  • GRU (mais rápido, resultado similar)                                    │
│  • Transformer (mais moderno, precisa mais dados)                          │
│  • Prophet/ARIMA (estatísticos, sem deep learning)                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Glossário de Termos

### 7.1 Termos Fundamentais

| Termo | Definição |
|-------|-----------|
| **Tensor** | Array multidimensional de números (generalização de vetor/matriz) |
| **Peso (Weight)** | Parâmetro aprendido que determina o comportamento da rede |
| **Bias** | Termo constante adicionado às operações (permite deslocamento) |
| **Época (Epoch)** | Uma passagem completa por todos os dados de treino |
| **Batch** | Subconjunto dos dados processado de uma vez |
| **Learning Rate** | Velocidade de atualização dos pesos (quão grandes são os passos) |
| **Loss** | Medida do erro do modelo (quanto menor, melhor) |
| **Gradiente** | Direção de maior aumento da loss (usado para otimização) |

### 7.2 Termos de Arquitetura

| Termo | Definição |
|-------|-----------|
| **Camada (Layer)** | Conjunto de neurônios que processam dados |
| **Neurônio** | Unidade básica que aplica peso + bias + ativação |
| **Ativação** | Função não-linear aplicada (ReLU, sigmoid, tanh) |
| **Forward Pass** | Passagem dos dados pela rede (entrada → saída) |
| **Backward Pass** | Cálculo dos gradientes (saída → entrada) |
| **Dropout** | Técnica de regularização que desliga neurônios aleatoriamente |

### 7.3 Termos Específicos de RNN/LSTM

| Termo | Definição |
|-------|-----------|
| **Hidden State (h)** | Estado oculto passado entre passos temporais |
| **Cell State (c)** | Memória de longo prazo no LSTM |
| **Gate** | Mecanismo que controla fluxo de informação (0 a 1) |
| **Forget Gate** | Decide o que esquecer da memória anterior |
| **Input Gate** | Decide o que adicionar à memória |
| **Output Gate** | Decide o que mostrar como saída |
| **Vanishing Gradient** | Problema onde gradientes ficam muito pequenos |

### 7.4 Tipos de Aprendizado

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TIPOS DE APRENDIZADO                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  SUPERVISIONADO                                                             │
│  ═══════════════                                                            │
│  • Dados têm "resposta certa" (labels)                                     │
│  • Modelo aprende a mapear entrada → saída                                 │
│  • Exemplo: preços → previsão (LSTM para ações)                           │
│                                                                             │
│  NÃO SUPERVISIONADO                                                         │
│  ══════════════════                                                         │
│  • Dados NÃO têm labels                                                    │
│  • Modelo descobre estruturas/padrões                                      │
│  • Exemplo: clustering de clientes, redução de dimensionalidade           │
│                                                                             │
│  SEMI-SUPERVISIONADO                                                        │
│  ═════════════════════                                                      │
│  • Poucos dados com labels + muitos sem                                    │
│  • Combina as duas abordagens                                              │
│  • Exemplo: classificar com poucos exemplos rotulados                      │
│                                                                             │
│  POR REFORÇO (Reinforcement Learning)                                       │
│  ═════════════════════════════════════                                      │
│  • Agente aprende por tentativa e erro                                     │
│  • Recebe recompensas/punições                                             │
│  • Exemplo: jogar games, robótica, trading                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Resumo Executivo

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         RESUMO GERAL                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  FAMÍLIAS PRINCIPAIS DE MODELOS:                                           │
│  • MLP: dados tabulares simples                                            │
│  • CNN: imagens, padrões espaciais                                         │
│  • RNN/LSTM: sequências, séries temporais                                  │
│  • Transformer: texto, sequências longas                                   │
│  • GNN: grafos, relações                                                   │
│                                                                             │
│  TIPOS DE DADOS:                                                            │
│  • Tabular (planilhas) → XGBoost, Random Forest                           │
│  • Série Temporal (ordem importa) → LSTM, GRU                             │
│  • Imagens → CNN (ResNet, EfficientNet)                                   │
│  • Texto → Transformer (BERT, GPT)                                        │
│                                                                             │
│  RNNs:                                                                      │
│  • Processam sequências mantendo memória                                   │
│  • LSTM resolve o problema de memória curta                                │
│  • Ideais para séries temporais de tamanho médio                          │
│                                                                             │
│  SÉRIE TEMPORAL:                                                            │
│  • Dados ordenados no tempo                                                │
│  • A ordem é fundamental                                                   │
│  • Componentes: tendência + sazonalidade + ciclo + ruído                  │
│                                                                             │
│  REGRA DE OURO:                                                             │
│  Comece simples. Só aumente complexidade se necessário.                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

*Documento gerado com base em sessão de estudo sobre Modelos de ML - Tech Challenge Fase 4*
