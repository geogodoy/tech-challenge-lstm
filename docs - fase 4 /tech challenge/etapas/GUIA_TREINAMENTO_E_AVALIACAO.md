# Guia Passo a Passo: Treinamento e Avaliação do Modelo LSTM

> Guia prático e didático para treinar e avaliar o modelo de previsão de preços de ações

---

## Índice

1. [Visão Geral do Processo](#1-visão-geral-do-processo)
2. [Pré-requisitos](#2-pré-requisitos)
3. [Parte 1: Treinamento do Modelo](#parte-1-treinamento-do-modelo)
   - [Passo 1: Carregar os Dados](#passo-1-carregar-os-dados)
   - [Passo 2: Criar o Modelo](#passo-2-criar-o-modelo)
   - [Passo 3: Configurar Hiperparâmetros](#passo-3-configurar-hiperparâmetros)
   - [Passo 4: Executar o Treinamento](#passo-4-executar-o-treinamento)
   - [Passo 5: Monitorar o Progresso](#passo-5-monitorar-o-progresso)
   - [Passo 6: Salvar o Modelo](#passo-6-salvar-o-modelo)
4. [Parte 2: Avaliação do Modelo](#parte-2-avaliação-do-modelo)
   - [Passo 7: Carregar o Modelo Salvo](#passo-7-carregar-o-modelo-salvo)
   - [Passo 8: Fazer Previsões](#passo-8-fazer-previsões)
   - [Passo 9: Calcular Métricas](#passo-9-calcular-métricas)
   - [Passo 10: Interpretar Resultados](#passo-10-interpretar-resultados)
5. [Diagnóstico e Ajustes](#5-diagnóstico-e-ajustes)
6. [Checklist Final](#6-checklist-final)
7. [Glossário](#7-glossário)

---

## 1. Visão Geral do Processo

```
┌─────────────────────────────────────────────────────────────────┐
│                    FLUXO COMPLETO                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  TREINAMENTO                         AVALIAÇÃO                  │
│  ────────────                        ─────────                  │
│                                                                 │
│  1. Carregar dados                   7. Carregar modelo         │
│         ↓                                   ↓                   │
│  2. Criar modelo                     8. Fazer previsões         │
│         ↓                                   ↓                   │
│  3. Configurar hiperparâmetros       9. Calcular métricas       │
│         ↓                                   ↓                   │
│  4. Executar treinamento            10. Interpretar resultados  │
│         ↓                                   ↓                   │
│  5. Monitorar progresso                    │                    │
│         ↓                                  │                    │
│  6. Salvar modelo ─────────────────────────┘                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### O que é Treinamento?

É o processo onde o modelo "aprende" padrões nos dados. Funciona como ensinar uma criança a jogar dardos:

1. **Joga** (forward pass) → faz uma previsão
2. **Vê o erro** (loss) → mede a distância do alvo
3. **Entende** (backward) → descobre o que causou o erro
4. **Ajusta** (optimizer) → corrige a mira
5. **Repete** → até acertar consistentemente

### O que é Avaliação?

É verificar se o modelo realmente aprendeu, testando com dados que ele **nunca viu** durante o treinamento.

---

## 2. Pré-requisitos

### Arquivos necessários

```
tech-challenge-lstm/
├── src/
│   ├── data_collection.py    ✓ Coleta de dados
│   ├── preprocessing.py      ✓ Pré-processamento
│   ├── model.py              ✓ Arquitetura LSTM
│   └── train.py              ✓ Função de treinamento
├── data/
│   └── data_PETR4_SA.csv     ✓ Dados históricos
└── models/                    (será criado)
```

### Verificar se tudo está pronto

```python
# Execute no terminal ou em um script Python
from preprocessing import preprocess_data
from model import create_model

# Testar se os dados carregam
X_train, X_test, y_train, y_test, scaler = preprocess_data()
print(f"✅ Dados carregados: {X_train.shape[0]} amostras de treino")

# Testar se o modelo cria
model = create_model()
print(f"✅ Modelo criado: {sum(p.numel() for p in model.parameters())} parâmetros")
```

---

# PARTE 1: TREINAMENTO DO MODELO

---

## Passo 1: Carregar os Dados

### O que fazer

```python
from preprocessing import preprocess_data

# Carregar dados pré-processados
X_train, X_test, y_train, y_test, scaler = preprocess_data()
```

### O que você recebe

| Variável | Shape | O que é |
|----------|-------|---------|
| `X_train` | (N, 60, 1) | Sequências de 60 dias para treino |
| `X_test` | (M, 60, 1) | Sequências de 60 dias para validação |
| `y_train` | (N, 1) | Preço do dia 61 (treino) |
| `y_test` | (M, 1) | Preço do dia 61 (validação) |
| `scaler` | MinMaxScaler | Para reverter normalização depois |

### Verificar os dados

```python
print(f"Amostras de treino: {X_train.shape[0]}")
print(f"Amostras de teste:  {X_test.shape[0]}")
print(f"Tamanho da sequência: {X_train.shape[1]} dias")
print(f"Features por dia: {X_train.shape[2]}")
```

**Saída esperada:**
```
Amostras de treino: 1131
Amostras de teste:  283
Tamanho da sequência: 60 dias
Features por dia: 1
```

### Por que esse passo é importante?

- Os dados já estão **normalizados** (valores entre 0 e 1)
- Já estão **divididos** em treino (80%) e teste (20%)
- Já estão no **formato correto** para a LSTM (tensores PyTorch)

---

## Passo 2: Criar o Modelo

### O que fazer

```python
from model import create_model

# Criar modelo com configuração padrão
model = create_model()

# Ou criar com configuração personalizada
model = create_model(
    input_size=1,      # 1 feature (preço Close)
    hidden_size=50,    # 50 neurônios na camada oculta
    num_layers=2,      # 2 camadas LSTM empilhadas
    dropout=0.2        # 20% de dropout para regularização
)
```

### Verificar o modelo

```python
print(model)
print(f"\nTotal de parâmetros: {sum(p.numel() for p in model.parameters()):,}")
```

**Saída esperada:**
```
StockLSTM(
  (lstm): LSTM(1, 50, num_layers=2, batch_first=True, dropout=0.2)
  (dropout): Dropout(p=0.2, inplace=False)
  (linear): Linear(in_features=50, out_features=1, bias=True)
)

Total de parâmetros: 31,051
```

### Entendendo a arquitetura

```
┌─────────────────────────────────────────────────────────────────┐
│                    ARQUITETURA DO MODELO                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ENTRADA: 60 dias de preços normalizados                        │
│           shape: (batch, 60, 1)                                 │
│                      ↓                                          │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  LSTM Camada 1: processa a sequência temporal           │   │
│  │  - Aprende padrões de curto prazo                       │   │
│  └─────────────────────────────────────────────────────────┘   │
│                      ↓                                          │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  LSTM Camada 2: refina os padrões                       │   │
│  │  - Aprende padrões mais abstratos                       │   │
│  └─────────────────────────────────────────────────────────┘   │
│                      ↓                                          │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Dropout (20%): regularização                           │   │
│  │  - Evita overfitting                                    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                      ↓                                          │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Linear: 50 → 1 neurônio                                │   │
│  │  - Converte para preço único                            │   │
│  └─────────────────────────────────────────────────────────┘   │
│                      ↓                                          │
│  SAÍDA: 1 preço previsto (normalizado)                         │
│         shape: (batch, 1)                                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Passo 3: Configurar Hiperparâmetros

### Hiperparâmetros principais

```python
# Configurações de treinamento
EPOCHS = 100          # Número de passagens pelos dados
LEARNING_RATE = 0.001 # Velocidade de aprendizado
```

### Guia de escolha

| Hiperparâmetro | Valor Padrão | Quando Aumentar | Quando Diminuir |
|----------------|--------------|-----------------|-----------------|
| **EPOCHS** | 100 | Loss ainda caindo no final | Overfitting detectado |
| **LEARNING_RATE** | 0.001 | Convergência muito lenta | Loss oscilando muito |
| **hidden_size** | 50 | Underfitting | Overfitting |
| **num_layers** | 2 | Padrões muito complexos | Modelo muito lento |
| **dropout** | 0.2 | Overfitting | Underfitting |

### Tabela de referência para Learning Rate

```
┌──────────────────────────────────────────────────────────────┐
│              ESCALA DE LEARNING RATE                         │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  0.1     → Muito rápido (instável, não recomendado)          │
│  0.01    → Rápido                                            │
│  0.001   → Moderado (PADRÃO para Adam) ← RECOMENDADO         │
│  0.0001  → Lento (para ajuste fino)                          │
│  0.00001 → Muito lento                                       │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## Passo 4: Executar o Treinamento

### Opção A: Executar via script (mais simples)

```bash
cd tech-challenge-lstm
python src/train.py
```

### Opção B: Executar via código (mais controle)

```python
from train import train_model
from model import create_model
from preprocessing import preprocess_data

# 1. Carregar dados
X_train, X_test, y_train, y_test, scaler = preprocess_data()

# 2. Criar modelo
model = create_model()

# 3. Treinar
model, train_losses, val_losses = train_model(
    model=model,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    epochs=100,
    learning_rate=0.001
)

print("✅ Treinamento concluído!")
```

### O que acontece durante o treinamento

```
Para cada época (1 a 100):
│
├── FASE DE TREINO
│   ├── model.train()           → Ativa dropout
│   ├── outputs = model(X_train)→ Forward pass
│   ├── loss = MSE(outputs, y)  → Calcula erro
│   ├── optimizer.zero_grad()   → Limpa gradientes
│   ├── loss.backward()         → Calcula gradientes
│   └── optimizer.step()        → Atualiza pesos
│
└── FASE DE VALIDAÇÃO
    ├── model.eval()            → Desativa dropout
    ├── with torch.no_grad()    → Economiza memória
    └── val_loss = MSE(...)     → Mede generalização
```

---

## Passo 5: Monitorar o Progresso

### Saída esperada durante o treinamento

```
============================================================
🏋️ Iniciando treinamento...
============================================================

Epoch [ 10/100] | Train Loss: 0.002840 | Val Loss: 0.008114 | Time: 0.9s
Epoch [ 20/100] | Train Loss: 0.001384 | Val Loss: 0.005497 | Time: 1.8s
Epoch [ 30/100] | Train Loss: 0.001069 | Val Loss: 0.004543 | Time: 2.7s
Epoch [ 40/100] | Train Loss: 0.000932 | Val Loss: 0.003865 | Time: 3.6s
Epoch [ 50/100] | Train Loss: 0.000860 | Val Loss: 0.003440 | Time: 4.6s
Epoch [ 60/100] | Train Loss: 0.000806 | Val Loss: 0.003109 | Time: 5.5s
Epoch [ 70/100] | Train Loss: 0.000774 | Val Loss: 0.002867 | Time: 6.4s
Epoch [ 80/100] | Train Loss: 0.000748 | Val Loss: 0.002668 | Time: 7.3s
Epoch [ 90/100] | Train Loss: 0.000719 | Val Loss: 0.002488 | Time: 8.3s
Epoch [100/100] | Train Loss: 0.000699 | Val Loss: 0.002358 | Time: 9.2s

============================================================
✅ Treinamento concluído!
============================================================
```

### Como interpretar

```
┌─────────────────────────────────────────────────────────────────┐
│                    INTERPRETAÇÃO                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ✅ BOM: Train Loss diminuindo                                  │
│     → O modelo está aprendendo                                  │
│                                                                 │
│  ✅ BOM: Val Loss diminuindo                                    │
│     → O modelo está generalizando (não só decorando)            │
│                                                                 │
│  ✅ BOM: Val Loss > Train Loss (mas não muito)                  │
│     → Normal, pois validação usa dados nunca vistos             │
│                                                                 │
│  ⚠️ ATENÇÃO: Val Loss parou de diminuir                         │
│     → Pode estar começando a overfitar                          │
│                                                                 │
│  ❌ PROBLEMA: Val Loss subindo enquanto Train desce             │
│     → Overfitting! Pare o treino e ajuste                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Gráfico de diagnóstico

```
Loss
  │
  │    ╲
  │     ╲  Val Loss
  │      ╲
  │       ╲__________ ← Ideal: ambos descem e estabilizam
  │        ╲
  │         ╲ Train Loss
  │          ╲________
  │
  └────────────────────────────────────► Épocas
    0    20    40    60    80    100
```

---

## Passo 6: Salvar o Modelo

### O que é salvo automaticamente

```python
# O train.py salva automaticamente em models/model_lstm.pth
torch.save({
    'model_state_dict': model.state_dict(),
    'model_config': {
        'input_size': 1,
        'hidden_size': 50,
        'num_layers': 2,
        'dropout': 0.2
    },
    'train_losses': train_losses,
    'val_losses': val_losses,
    'final_train_loss': train_losses[-1],
    'final_val_loss': val_losses[-1]
}, 'models/model_lstm.pth')
```

### Verificar se foi salvo

```python
import os

if os.path.exists('models/model_lstm.pth'):
    print("✅ Modelo salvo com sucesso!")
    size = os.path.getsize('models/model_lstm.pth') / 1024
    print(f"   Tamanho: {size:.1f} KB")
else:
    print("❌ Modelo não foi salvo")
```

### Artefatos gerados

```
models/
├── model_lstm.pth          # Pesos do modelo treinado
└── training_history.png    # Gráfico de loss ao longo das épocas
```

---

# PARTE 2: AVALIAÇÃO DO MODELO

---

## Passo 7: Carregar o Modelo Salvo

### Código

```python
import torch
from model import StockLSTM

# 1. Carregar o checkpoint
checkpoint = torch.load('models/model_lstm.pth')

# 2. Recriar o modelo com a mesma arquitetura
model = StockLSTM(**checkpoint['model_config'])

# 3. Carregar os pesos treinados
model.load_state_dict(checkpoint['model_state_dict'])

# 4. Colocar em modo de avaliação
model.eval()

print("✅ Modelo carregado!")
print(f"   Train Loss final: {checkpoint['final_train_loss']:.6f}")
print(f"   Val Loss final:   {checkpoint['final_val_loss']:.6f}")
```

### Por que model.eval()?

```
model.train():
├── Dropout ATIVO (20% dos neurônios desligados)
└── Usado durante o treinamento

model.eval():
├── Dropout DESATIVO (100% dos neurônios ativos)
└── Usado para fazer previsões reais
```

---

## Passo 8: Fazer Previsões

### Código completo

```python
import torch
from preprocessing import preprocess_data

# Carregar dados de teste
X_train, X_test, y_train, y_test, scaler = preprocess_data()

# Fazer previsões
model.eval()
with torch.no_grad():
    predictions = model(X_test)

# Converter para numpy
predictions_np = predictions.numpy()
actual_np = y_test.numpy()

print(f"Previsões feitas: {len(predictions_np)} amostras")
```

### Reverter normalização (voltar para R$)

```python
import numpy as np

# Os dados estão normalizados (0-1), precisamos reverter para R$
predictions_reais = scaler.inverse_transform(predictions_np)
actual_reais = scaler.inverse_transform(actual_np)

print(f"\nExemplos de previsões:")
print(f"{'Previsto':>12} | {'Real':>12} | {'Erro':>12}")
print("-" * 42)
for i in range(5):
    prev = predictions_reais[i][0]
    real = actual_reais[i][0]
    erro = abs(prev - real)
    print(f"R$ {prev:>9.2f} | R$ {real:>9.2f} | R$ {erro:>9.2f}")
```

**Saída esperada:**
```
Exemplos de previsões:
    Previsto |         Real |         Erro
------------------------------------------
R$    25.43 | R$    25.12 | R$     0.31
R$    24.89 | R$    24.95 | R$     0.06
R$    26.01 | R$    26.45 | R$     0.44
R$    25.78 | R$    25.50 | R$     0.28
R$    24.56 | R$    24.80 | R$     0.24
```

---

## Passo 9: Calcular Métricas

### Métricas principais para regressão

```python
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Calcular métricas
mse = mean_squared_error(actual_reais, predictions_reais)
rmse = np.sqrt(mse)
mae = mean_absolute_error(actual_reais, predictions_reais)
mape = np.mean(np.abs((actual_reais - predictions_reais) / actual_reais)) * 100

print("\n" + "=" * 50)
print("📊 MÉTRICAS DE AVALIAÇÃO")
print("=" * 50)
print(f"MSE  (Mean Squared Error):     {mse:.4f}")
print(f"RMSE (Root Mean Squared Error): R$ {rmse:.2f}")
print(f"MAE  (Mean Absolute Error):     R$ {mae:.2f}")
print(f"MAPE (Mean Absolute % Error):   {mape:.2f}%")
print("=" * 50)
```

### O que cada métrica significa

| Métrica | Fórmula | Interpretação |
|---------|---------|---------------|
| **MSE** | média(erro²) | Penaliza erros grandes, em unidades² |
| **RMSE** | √MSE | Erro médio na mesma unidade (R$) |
| **MAE** | média(\|erro\|) | Erro médio absoluto (R$) |
| **MAPE** | média(\|erro/real\|) × 100 | Erro percentual médio (%) |

### Exemplo de interpretação

```
RMSE = R$ 1.17 significa:
└─ Em média, o modelo erra cerca de R$ 1.17 na previsão

MAPE = 4.5% significa:
└─ Em média, o modelo erra cerca de 4.5% do valor real
```

---

## Passo 10: Interpretar Resultados

### Tabela de referência para MAPE

| MAPE | Qualidade da Previsão |
|------|----------------------|
| < 5% | Excelente |
| 5-10% | Boa |
| 10-20% | Aceitável |
| 20-50% | Razoável |
| > 50% | Ruim |

### Visualizar previsões vs reais

```python
import matplotlib.pyplot as plt

# Plotar comparação
plt.figure(figsize=(12, 6))

# Últimas 100 amostras para visualização
n_samples = 100
plt.plot(actual_reais[-n_samples:], label='Real', color='blue', linewidth=2)
plt.plot(predictions_reais[-n_samples:], label='Previsto', color='red', 
         linewidth=2, linestyle='--')

plt.title('Previsão vs Valor Real (Últimas 100 amostras)')
plt.xlabel('Amostra')
plt.ylabel('Preço (R$)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('models/predictions_vs_actual.png', dpi=150)
plt.show()

print("✅ Gráfico salvo em models/predictions_vs_actual.png")
```

### Diagnóstico final

```
┌─────────────────────────────────────────────────────────────────┐
│                    DIAGNÓSTICO FINAL                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  SE MAPE < 10% e RMSE < R$ 2.00:                                │
│  ✅ Modelo está bom! Pode usar para previsões                   │
│                                                                 │
│  SE MAPE entre 10-20%:                                          │
│  ⚠️ Modelo aceitável, considere ajustes:                        │
│     - Mais épocas de treinamento                                │
│     - Ajustar hidden_size                                       │
│     - Coletar mais dados                                        │
│                                                                 │
│  SE MAPE > 20%:                                                 │
│  ❌ Modelo precisa de melhorias:                                │
│     - Revisar pré-processamento                                 │
│     - Tentar arquitetura diferente                              │
│     - Verificar qualidade dos dados                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Diagnóstico e Ajustes

### Se o modelo não está bom, o que ajustar?

| Problema | Sintoma | Solução |
|----------|---------|---------|
| **Underfitting** | Train e Val loss altos | ↑ hidden_size, ↑ epochs, ↑ num_layers |
| **Overfitting** | Train baixo, Val alto | ↑ dropout, ↓ epochs, early stopping |
| **Convergência lenta** | Loss demora a cair | ↑ learning_rate (com cuidado) |
| **Loss oscilando** | Sobe e desce muito | ↓ learning_rate |
| **Loss estagnado** | Para de diminuir | ↑ learning_rate, mudar arquitetura |

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

5. Epochs
   └─ Comece com 100, aumente se loss ainda estiver caindo
```

### Exemplo de experimentos

```
Experimento 1: epochs=100, lr=0.001, hidden=50     → MAPE: 8.5%
Experimento 2: epochs=150, lr=0.001, hidden=50     → MAPE: 7.2% ✓
Experimento 3: epochs=150, lr=0.0005, hidden=50    → MAPE: 6.8% ✓
Experimento 4: epochs=150, lr=0.0005, hidden=100   → MAPE: 7.5% (piorou)
Experimento 5: epochs=150, lr=0.0005, hidden=50, dropout=0.3 → MAPE: 6.1% ✓

Melhor configuração: Experimento 5
```

---

## 6. Checklist Final

### Antes do Treinamento

- [ ] Dados coletados e salvos em CSV
- [ ] Dados pré-processados (normalizados, sequências criadas)
- [ ] Dados divididos em treino/teste
- [ ] Modelo criado com arquitetura definida
- [ ] Hiperparâmetros escolhidos

### Durante o Treinamento

- [ ] Train loss está diminuindo
- [ ] Val loss está diminuindo
- [ ] Não há sinais de overfitting
- [ ] Tempo de execução está razoável

### Após o Treinamento

- [ ] Modelo salvo em arquivo .pth
- [ ] Gráfico de histórico gerado
- [ ] Métricas calculadas (RMSE, MAE, MAPE)
- [ ] Resultados documentados

### Avaliação

- [ ] Previsões feitas nos dados de teste
- [ ] Normalização revertida para R$
- [ ] Métricas interpretadas
- [ ] Gráfico de previsões vs reais gerado
- [ ] Decisão: modelo está bom ou precisa ajustar?

---

## 7. Glossário

| Termo | Definição |
|-------|-----------|
| **Época (Epoch)** | Uma passagem completa por todos os dados de treino |
| **Loss** | Medida do erro/imprecisão do modelo |
| **MSE** | Mean Squared Error - erro quadrático médio |
| **RMSE** | Raiz do MSE - erro na mesma unidade dos dados |
| **MAE** | Mean Absolute Error - erro absoluto médio |
| **MAPE** | Mean Absolute Percentage Error - erro percentual |
| **Forward Pass** | Dados entram no modelo e saem como previsão |
| **Backward Pass** | Cálculo de gradientes (backpropagation) |
| **Gradiente** | Indica quanto cada peso contribuiu para o erro |
| **Learning Rate** | Tamanho do passo na atualização dos pesos |
| **Adam** | Otimizador adaptativo (ajusta lr por parâmetro) |
| **Overfitting** | Modelo decorou os dados ao invés de aprender |
| **Underfitting** | Modelo não aprendeu o suficiente |
| **Dropout** | Regularização que desliga neurônios aleatoriamente |
| **Scaler** | Objeto que normaliza/desnormaliza os dados |
| **Checkpoint** | Arquivo com pesos salvos do modelo |

---

## Script Completo de Execução

```python
"""
Script completo para treinar e avaliar o modelo LSTM
Execute: python treinar_e_avaliar.py
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

from preprocessing import preprocess_data
from model import create_model, StockLSTM
from train import train_model

# ============================================================
# PARTE 1: TREINAMENTO
# ============================================================

print("=" * 60)
print("ETAPA 1: Carregando dados...")
print("=" * 60)
X_train, X_test, y_train, y_test, scaler = preprocess_data()
print(f"✅ Treino: {X_train.shape[0]} amostras")
print(f"✅ Teste:  {X_test.shape[0]} amostras")

print("\n" + "=" * 60)
print("ETAPA 2: Criando modelo...")
print("=" * 60)
model = create_model()
n_params = sum(p.numel() for p in model.parameters())
print(f"✅ Modelo criado: {n_params:,} parâmetros")

print("\n" + "=" * 60)
print("ETAPA 3: Treinando...")
print("=" * 60)
model, train_losses, val_losses = train_model(
    model=model,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    epochs=100,
    learning_rate=0.001
)

# ============================================================
# PARTE 2: AVALIAÇÃO
# ============================================================

print("\n" + "=" * 60)
print("ETAPA 4: Avaliando...")
print("=" * 60)

model.eval()
with torch.no_grad():
    predictions = model(X_test)

# Reverter normalização
predictions_reais = scaler.inverse_transform(predictions.numpy())
actual_reais = scaler.inverse_transform(y_test.numpy())

# Calcular métricas
mse = mean_squared_error(actual_reais, predictions_reais)
rmse = np.sqrt(mse)
mae = mean_absolute_error(actual_reais, predictions_reais)
mape = np.mean(np.abs((actual_reais - predictions_reais) / actual_reais)) * 100

print(f"\n📊 MÉTRICAS:")
print(f"   RMSE: R$ {rmse:.2f}")
print(f"   MAE:  R$ {mae:.2f}")
print(f"   MAPE: {mape:.2f}%")

# Diagnóstico
print(f"\n🔍 DIAGNÓSTICO:")
if mape < 5:
    print("   ✅ Excelente! Modelo muito preciso.")
elif mape < 10:
    print("   ✅ Bom! Modelo com boa precisão.")
elif mape < 20:
    print("   ⚠️ Aceitável. Considere ajustes para melhorar.")
else:
    print("   ❌ Precisa melhorar. Revise hiperparâmetros e dados.")

print("\n" + "=" * 60)
print("✅ PROCESSO CONCLUÍDO!")
print("=" * 60)
```

---

*Guia criado para auxiliar no treinamento e avaliação do modelo LSTM de previsão de preços de ações.*
