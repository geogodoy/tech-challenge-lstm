# 📌 ETAPA 6: Avaliação

## 📋 Resumo
| Item | Valor |
|------|-------|
| **Status** | ✅ Concluída |
| **Data** | 2026-02-19 |
| **Tempo Estimado** | 30 min |
| **Tempo Real** | ~20 min |

---

## 🎯 Objetivo
Avaliar a performance do modelo treinado calculando métricas quantitativas (MSE, RMSE, MAE, MAPE) e gerando visualizações comparativas entre previsões e valores reais.

---

## 🎓 Conexão com as Aulas

### Aula 02 - Métricas de Avaliação
**Conceitos fundamentais aplicados:**

#### Mean Squared Error (MSE)
> *"A função de perda mais comum para problemas de regressão. Penaliza erros grandes mais severamente que erros pequenos devido ao termo quadrático."*

#### Root Mean Squared Error (RMSE)
> *"A raiz do MSE, trazendo o erro de volta para a mesma unidade dos dados originais, facilitando a interpretação."*

#### Mean Absolute Error (MAE)
> *"Média dos valores absolutos dos erros. Menos sensível a outliers que o MSE/RMSE."*

#### Mean Absolute Percentage Error (MAPE)
> *"Expressa o erro como uma porcentagem do valor real, permitindo comparações independentes da escala."*

---

## 📁 Arquivo Implementado

### `src/evaluate.py`

#### Estrutura do Código

```python
# Linhas 1-5: Cabeçalho
# ═══════════════════════════════════════════════════════════════
# 📌 ETAPA 6: Avaliação do Modelo
# 🎯 Objetivo: Calcular métricas e avaliar performance
# 📍 Referência: GUIA_TREINAMENTO_E_AVALIACAO.md - Parte 2
# ═══════════════════════════════════════════════════════════════
```

#### Imports Necessários
```python
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend não-interativo para ambientes headless
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pathlib import Path

from model import StockLSTM
from preprocessing import preprocess_data
```

---

## 🔬 Funções Principais

### 1. `load_trained_model()` - Carregar Modelo

```python
def load_trained_model(model_path: Path = None) -> tuple:
    """
    Carrega o modelo treinado e suas configurações.
    
    O que faz:
    1. Carrega o checkpoint salvo (.pth)
    2. Recria a arquitetura LSTM com mesmas configurações
    3. Carrega os pesos treinados
    4. Coloca em modo de avaliação (model.eval())
    
    Returns:
        Tuple com (model, checkpoint)
    """
    checkpoint = torch.load(model_path, weights_only=False)
    
    # Recriar arquitetura com mesma configuração
    model = StockLSTM(**checkpoint['model_config'])
    
    # Carregar pesos treinados
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Modo avaliação (desativa dropout)
    model.eval()
    
    return model, checkpoint
```

**Por que `model.eval()`?**
- Desativa Dropout: Durante treino, dropout "desliga" 20% dos neurônios aleatoriamente. Na avaliação, queremos usar TODOS os neurônios.
- Desativa BatchNorm updates (se houver): Estatísticas não são atualizadas.
- Não afeta gradientes diretamente, mas `torch.no_grad()` sim.

---

### 2. `make_predictions()` - Gerar Previsões

```python
def make_predictions(model, X_test, scaler):
    """
    Faz previsões usando o modelo treinado.
    
    O que faz:
    1. Coloca modelo em modo eval()
    2. Desativa cálculo de gradientes (economiza memória)
    3. Executa forward pass com dados de teste
    4. Retorna previsões como numpy array
    """
    model.eval()
    with torch.no_grad():  # Não precisamos de gradientes na inferência
        predictions = model(X_test)
    
    predictions_np = predictions.numpy()
    return predictions_np
```

**Por que `torch.no_grad()`?**
- **Economia de memória:** Gradientes ocupam memória. Na avaliação, não precisamos deles.
- **Performance:** Cálculos mais rápidos sem construir o grafo computacional.
- **Segurança:** Garante que não modificamos pesos acidentalmente.

---

### 3. `calculate_metrics()` - Calcular Métricas

```python
def calculate_metrics(actual_reais, predictions_reais):
    """
    Calcula as 4 métricas principais de avaliação.
    
    IMPORTANTE: Usa valores em R$ (desnormalizados), não normalizados!
    """
    # MSE - Mean Squared Error
    # Fórmula: (1/n) × Σ(y_pred - y_real)²
    # Unidade: (R$)²
    mse = mean_squared_error(actual_reais, predictions_reais)
    
    # RMSE - Root Mean Squared Error
    # Fórmula: √MSE
    # Unidade: R$ (mesma dos dados)
    rmse = np.sqrt(mse)
    
    # MAE - Mean Absolute Error
    # Fórmula: (1/n) × Σ|y_pred - y_real|
    # Unidade: R$
    mae = mean_absolute_error(actual_reais, predictions_reais)
    
    # MAPE - Mean Absolute Percentage Error
    # Fórmula: (1/n) × Σ|((y_pred - y_real) / y_real)| × 100
    # Unidade: %
    mape = np.mean(np.abs((actual_reais - predictions_reais) / actual_reais)) * 100
    
    return {'mse': mse, 'rmse': rmse, 'mae': mae, 'mape': mape}
```

---

## 📊 Entendendo as Métricas

### Tabela Explicativa

| Métrica | Fórmula | Unidade | Interpretação |
|---------|---------|---------|---------------|
| **MSE** | `(1/n) × Σ(pred - real)²` | (R$)² | Penaliza erros grandes. Difícil interpretar. |
| **RMSE** | `√MSE` | R$ | Erro médio na mesma unidade dos dados. |
| **MAE** | `(1/n) × Σ\|pred - real\|` | R$ | Erro médio absoluto. Menos sensível a outliers. |
| **MAPE** | `(1/n) × Σ\|(pred - real)/real\| × 100` | % | Erro percentual. Independe da escala. |

### Por que usar MAPE como métrica principal?

```
Cenário 1: Ação vale R$ 10,00
  - Erro de R$ 0,50 = 5% de erro
  - "Errar R$ 0,50 num papel de R$ 10 é significativo"

Cenário 2: Ação vale R$ 100,00
  - Erro de R$ 0,50 = 0,5% de erro
  - "Errar R$ 0,50 num papel de R$ 100 é irrelevante"

O MAPE normaliza o erro pelo valor real, permitindo comparação justa.
```

### Escala de Qualidade (MAPE)

```
┌────────────────────────────────────────────────────────────────┐
│                    ESCALA DE QUALIDADE                         │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  MAPE < 5%     →  🟢 EXCELENTE  ← Nosso modelo (3.83%)        │
│  MAPE 5-10%    →  🟡 BOM                                       │
│  MAPE 10-20%   →  🟠 ACEITÁVEL                                 │
│  MAPE 20-50%   →  🔴 RAZOÁVEL                                  │
│  MAPE > 50%    →  ⚫ RUIM                                       │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Fluxo da Avaliação

```
┌─────────────────────────────────────────────────────────────────┐
│                    PIPELINE DE AVALIAÇÃO                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. CARREGAR MODELO                                             │
│  ─────────────────                                              │
│     checkpoint = torch.load('models/model_lstm.pth')            │
│     model = StockLSTM(**checkpoint['model_config'])             │
│     model.load_state_dict(checkpoint['model_state_dict'])       │
│     model.eval()                                                │
│                                                                 │
│                        │                                        │
│                        ▼                                        │
│                                                                 │
│  2. CARREGAR DADOS DE TESTE                                     │
│  ─────────────────────────                                      │
│     X_train, X_test, y_train, y_test, scaler = preprocess_data()│
│     → Usa o mesmo scaler do treino                              │
│     → 286 amostras de teste                                     │
│                                                                 │
│                        │                                        │
│                        ▼                                        │
│                                                                 │
│  3. FAZER PREVISÕES                                             │
│  ──────────────────                                             │
│     with torch.no_grad():                                       │
│         predictions = model(X_test)                             │
│     → Saída: valores normalizados (0-1)                         │
│                                                                 │
│                        │                                        │
│                        ▼                                        │
│                                                                 │
│  4. DESNORMALIZAR (VOLTAR PARA R$)                              │
│  ─────────────────────────────────                              │
│     predictions_reais = scaler.inverse_transform(predictions)   │
│     actual_reais = scaler.inverse_transform(y_test)             │
│     → Agora temos valores em R$                                 │
│                                                                 │
│                        │                                        │
│                        ▼                                        │
│                                                                 │
│  5. CALCULAR MÉTRICAS                                           │
│  ────────────────────                                           │
│     MSE  = mean_squared_error(actual, predictions)              │
│     RMSE = √MSE                                                 │
│     MAE  = mean_absolute_error(actual, predictions)             │
│     MAPE = mean(|((pred - real) / real)|) × 100                 │
│                                                                 │
│                        │                                        │
│                        ▼                                        │
│                                                                 │
│  6. GERAR VISUALIZAÇÕES                                         │
│  ──────────────────────                                         │
│     → Gráfico temporal: Previsto vs Real                        │
│     → Scatter plot: Correlação                                  │
│     → Salvar em models/predictions_vs_actual.png                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📈 Resultados da Avaliação

### Saída do Console

```
============================================================
📌 ETAPA 6: Avaliação do Modelo LSTM
============================================================

📥 Carregando modelo treinado...
   ✅ Modelo carregado!
   Train Loss final: 0.000693
   Val Loss final:   0.001367

📥 Carregando dados de teste...
   Amostras de teste: 286

🔮 Fazendo previsões...
   Previsões feitas: 286 amostras

📋 Exemplos de previsões:
      Previsto |        Real |        Erro
   ------------------------------------------
   R$    22.89 | R$    22.64 | R$     0.25
   R$    22.95 | R$    22.78 | R$     0.17
   R$    23.04 | R$    23.16 | R$     0.12
   R$    23.17 | R$    23.36 | R$     0.19
   R$    23.32 | R$    23.43 | R$     0.11

==================================================
📊 MÉTRICAS DE AVALIAÇÃO
==================================================
MSE  (Mean Squared Error):     0.7964
RMSE (Root Mean Squared Error): R$ 0.89
MAE  (Mean Absolute Error):     R$ 0.70
MAPE (Mean Absolute % Error):   3.83%
==================================================

🔍 DIAGNÓSTICO:
   ✅ Excelente! Modelo muito preciso.

📊 Gerando gráficos...
📊 Gráfico salvo em: models/predictions_vs_actual.png

============================================================
✅ AVALIAÇÃO CONCLUÍDA!
============================================================

📋 Resumo:
   RMSE: R$ 0.89 (erro médio em reais)
   MAPE: 3.83% (erro percentual médio)
   Status: EXCELENTE
```

### Interpretação dos Resultados

| Métrica | Valor | O que significa |
|---------|-------|-----------------|
| **MSE** | 0.7964 | Média dos erros ao quadrado. (R$)² |
| **RMSE** | R$ 0.89 | "Em média, o modelo erra R$ 0.89 por previsão" |
| **MAE** | R$ 0.70 | "Erro absoluto médio é R$ 0.70" |
| **MAPE** | 3.83% | "Em média, o modelo erra 3.83% do valor real" |

**Exemplo prático:**
```
Se a ação vale R$ 23.00:
  - Erro esperado: 3.83% × R$ 23.00 = R$ 0.88
  - Faixa de previsão: R$ 22.12 a R$ 23.88
```

---

## 📊 Gráficos Gerados

### `models/predictions_vs_actual.png`

O arquivo contém dois gráficos lado a lado:

```
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│  GRÁFICO 1: Comparação Temporal         GRÁFICO 2: Correlação    │
│  ─────────────────────────────────      ────────────────────     │
│                                                                  │
│  Preço │                                Previsto │    ⋰         │
│  (R$)  │    ___                         (R$)     │   ⋰⋰         │
│   24   │   /   \    Real (azul)           24     │  ⋰⋰          │
│        │  /     \                               │ ⋰⋰ *         │
│   23   │ /   ___ \  Previsto (vermelho)   23   │⋰⋰ * *        │
│        │/   /    \                             │⋰* * *         │
│   22   │   /      \_____                 22   │* * *           │
│        │                                      │ *              │
│        └────────────────────►                 └────────────────►│
│         Amostras (últimas 100)                    Real (R$)     │
│                                                                  │
│  • Linhas próximas = boa previsão       • Pontos na diagonal =  │
│  • Padrões similares                      previsão perfeita     │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 💾 Artefatos Gerados

### Arquivos Criados/Atualizados

| Arquivo | Descrição |
|---------|-----------|
| `src/evaluate.py` | Script completo de avaliação |
| `models/predictions_vs_actual.png` | Gráficos de comparação |

### Código para Reproduzir

```python
# Executar avaliação completa
from src.evaluate import evaluate_model

metrics, diagnosis = evaluate_model()

print(f"MAPE: {metrics['mape']:.2f}%")
print(f"Status: {diagnosis}")
```

---

## ⚠️ Pontos Importantes

### 1. Normalização vs Desnormalização

```python
# ERRADO - Calcular métricas com dados normalizados
mape = calculate_metrics(y_test_normalized, predictions_normalized)
# Resultado: MAPE sem significado prático (valores entre 0-1)

# CORRETO - Calcular métricas com dados em R$
predictions_reais = scaler.inverse_transform(predictions)
actual_reais = scaler.inverse_transform(y_test)
mape = calculate_metrics(actual_reais, predictions_reais)
# Resultado: MAPE em % do valor real da ação
```

### 2. Usar mesmo Scaler do Treino

```python
# ERRADO - Criar novo scaler para teste
new_scaler = MinMaxScaler()
new_scaler.fit(test_data)  # ERRADO! Diferentes min/max

# CORRETO - Usar mesmo scaler do treino
scaler = joblib.load('models/scaler.pkl')  # Mesmo do treino
predictions_reais = scaler.inverse_transform(predictions)
```

### 3. Modo de Avaliação

```python
# OBRIGATÓRIO antes de fazer previsões
model.eval()          # Desativa dropout
with torch.no_grad(): # Desativa gradientes
    predictions = model(X_test)
```

---

## ✅ Checklist de Conclusão

- [x] Modelo carregado corretamente
- [x] Dados de teste carregados (286 amostras)
- [x] Previsões geradas com torch.no_grad()
- [x] Valores desnormalizados para R$
- [x] MSE calculado: 0.7964
- [x] RMSE calculado: R$ 0.89
- [x] MAE calculado: R$ 0.70
- [x] MAPE calculado: 3.83%
- [x] Diagnóstico: EXCELENTE (< 5%)
- [x] Gráficos gerados e salvos
- [x] Comparação visual real vs previsto

---

## 🔗 Próxima Etapa

**→ ETAPA 7: Salvamento e Persistência** (Concluída)
- Salvar modelo em formato .pth
- Salvar scaler em formato .pkl
- Documentar como carregar para inferência
