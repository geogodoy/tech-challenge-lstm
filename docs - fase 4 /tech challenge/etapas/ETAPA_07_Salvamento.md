# 📌 ETAPA 7: Salvamento e Persistência

## 📋 Resumo
| Item | Valor |
|------|-------|
| **Status** | ✅ Concluída |
| **Data** | 2026-02-19 |
| **Tempo Estimado** | 15 min |
| **Tempo Real** | ~10 min |

---

## 🎯 Objetivo
Persistir o modelo treinado e todos os artefatos necessários para que possam ser carregados posteriormente para inferência (API) ou re-treinamento, sem perder a configuração original.

---

## 🎓 Conexão com as Aulas

### Por que Salvar o Modelo?

> *"Um modelo de machine learning sem persistência é como um programa que perde todo o estado quando fechado. O treinamento pode levar horas ou dias - seria desperdício re-treinar toda vez."*

### Conceitos de Serialização

> *"Serialização é o processo de converter objetos em memória para um formato que pode ser armazenado em disco ou transmitido pela rede."*

---

## 📁 Arquivos de Persistência

### Estrutura de Artefatos

```
models/
├── model_lstm.pth          # Pesos e config do modelo (PyTorch)
├── scaler.pkl              # Normalizador (scikit-learn)
├── config.pkl              # Configurações do pipeline
├── training_history.png    # Gráfico de loss
└── predictions_vs_actual.png # Gráfico de previsões
```

---

## 💾 1. Salvamento do Modelo PyTorch

### O que Salvar?

```python
# Apenas o state_dict (RECOMENDADO)
torch.save(model.state_dict(), 'model.pth')

# OU modelo completo com metadados (MELHOR para deploy)
torch.save({
    'model_state_dict': model.state_dict(),
    'model_config': {...},
    'training_info': {...}
}, 'model.pth')
```

### Por que usar `state_dict()` e não o modelo inteiro?

```
Opção 1: torch.save(model, 'model.pth')
─────────────────────────────────────────
❌ Salva o modelo inteiro
❌ Depende da estrutura exata das classes
❌ Problemas se mover arquivos ou renomear classes
❌ Maior tamanho de arquivo

Opção 2: torch.save(model.state_dict(), 'model.pth')
─────────────────────────────────────────────────────
✅ Salva apenas os pesos (parâmetros)
✅ Independente da localização do código
✅ Mais flexível para modificações
✅ Menor tamanho de arquivo
✅ Padrão recomendado pelo PyTorch
```

### Nosso Checkpoint Completo

```python
# Arquivo: src/train.py - Função save_trained_model()

def save_trained_model(model, train_losses, val_losses, save_path=None):
    """
    Salva o modelo treinado com todas as informações necessárias.
    """
    if save_path is None:
        save_path = MODELS_DIR / "model_lstm.pth"
    
    checkpoint = {
        # 1. PESOS DO MODELO
        # Dicionário com todos os parâmetros treináveis
        # Formato: {'lstm.weight_ih_l0': tensor(...), 'lstm.weight_hh_l0': tensor(...), ...}
        'model_state_dict': model.state_dict(),
        
        # 2. CONFIGURAÇÃO DA ARQUITETURA
        # Necessário para recriar o modelo com mesma estrutura
        'model_config': {
            'input_size': 1,       # Features de entrada
            'hidden_size': 100,    # Neurônios LSTM (otimizado)
            'num_layers': 2,       # Camadas LSTM
            'dropout': 0.2         # Taxa de regularização
        },
        
        # 3. HISTÓRICO DE TREINAMENTO
        # Útil para análise e debugging
        'train_losses': train_losses,   # Lista de losses por época
        'val_losses': val_losses,       # Lista de val losses por época
        
        # 4. MÉTRICAS FINAIS
        # Snapshot do estado final
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1],
        
        # 5. METADADOS (opcional, mas útil)
        'epochs_trained': len(train_losses),
        'best_val_loss': min(val_losses),
        'best_epoch': val_losses.index(min(val_losses)) + 1
    }
    
    torch.save(checkpoint, save_path)
    print(f"💾 Modelo salvo em: {save_path}")
```

### Conteúdo do `model_state_dict`

```python
# O que está dentro de model.state_dict():
{
    'lstm.weight_ih_l0': tensor([...]),  # Pesos input-hidden camada 0
    'lstm.weight_hh_l0': tensor([...]),  # Pesos hidden-hidden camada 0
    'lstm.bias_ih_l0': tensor([...]),    # Bias input-hidden camada 0
    'lstm.bias_hh_l0': tensor([...]),    # Bias hidden-hidden camada 0
    'lstm.weight_ih_l1': tensor([...]),  # Pesos input-hidden camada 1
    'lstm.weight_hh_l1': tensor([...]),  # Pesos hidden-hidden camada 1
    'lstm.bias_ih_l1': tensor([...]),    # Bias input-hidden camada 1
    'lstm.bias_hh_l1': tensor([...]),    # Bias hidden-hidden camada 1
    'fc.weight': tensor([...]),          # Pesos camada linear
    'fc.bias': tensor([...])             # Bias camada linear
}

# Exemplo de dimensões para hidden_size=100:
lstm.weight_ih_l0: shape (400, 1)     # 4 gates × 100 hidden × 1 input
lstm.weight_hh_l0: shape (400, 100)   # 4 gates × 100 hidden × 100 hidden
```

---

## 💾 2. Salvamento do Scaler

### Por que Salvar o Scaler?

```
PROBLEMA:
─────────
Durante o treino, normalizamos os dados usando MinMaxScaler.
O scaler "aprendeu" os valores mínimo e máximo dos dados de treino.

Treino: preço min = R$ 3.24, max = R$ 27.38
Scaler transforma: R$ 15.31 → 0.5 (meio da escala)

Se criarmos um NOVO scaler para inferência:
Inferência: preço atual min = R$ 22.00, max = R$ 25.00
Novo scaler transforma: R$ 23.50 → 0.5 (DIFERENTE!)

RESULTADO: Previsões completamente erradas!

SOLUÇÃO:
────────
Salvar o scaler original e usá-lo na inferência.
```

### Código de Salvamento

```python
# Arquivo: src/preprocessing.py - Função preprocess_data()

import joblib  # Biblioteca para serialização de objetos Python

def preprocess_data(save_scaler=True):
    """
    Pré-processa os dados e opcionalmente salva o scaler.
    """
    # Normalização
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    
    # Salvar scaler para uso posterior
    if save_scaler:
        scaler_path = MODELS_DIR / "scaler.pkl"
        joblib.dump(scaler, scaler_path)
        print(f"📐 Scaler salvo em: {scaler_path}")
    
    return X_train, X_test, y_train, y_test, scaler
```

### Conteúdo do Scaler Salvo

```python
# O que está dentro do scaler:
{
    'feature_range': (0, 1),
    'data_min_': array([3.24]),      # Preço mínimo nos dados de treino
    'data_max_': array([27.38]),     # Preço máximo nos dados de treino
    'scale_': array([0.04144...]),   # 1 / (max - min)
    'min_': array([-0.1343...]),     # -min * scale
}

# Fórmula de normalização:
# valor_normalizado = (valor_original - data_min) / (data_max - data_min)

# Fórmula de desnormalização:
# valor_original = valor_normalizado * (data_max - data_min) + data_min
```

---

## 💾 3. Salvamento de Configurações

### Arquivo `config.pkl`

```python
# Salvar configurações do pipeline
import joblib

config = {
    'seq_length': 60,           # Janela temporal (dias)
    'train_split': 0.8,         # 80% treino, 20% teste
    'feature_column': 'Close',  # Coluna usada
    'ticker': 'PETR4.SA',       # Ativo
    'date_range': {
        'start': '2018-01-01',
        'end': '2024-01-01'
    }
}

joblib.dump(config, 'models/config.pkl')
```

---

## 🔄 Carregamento para Inferência

### Pipeline Completo de Carregamento

```python
def load_for_inference():
    """
    Carrega todos os artefatos necessários para fazer previsões.
    
    Returns:
        model: Modelo LSTM pronto para inferência
        scaler: Normalizador para transformar novos dados
        config: Configurações do pipeline
    """
    # 1. CARREGAR MODELO
    # ──────────────────
    checkpoint = torch.load('models/model_lstm.pth', weights_only=False)
    
    # Recriar arquitetura (PRECISA da classe StockLSTM)
    from model import StockLSTM
    model = StockLSTM(**checkpoint['model_config'])
    
    # Carregar pesos
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Modo avaliação (IMPORTANTE!)
    model.eval()
    
    # 2. CARREGAR SCALER
    # ──────────────────
    import joblib
    scaler = joblib.load('models/scaler.pkl')
    
    # 3. CARREGAR CONFIG (opcional)
    # ─────────────────────────────
    config = joblib.load('models/config.pkl')
    
    return model, scaler, config
```

### Exemplo de Uso para Previsão

```python
# Carregar artefatos
model, scaler, config = load_for_inference()

# Novos dados (últimos 60 dias)
new_data = get_last_60_days('PETR4.SA')  # Shape: (60, 1)

# Normalizar com o MESMO scaler do treino
new_data_normalized = scaler.transform(new_data)

# Converter para tensor
X_new = torch.FloatTensor(new_data_normalized).unsqueeze(0)  # Shape: (1, 60, 1)

# Fazer previsão
with torch.no_grad():
    prediction_normalized = model(X_new)

# Desnormalizar para R$
prediction_reais = scaler.inverse_transform(prediction_normalized.numpy())

print(f"Previsão: R$ {prediction_reais[0][0]:.2f}")
```

---

## 📊 Tamanhos dos Arquivos

| Arquivo | Tamanho | Conteúdo |
|---------|---------|----------|
| `model_lstm.pth` | ~500 KB | Pesos (~121k parâmetros × 4 bytes) + metadados |
| `scaler.pkl` | ~1 KB | Parâmetros de normalização |
| `config.pkl` | ~1 KB | Configurações do pipeline |
| `training_history.png` | ~150 KB | Gráfico de loss |
| `predictions_vs_actual.png` | ~200 KB | Gráficos de avaliação |
| **TOTAL** | ~850 KB | Tudo necessário para deploy |

---

## ⚠️ Boas Práticas de Salvamento

### 1. Versionar Modelos

```python
# Incluir versão no nome do arquivo
torch.save(checkpoint, f'models/model_lstm_v{version}.pth')

# Ou usar timestamp
from datetime import datetime
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
torch.save(checkpoint, f'models/model_lstm_{timestamp}.pth')
```

### 2. Validar Carregamento

```python
# Sempre testar se o modelo carrega corretamente
def validate_model_loading(model_path):
    """Verifica se o modelo carrega sem erros."""
    try:
        checkpoint = torch.load(model_path, weights_only=False)
        model = StockLSTM(**checkpoint['model_config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Teste com input dummy
        dummy_input = torch.randn(1, 60, 1)
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"✅ Modelo validado! Output shape: {output.shape}")
        return True
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")
        return False
```

### 3. Não Salvar Dados Sensíveis

```python
# EVITAR salvar dados de treino no checkpoint
checkpoint = {
    'model_state_dict': model.state_dict(),
    'model_config': {...},
    # NÃO INCLUIR: 'X_train': X_train, 'y_train': y_train
}
```

### 4. Usar `weights_only` Quando Possível

```python
# Mais seguro (evita execução de código arbitrário)
checkpoint = torch.load('model.pth', weights_only=True)

# Necessário se salvou objetos customizados
checkpoint = torch.load('model.pth', weights_only=False)
```

---

## 🔄 Fluxo Completo

```
┌─────────────────────────────────────────────────────────────────┐
│                    FLUXO DE PERSISTÊNCIA                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  TREINAMENTO                          INFERÊNCIA                │
│  ────────────                         ──────────                │
│                                                                 │
│  ┌─────────────┐                     ┌─────────────┐            │
│  │   Treinar   │                     │  Carregar   │            │
│  │   Modelo    │                     │   Modelo    │            │
│  └──────┬──────┘                     └──────┬──────┘            │
│         │                                   │                   │
│         ▼                                   │                   │
│  ┌─────────────┐      model.pth      ┌──────┴──────┐            │
│  │    Salvar   │ ──────────────────► │ load_state  │            │
│  │ state_dict  │                     │    _dict    │            │
│  └──────┬──────┘                     └──────┬──────┘            │
│         │                                   │                   │
│         ▼                                   ▼                   │
│  ┌─────────────┐      scaler.pkl     ┌─────────────┐            │
│  │   Salvar    │ ──────────────────► │  Carregar   │            │
│  │   Scaler    │                     │   Scaler    │            │
│  └──────┬──────┘                     └──────┬──────┘            │
│         │                                   │                   │
│         ▼                                   ▼                   │
│  ┌─────────────┐      config.pkl     ┌─────────────┐            │
│  │   Salvar    │ ──────────────────► │  Carregar   │            │
│  │   Config    │                     │   Config    │            │
│  └─────────────┘                     └──────┬──────┘            │
│                                             │                   │
│                                             ▼                   │
│                                      ┌─────────────┐            │
│                                      │   Prever    │            │
│                                      │  (API/CLI)  │            │
│                                      └─────────────┘            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## ✅ Checklist de Conclusão

### Modelo
- [x] `model_state_dict` salvo
- [x] `model_config` incluído no checkpoint
- [x] Histórico de losses salvo
- [x] Métricas finais incluídas
- [x] Modelo carrega corretamente
- [x] Previsão funciona após carregamento

### Scaler
- [x] Scaler salvo em formato .pkl
- [x] Carrega corretamente com joblib
- [x] `inverse_transform` funciona

### Configurações
- [x] `seq_length` documentado (60)
- [x] `train_split` documentado (0.8)
- [x] Parâmetros de data documentados

### Validação
- [x] Modelo validado após carregamento
- [x] Output shape correto
- [x] Previsões fazem sentido

---

## 🔗 Próxima Etapa

**→ ETAPA 8: API FastAPI**
- Criar endpoints `/predict` e `/health`
- Carregar modelo na inicialização
- Receber dados via JSON
- Retornar previsões em R$
