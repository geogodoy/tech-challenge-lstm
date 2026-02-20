# 📌 ETAPA 1: Configuração do Ambiente

## 📋 Resumo
| Item | Valor |
|------|-------|
| **Status** | ✅ Concluída |
| **Data** | 2026-02-17 |
| **Tempo Estimado** | 30 min |
| **Tempo Real** | ~15 min |

---

## 🎯 Objetivo
Preparar o ambiente de desenvolvimento com todas as dependências necessárias para o projeto de previsão de ações com LSTM.

---

## 🎓 Conexão com as Aulas

### Aula 03 - Arquiteturas de Redes Neurais Profundas
**Arquivo:** `docs - fase 4 /etapa 1 - redes neurais e deep learning/Aula 03 - Arquiteturas de Redes Neurais Profundas.txt`

> *"Vamos embarcar em uma exploração avançada das redes neurais... as Redes Neurais Recorrentes (RNNs), que são a espinha dorsal para o processamento de sequências e essenciais em tarefas como tradução automática e geração de texto."*

**Conceitos aplicados nesta etapa:**
- **PyTorch como framework**: O material da Aula 03 demonstra implementações em PyTorch (linhas 49-51):
  ```python
  import torch
  import torch.nn as nn
  import torch.optim as optim
  ```
- **Bibliotecas de suporte**: O setup inclui `scikit-learn` para pré-processamento e `matplotlib` para visualização, ferramentas essenciais mencionadas no contexto de engenharia de ML.

### Conexão Teórica
O setup do ambiente é a base para implementar os conceitos de:
- **RNNs e LSTMs** para séries temporais
- **Backpropagation Through Time (BPTT)** 
- **Regularização com Dropout**
- **Otimização com Adam**

---

## 📁 Estrutura de Pastas Criada

```
tech-challenge-lstm/
├── README.md              # Descrição do projeto
├── requirements.txt       # Dependências Python
├── .gitignore             # Arquivos ignorados pelo Git
├── Dockerfile             # Containerização
├── PROGRESS.md            # Acompanhamento do projeto
├── src/                   # Código-fonte
│   ├── data_collection.py # Etapa 2 - Coleta
│   ├── preprocessing.py   # Etapa 3 - Pré-processamento
│   ├── model.py           # Etapa 4 - Modelo LSTM
│   ├── train.py           # Etapa 5 - Treinamento
│   ├── evaluate.py        # Etapa 6 - Avaliação
│   └── app.py             # Etapa 8 - API FastAPI
├── models/                # Modelos salvos
├── data/                  # Dados baixados
├── notebooks/             # Jupyter notebooks
└── docs - fase 4 /        # Material de referência
```

---

## 📦 Dependências Instaladas

### Arquivo: `requirements.txt`

```txt
yfinance>=0.2.0       # Coleta de dados financeiros
pandas>=2.0.0         # Manipulação de dados
numpy>=1.24.0         # Operações numéricas
torch>=2.0.0          # Framework de Deep Learning
scikit-learn>=1.3.0   # Pré-processamento (MinMaxScaler)
fastapi>=0.100.0      # API REST
uvicorn>=0.23.0       # Servidor ASGI
matplotlib>=3.7.0     # Visualização
joblib>=1.3.0         # Serialização de objetos
```

### Por que estas bibliotecas?

| Biblioteca | Uso no Projeto | Referência na Aula |
|------------|----------------|-------------------|
| `torch` | Framework para LSTM | Aula 03: "implementações práticas em PyTorch" |
| `scikit-learn` | MinMaxScaler para normalização | Aula 02: "Normalização essencial para evitar que valores grandes dominem" |
| `pandas` | Manipulação de séries temporais | Dados tabulares organizados |
| `yfinance` | Coleta de dados de ações | Requisito do Tech Challenge |

---

## 💻 Comandos Executados

### 1. Criar ambiente virtual
```bash
python -m venv venv
```

### 2. Ativar ambiente
```bash
source venv/bin/activate  # Linux/Mac
```

### 3. Instalar dependências
```bash
pip install -r requirements.txt
```

### 4. Verificar instalação
```bash
python -c "import yfinance, pandas, numpy, torch, sklearn, fastapi; print('Ambiente OK!')"
# Resultado: ✅ Ambiente OK!
```

---

## 📊 Versões Instaladas

| Pacote | Versão |
|--------|--------|
| yfinance | 1.2.0 |
| pandas | 3.0.0 |
| numpy | 2.4.2 |
| torch | 2.10.0 |
| scikit-learn | 1.8.0 |
| fastapi | 0.129.0 |
| uvicorn | 0.41.0 |
| matplotlib | 3.10.8 |
| joblib | 1.5.3 |

---

## ✅ Checklist de Conclusão

- [x] Estrutura de pastas criada
- [x] Ambiente virtual configurado (`venv`)
- [x] Dependências instaladas
- [x] Verificação de imports bem-sucedida
- [x] `.gitignore` configurado
- [x] `Dockerfile` base criado

---

## 🔗 Próxima Etapa

**→ ETAPA 2: Coleta de Dados**
- Usar `yfinance` para baixar dados históricos
- Escolher ticker e período temporal
