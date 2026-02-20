# 📊 Progresso do Tech Challenge - LSTM Stock Predictor

Este arquivo rastreia o progresso do projeto em relação ao guia:
**`docs - fase 4 /tech challenge/Guia de Predição de Ações com LSTM- Tech Challenge 4`**

---

## 🗺️ Mapa Geral do Projeto

```text
[✅] ETAPA 1: Setup do Ambiente          ➔ CONCLUÍDA (2026-02-17)
[✅] ETAPA 2: Coleta de Dados (yfinance)  ➔ CONCLUÍDA (2026-02-17)
[✅] ETAPA 3: Pré-processamento           ➔ CONCLUÍDA (2026-02-17)
[✅] ETAPA 4: Modelo LSTM                 ➔ CONCLUÍDA (2026-02-17)
[✅] ETAPA 5: Treinamento                 ➔ CONCLUÍDA (2026-02-17)
[ ] ETAPA 6: Avaliação                   ➔ Pendente
[ ] ETAPA 7: Salvamento                  ➔ Pendente
[ ] ETAPA 8: API FastAPI                 ➔ Pendente
[ ] ETAPA 9: Docker e Monitoramento      ➔ Pendente
```

---

## ✅ ETAPA 1: Configuração do Ambiente

**📅 Data de Conclusão:** 2026-02-17  
**⏱️ Tempo Estimado:** 30 min | **Tempo Real:** ~15 min

### O que foi feito:

1. **Estrutura de pastas criada** (conforme `setup_github_projeto.md` linhas 18-34)
   ```
   tech-challenge-lstm/
   ├── README.md              ✅
   ├── requirements.txt       ✅
   ├── .gitignore             ✅
   ├── Dockerfile             ✅
   ├── src/                   ✅
   │   ├── data_collection.py ✅ (implementado)
   │   ├── preprocessing.py   ✅ (implementado)
   │   ├── model.py           ✅ (implementado)
   │   ├── train.py           ✅ (implementado)
   │   ├── evaluate.py        ✅ (pendente)
   │   └── app.py             ✅ (pendente)
   ├── models/                ✅
   ├── data/                  ✅
   ├── notebooks/             ✅
   └── docs - fase 4 /        ✅ (contexto)
   ```

2. **Ambiente virtual criado e ativado**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Dependências instaladas** (Guia linhas 71-81)
   
   | Pacote | Versão Instalada | Status |
   |--------|------------------|--------|
   | yfinance | 1.2.0 | ✅ |
   | pandas | 3.0.0 | ✅ |
   | numpy | 2.4.2 | ✅ |
   | torch | 2.10.0 | ✅ |
   | scikit-learn | 1.8.0 | ✅ |
   | fastapi | 0.129.0 | ✅ |
   | uvicorn | 0.41.0 | ✅ |
   | matplotlib | 3.10.8 | ✅ |
   | joblib | 1.5.3 | ✅ |

4. **Verificação do ambiente**
   ```bash
   python -c "import yfinance, pandas, numpy, torch, sklearn, fastapi; print('Ambiente OK!')"
   # Resultado: ✅ Ambiente OK!
   ```

### Arquivos criados/modificados:
- `requirements.txt` - Lista de dependências
- `.gitignore` - Arquivos a ignorar no Git
- `Dockerfile` - Configuração para containerização
- `README.md` - Descrição do projeto
- `venv/` - Ambiente virtual Python

---

## ✅ ETAPA 2: Coleta de Dados (yfinance)

**📅 Data de Conclusão:** 2026-02-17  
**⏱️ Tempo Estimado:** 30 min | **Tempo Real:** ~10 min

### O que foi feito:

1. **Implementado `src/data_collection.py`** com:
   - Função `download_stock_data()` para baixar dados do yfinance
   - Função `load_stock_data()` para carregar dados salvos
   - Configurações flexíveis (ticker, período)
   - Salvamento automático em CSV

2. **Configurações escolhidas:**
   - **Ticker:** `PETR4.SA` (Petrobras)
   - **Período:** 2018-01-01 até 2024-01-01
   - **Dados obtidos:** 1487 registros

3. **Estatísticas dos dados:**
   ```
   Shape: (1487, 5)
   Período real: 2018-01-02 até 2023-12-28
   Preço mínimo:  R$ 3.24
   Preço máximo:  R$ 27.38
   Preço médio:   R$ 10.17
   ```

4. **Arquivo gerado:**
   - `data/data_PETR4_SA.csv` - Dados históricos salvos

### Arquivos criados/modificados:
- `src/data_collection.py` - Script de coleta de dados
- `data/data_PETR4_SA.csv` - Dados baixados

### 🎉 Checkpoint: "Dados na mão!" ✅

---

## ✅ ETAPA 3: Pré-processamento

**📅 Data de Conclusão:** 2026-02-17  
**⏱️ Tempo Estimado:** 45 min | **Tempo Real:** ~10 min

### O que foi feito:

1. **Implementado `src/preprocessing.py`** com funções:
   - `normalize_data()` - Normaliza com MinMaxScaler (0-1)
   - `create_sequences()` - Cria janelas deslizantes
   - `train_test_split()` - Divide treino/teste
   - `to_tensors()` - Converte para tensores PyTorch
   - `preprocess_data()` - Pipeline completa

2. **Parâmetros configurados:**
   - **SEQ_LENGTH:** 60 dias (~3 meses)
   - **TRAIN_SPLIT:** 80% treino / 20% teste

3. **Resultado do pré-processamento:**
   ```
   Dados originais:  1487 registros
   Após sequências:  1427 amostras
   
   X_train: (1141, 60, 1) - 1141 amostras de treino
   y_train: (1141, 1)
   X_test:  (286, 60, 1)  - 286 amostras de teste
   y_test:  (286, 1)
   ```

4. **Artefatos salvos em `models/`:**
   - `scaler.pkl` - Scaler para reverter normalização
   - `config.pkl` - Configurações (seq_length, ticker, etc.)

### Arquivos criados/modificados:
- `src/preprocessing.py` - Script de pré-processamento
- `models/scaler.pkl` - Scaler serializado
- `models/config.pkl` - Configurações do modelo

### 🎉 Checkpoint: "Dados prontos!" ✅

---

## ✅ ETAPA 4: Modelo LSTM

**📅 Data de Conclusão:** 2026-02-17  
**⏱️ Tempo Estimado:** 45 min | **Tempo Real:** ~10 min

### O que foi feito:

1. **Implementado `src/model.py`** com:
   - Classe `StockLSTM` herdando de `nn.Module`
   - Função `create_model()` para instanciar
   - Função `count_parameters()` para debug

2. **Arquitetura definida:**
   ```
   StockLSTM(
     (lstm): LSTM(1, 50, num_layers=2, batch_first=True, dropout=0.2)
     (dropout): Dropout(p=0.2)
     (linear): Linear(in_features=50, out_features=1)
   )
   ```

3. **Hiperparâmetros configurados:**
   - `input_size`: 1 (apenas preço Close)
   - `hidden_size`: 50 (dimensão do estado oculto)
   - `num_layers`: 2 (LSTM empilhadas)
   - `dropout`: 0.2 (20% regularização)

4. **Estatísticas do modelo:**
   - **Parâmetros treináveis:** 31,051
   - **Dispositivo:** CPU (GPU se disponível)

### Arquivos criados/modificados:
- `src/model.py` - Definição da arquitetura LSTM

### 🎉 Checkpoint: "O cérebro nasceu!" ✅

---

## ✅ ETAPA 5: Treinamento

**📅 Data de Conclusão:** 2026-02-17  
**⏱️ Tempo Estimado:** 1h+ | **Tempo Real:** ~20 min (19s treino)

### O que foi feito:

1. **Implementado `src/train.py`** com funções:
   - `train_model()` - Loop de treinamento completo
   - `plot_training_history()` - Visualização de perdas
   - `save_trained_model()` - Salva modelo treinado

2. **Configuração do treinamento:**
   - **Dispositivo:** CPU
   - **Épocas:** 100
   - **Learning Rate:** 0.001
   - **Loss Function:** MSELoss
   - **Otimizador:** Adam

3. **Resultados do treinamento:**
   ```
   Tempo total:       18.7s (0.19s/época)
   Train Loss final:  0.001405
   Val Loss final:    0.002383
   Melhor Val Loss:   0.001508 (época 97)
   ```

4. **Artefatos gerados:**
   - `models/model_lstm.pth` - Modelo treinado
   - `models/training_history.png` - Gráfico de perdas

### Arquivos criados/modificados:
- `src/train.py` - Script de treinamento
- `models/model_lstm.pth` - Modelo serializado
- `models/training_history.png` - Visualização

### 🎉 Checkpoint: "Modelo treinado!" ✅

---

## 🔜 ETAPA 6: Avaliação

**📅 Status:** ⏳ PENDENTE

### Objetivos:
- [ ] Calcular MAE (Erro Médio Absoluto)
- [ ] Calcular RMSE (Raiz do Erro Quadrático Médio)
- [ ] Calcular MAPE (Erro Percentual Médio)
- [ ] Gerar gráfico de previsões vs valores reais

### Arquivo a implementar:
- `src/evaluate.py`

---

## 🔜 ETAPA 7: Salvamento

**📅 Status:** ⏳ PENDENTE

### Objetivos:
- [ ] Salvar modelo treinado (`model_lstm.pth`)
- [ ] Salvar scaler (`scaler.pkl`)
- [ ] Salvar configurações (`config.pkl`)

### Arquivos a gerar:
- `models/model_lstm.pth`
- `models/scaler.pkl`
- `models/config.pkl`

---

## 🔜 ETAPA 8: API FastAPI

**📅 Status:** ⏳ PENDENTE

### Objetivos:
- [ ] Criar endpoint `/predict`
- [ ] Criar endpoint `/health`
- [ ] Carregar modelo no startup
- [ ] Validar entrada e retornar previsão

### Arquivo a implementar:
- `src/app.py`

---

## 🔜 ETAPA 9: Docker e Monitoramento

**📅 Status:** ⏳ PENDENTE (parcial)

### Objetivos:
- [x] Criar Dockerfile (já existe estrutura base)
- [ ] Criar docker-compose.yml (opcional)
- [ ] Testar build e execução do container
- [ ] Configurar healthcheck

---

## 📋 Checklist de Entrega Final

- [x] Código-fonte no repositório Git
- [x] requirements.txt com versões
- [ ] README.md documentando o projeto (expandir)
- [ ] Modelo treinado (.pth) e scaler (.pkl)
- [x] Dockerfile funcional (estrutura base)
- [ ] Métricas de avaliação calculadas
- [ ] Vídeo demonstrando a API funcionando

---

## 📝 Notas e Observações

### Decisões tomadas:
- **Ticker escolhido:** `PETR4.SA` (Petrobras - ação brasileira)
- **Período de dados:** 2018-01-01 até 2024-01-01
- **SEQ_LENGTH:** 60 dias (recomendado no guia)
- **Split treino/teste:** 80%/20%

### Problemas encontrados:
- Nenhum até o momento

---

*Última atualização: 2026-02-17*
