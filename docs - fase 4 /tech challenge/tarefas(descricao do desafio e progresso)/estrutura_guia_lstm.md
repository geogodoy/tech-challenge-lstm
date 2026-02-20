# Estrutura do Guia de Predição de Ações com LSTM - Tech Challenge 4

O guia está estruturado em **4 partes principais**, com a Parte 3 contendo **9 etapas detalhadas** de implementação.

---

## PARTE 1: MAPA DO TESOURO (Linhas 5-31)

Visão geral do projeto em formato de fluxograma visual, mostrando:
- A jornada linear do projeto
- Checkpoints de validação em cada fase
- Estimativas de tempo para cada etapa
- Dependências entre as fases

---

## PARTE 2: CONEXÃO TEORIA ↔ PRÁTICA (Linhas 33-46)

Tabela de mapeamento que relaciona cada conceito do Tech Challenge com as aulas correspondentes:
- LSTM e RNNs
- Backpropagation Through Time
- Dropout e Regularização
- Otimizador Adam
- Séries Temporais
- Deploy e Produção

---

## PARTE 3: GUIA PASSO A PASSO COMPLETO (Linhas 48-560)

Esta é a parte central, dividida em **9 etapas técnicas**:

| Etapa | Título | Linhas |
|-------|--------|--------|
| 1 | Configuração do Ambiente | 50-83 |
| 2 | Coleta de Dados (yfinance) | 85-121 |
| 3 | Pré-processamento para Séries Temporais | 123-183 |
| 4 | Construção do Modelo LSTM | 186-239 |
| 5 | Treinamento e Ajuste | 243-308 |
| 6 | Avaliação do Modelo | 311-359 |
| 7 | Salvamento do Modelo | 363-398 |
| 8 | Criação da API (FastAPI) | 401-498 |
| 9 | Docker e Monitoramento | 501-560 |

### Padrão de cada etapa:

Cada etapa segue um padrão consistente:
- **Conceitos das Aulas** - Teoria relacionada
- **Objetivo** - Meta clara da etapa
- **Checklist** (quando aplicável)
- **Código Completo Comentado**
- **Checkpoint** de validação
- **Armadilhas** (em algumas etapas)

---

## PARTE 4: RECURSOS DE APOIO (Linhas 563-596)

Material de suporte contendo:

### Glossário
Termos técnicos explicados:
- Epoch
- Hidden State
- MAE
- RMSE
- MAPE
- Scaler

### Cronograma Sugerido
6 sessões de 45-60 minutos com pausas para TDAH

### FAQ
Perguntas frequentes e soluções

### Checklist de Entrega
Lista final do que entregar no projeto

---

## Observações sobre a Estrutura

A estrutura foi pensada para ser **linear e progressiva**, com cada etapa dependendo da anterior, e inclui elementos de gamificação (checkpoints, recompensas) para manter a motivação durante a execução.
