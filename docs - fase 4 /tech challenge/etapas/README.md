# 📚 Documentação das Etapas - Tech Challenge LSTM

Este diretório contém a documentação detalhada de cada etapa executada no projeto de previsão de ações com LSTM.

---

## 🗂️ Índice de Etapas

| # | Etapa | Status | Documento |
|---|-------|--------|-----------|
| 1 | [Configuração do Ambiente](./ETAPA_01_Setup_Ambiente.md) | ✅ Concluída | Setup, dependências, estrutura |
| 2 | [Coleta de Dados](./ETAPA_02_Coleta_Dados.md) | ✅ Concluída | yfinance, PETR4.SA |
| 3 | [Pré-processamento](./ETAPA_03_Preprocessamento.md) | ✅ Concluída | Normalização, janelas temporais |
| 4 | [Modelo LSTM](./ETAPA_04_Modelo_LSTM.md) | ✅ Concluída | Arquitetura PyTorch |
| 5 | [Treinamento](./ETAPA_05_Treinamento.md) | ✅ Concluída | Loop, backpropagation, Adam |
| 6 | Avaliação | ✅ Concluída | MAE, RMSE, MAPE |
| 7 | Salvamento | ✅ Concluída | Serialização |
| 8 | [API FastAPI](./ETAPA_08_API_FastAPI.md) | ✅ Concluída | Endpoints REST, validação |
| 9 | Docker e Deploy | ⏳ Pendente | Containerização |

---

## 📖 Estrutura de Cada Documento

Cada documento de etapa segue a estrutura:

1. **📋 Resumo** - Tabela com status, data, referências
2. **🎯 Objetivo** - O que a etapa resolve
3. **🎓 Conexão com as Aulas** - Citações e conceitos teóricos
4. **📁 Código Implementado** - Arquivos e funções
5. **🔬 Análise Detalhada** - Explicação técnica
6. **✅ Checklist** - Itens concluídos
7. **🔗 Próxima Etapa** - Link para continuidade

---

## 🎓 Referências às Aulas

Os documentos fazem referência aos seguintes materiais:

| Material | Localização | Uso Principal |
|----------|-------------|---------------|
| Aula 02 - Teoria de Redes Neurais | `docs - fase 4 /etapa 1/` | Normalização, fundamentos |
| Aula 03 - Arquiteturas | `docs - fase 4 /etapa 1/` | LSTM, RNN, backpropagation |
| Aula 04 - Técnicas de Aplicação | `docs - fase 4 /etapa 1/` | Práticas de ML |
| Guia Tech Challenge | `docs - fase 4 /tech challenge/` | Roteiro completo |

---

## 📊 Progresso Geral

```
[███████████████████████████████░░░░] 89% (8/9 etapas)
```

### Timeline

```
Etapa 1 → Etapa 2 → Etapa 3 → Etapa 4 → Etapa 5 → Etapa 6 → Etapa 7 → Etapa 8 → [AGORA]
   ✅        ✅        ✅        ✅        ✅        ✅        ✅        ✅       ⏳
 Setup    Coleta   Preproc   Modelo   Treino   Avaliação  Salvam.    API     Docker
```

---

## 🔗 Links Rápidos

- [PROGRESS.md](../../PROGRESS.md) - Acompanhamento geral
- [Guia LSTM](../docs%20-%20fase%204%20/tech%20challenge/Guia%20de%20Predição%20de%20Ações%20com%20LSTM-%20Tech%20Challenge%204) - Roteiro original
- [README do Projeto](../../README.md) - Visão geral
