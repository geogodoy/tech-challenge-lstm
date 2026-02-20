## Resumo Visual:

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                                    │
│  VOCÊ CONTROLA (pode mudar):                                                                      │
│  ══════════════════════════                                                                        │
│                                                                                                    │
│  O QUÊ                          │ ONDE              │ POR QUÊ                                     │
│  ───────────────────────────────┼───────────────────┼──────────────────────────────────────────── │
│  QUANTOS neurônios (50)         │ model.py L49      │ Balanceia aprendizado vs overfitting       │
│  QUANTAS camadas (2)            │ model.py L51      │ Captura padrões curto e médio prazo        │
│  QUANTO treina (100 epochs)     │ train.py L24      │ Suficiente p/ convergir sem overtreinar    │
│  QUANTOS dias olha (60)         │ preprocessing L23 │ ~3 meses captura tendências médio prazo    │
│  QUAL otimizador (Adam)         │ train.py L75      │ Converge rápido, ajusta LR automaticamente │
│  QUAL loss (MSE)                │ train.py L72      │ Penaliza erros grandes, bom p/ regressão   │
│  QUAL normalização (MinMax)     │ preprocessing L45 │ LSTM funciona melhor com valores 0-1       │
│  QUANTO dropout (0.2)           │ model.py L52      │ 20% regularização p/ evitar overfitting    │
│                                                                                                    │
│  ──────────────────────────────────────────────────────────────────────────────────────────────── │
│                                                                                                    │
│  PyTorch IMPLEMENTA (você usa):                                                                   │
│  ═══════════════════════════════                                                                   │
│  • COMO os portões funcionam              → dentro nn.LSTM                                        │
│  • COMO backprop calcula                  → loss.backward()                                       │
│  • COMO pesos atualizam                   → optimizer.step()                                      │
│                                                                                                    │
│  ──────────────────────────────────────────────────────────────────────────────────────────────── │
│                                                                                                    │
│  LSTM É DEFINIDO POR (não pode mudar):                                                           │
│  ═════════════════════════════════════                                                             │
│  • TER E USAR 3 portões                   → É o que é LSTM                                       │
│  • TER cell state + hidden state          → É o que é LSTM                                       │
│  • USAR fórmulas específicas              → É o que é LSTM                                       │
│  • PROCESSAR sequencialmente              → É o que é RNN                                        │
│                                                                                                    │
└────────────────────────────────────────────────────────────────────────────────────────────────────┘
```
