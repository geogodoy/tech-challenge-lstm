# 🎬 Roteiro de Apresentação - Tech Challenge LSTM

> **Duração Total:** 8 minutos  
> **Apresentadores:** Geovana e Cleiton  
> **Formato:** Gravação de tela + áudio (sem slides)  
> **Projeto:** Predição de Preços de Ações com LSTM

---

## 📋 Divisão de Tempo

| Parte | Apresentador | Duração | Conteúdo |
|-------|--------------|---------|----------|
| 1 | **Cleiton** | 4 min | Introdução, Código e Modelo |
| 2 | **Geovana** | 4 min | API, Docker e Demonstração |

---

## 🖥️ Preparação da Tela

Antes de gravar, deixe abertas as seguintes janelas/abas:

1. **VS Code** com o projeto aberto
2. **Terminal** com Docker rodando
3. **Navegador** com Swagger (http://localhost:8000/docs)
4. **Finder/Explorer** na pasta `models/` com os gráficos

---

# 🎤 PARTE 1 - CLEITON (4 minutos)

## [0:00 - 0:30] Abertura

**TELA:** VS Code aberto no arquivo `PROGRESS.md`

**FALA:**
> "Olá! Somos Cleiton e Geovana, e vamos apresentar nosso Tech Challenge da Fase 4: um sistema de predição de preços de ações utilizando redes neurais LSTM."

> "Escolhemos a ação **PETR4** da Petrobras para demonstrar o modelo. Como podem ver aqui no nosso arquivo de progresso, o projeto está 100% completo com todas as 9 etapas finalizadas."

**AÇÃO:** Scrollar brevemente pelo PROGRESS.md mostrando as etapas concluídas

---

### 📊 VISUAL: Resumo do Progresso (para scroll)

```
╔═══════════════════════════════════════════════════════════════════╗
║           🚀 TECH CHALLENGE LSTM - PETR4.SA                       ║
║                     PROGRESSO: 100% ████████████████████          ║
╚═══════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────┐
│  ✅ ETAPA 1   │ Setup do Ambiente        │ 15 min  │  2026-02-17   │
│───────────────┼──────────────────────────┼─────────┼───────────────│
│  ✅ ETAPA 2   │ Coleta de Dados          │ 10 min  │  1.487 registros
│───────────────┼──────────────────────────┼─────────┼───────────────│
│  ✅ ETAPA 3   │ Pré-processamento        │ 10 min  │  60 dias/janela
│───────────────┼──────────────────────────┼─────────┼───────────────│
│  ✅ ETAPA 4   │ Modelo LSTM              │ 10 min  │  31.051 params
│───────────────┼──────────────────────────┼─────────┼───────────────│
│  ✅ ETAPA 5   │ Treinamento              │ 20 min  │  100 épocas
│───────────────┼──────────────────────────┼─────────┼───────────────│
│  ✅ ETAPA 6   │ Avaliação                │    ✓    │  MAPE: 3.83%
│───────────────┼──────────────────────────┼─────────┼───────────────│
│  ✅ ETAPA 7   │ Salvamento               │    ✓    │  3 artefatos
│───────────────┼──────────────────────────┼─────────┼───────────────│
│  ✅ ETAPA 8   │ API FastAPI              │    ✓    │  /predict
│───────────────┼──────────────────────────┼─────────┼───────────────│
│  ✅ ETAPA 9   │ Docker                   │    ✓    │  stock-api
└─────────────────────────────────────────────────────────────────────┘

╔═══════════════════════════════════════════════════════════════════╗
║  📈 RESULTADOS FINAIS                                             ║
╠═══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║   TICKER:        PETR4.SA (Petrobras)                             ║
║   PERÍODO:       2018-01-01 → 2024-01-01 (6 anos)                 ║
║   DADOS:         1.487 registros de preço                         ║
║                                                                   ║
║   ┌─────────────────────────────────────────────────────────┐     ║
║   │  🧠 MODELO LSTM                                         │     ║
║   │  ├── 2 camadas LSTM × 100 neurônios                     │     ║
║   │  ├── Dropout: 20%                                       │     ║
║   │  └── Parâmetros: 31.051                                 │     ║
║   └─────────────────────────────────────────────────────────┘     ║
║                                                                   ║
║   ┌─────────────────────────────────────────────────────────┐     ║
║   │  📊 MÉTRICAS                                            │     ║
║   │  ├── MAPE:  3.83%   (erro < 4%)                         │     ║
║   │  ├── RMSE:  R$ 0.89 (erro < R$1)                        │     ║
║   │  └── Loss:  0.0014  (convergiu)                         │     ║
║   └─────────────────────────────────────────────────────────┘     ║
║                                                                   ║
║   ┌─────────────────────────────────────────────────────────┐     ║
║   │  🚀 API                                                 │     ║
║   │  ├── Framework: FastAPI                                 │     ║
║   │  ├── Endpoints: /health, /predict                       │     ║
║   │  └── Tempo resp: ~12ms                                  │     ║
║   └─────────────────────────────────────────────────────────┘     ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
```

---

## [0:30 - 1:00] Estrutura do Projeto

**TELA:** Explorador de arquivos do VS Code (sidebar)

**FALA:**
> "Nosso projeto segue uma estrutura organizada. Temos:"
> - "A pasta `src/` com todo o código-fonte"
> - "A pasta `models/` com o modelo treinado e os gráficos"
> - "A pasta `data/` com os dados históricos"
> - "E os arquivos Docker para containerização"

**AÇÃO:** Clicar nas pastas enquanto fala, expandindo cada uma brevemente

---

## [1:00 - 1:45] Coleta de Dados

**TELA:** Abrir o arquivo `src/data_collection.py`

**FALA:**
> "A primeira etapa foi a coleta de dados. Usamos a biblioteca **yfinance** para baixar o histórico de preços."

**AÇÃO:** Mostrar o código, destacando:
- O ticker PETR4.SA
- O período de 2018 a 2024

> "Coletamos 6 anos de dados históricos - cerca de 1.400 registros de preços de fechamento."

**TELA:** Abrir o arquivo `data/data_PETR4_SA.csv` brevemente

> "Os dados ficam salvos neste CSV para não precisarmos baixar toda vez."

---

## [1:45 - 2:30] Pré-processamento

**TELA:** Abrir o arquivo `src/preprocessing.py`

**FALA:**
> "No pré-processamento, fazemos três coisas importantes:"

**AÇÃO:** Scrollar pelo código enquanto explica:

> "Primeiro, **normalizamos** os dados entre 0 e 1 usando MinMaxScaler. Isso é essencial porque redes neurais funcionam melhor com valores pequenos."

> "Segundo, criamos **janelas deslizantes de 60 dias**. O modelo usa os últimos 60 dias de preços para prever o dia seguinte."

> "Terceiro, dividimos em **80% treino e 20% teste**."

---

## [2:30 - 3:15] Modelo LSTM

**TELA:** Abrir o arquivo `src/model.py`

**FALA:**
> "Aqui está o coração do projeto: a classe **StockLSTM**."

**AÇÃO:** Mostrar a classe, destacando:

> "Escolhemos LSTM porque ela é ideal para séries temporais. Diferente de RNNs comuns, a LSTM tem portões que controlam o que manter na memória."

> "Nossa arquitetura tem:"
> - "2 camadas LSTM com 100 neurônios cada"
> - "Dropout de 20% para evitar overfitting"
> - "Uma camada Linear que gera o preço previsto"

---

## [3:15 - 4:00] Treinamento e Resultados

**TELA:** Abrir a imagem `models/training_history.png`

**FALA:**
> "Treinamos por 100 épocas usando o otimizador Adam. Neste gráfico vemos as curvas de loss de treino e validação convergindo, sem sinais de overfitting."

**TELA:** Abrir a imagem `models/predictions_vs_actual.png`

> "E aqui temos o resultado: a linha azul são os valores reais e a vermelha são as previsões do modelo. Vejam como acompanha bem o padrão."

> "Nossas métricas finais:"
> - "**MAPE de 3,83%** - o modelo erra em média menos de 4%"
> - "**RMSE de 89 centavos** - erro médio de menos de 1 real"

> "Agora passo para a Geovana mostrar a API funcionando."

---

# 🎤 PARTE 2 - GEOVANA (4 minutos)

## [4:00 - 4:45] API FastAPI

**TELA:** Abrir o arquivo `src/app.py`

**FALA:**
> "Obrigado, Cleiton. Agora vou mostrar como transformamos esse modelo em uma aplicação de produção."

> "Criamos uma API REST com FastAPI. Ela tem dois endpoints principais:"

**AÇÃO:** Scrollar mostrando o código:

> "O `/health` que verifica se o modelo está carregado..."

> "E o `/predict` que recebe uma lista de preços e retorna a previsão."

**AÇÃO:** Mostrar os schemas Pydantic:

> "Usamos Pydantic para validar a entrada - o usuário precisa enviar pelo menos 60 preços."

---

## [4:45 - 5:15] Docker

**TELA:** Abrir o arquivo `Dockerfile`

**FALA:**
> "Para garantir que rode em qualquer ambiente, containerizamos com Docker."

**AÇÃO:** Mostrar o Dockerfile:

> "Usamos Python 3.10, instalamos as dependências, copiamos o modelo treinado e configuramos o health check automático."

**TELA:** Abrir o terminal

> "Vou mostrar que o container já está rodando."

**AÇÃO:** Digitar:
```bash
docker ps
```

> "Aqui está nosso container `stock-api` ativo na porta 8000."

---

## [5:15 - 5:45] Demo: Health Check

**TELA:** Terminal

**FALA:**
> "Vamos testar a API. Primeiro, o health check:"

**AÇÃO:** Digitar e executar:
```bash
curl http://localhost:8000/health
```

> "A resposta mostra que o modelo está carregado, rodando em CPU, configurado para PETR4 com janela de 60 dias."

---

## [5:45 - 6:45] Demo: Previsão

**TELA:** Navegador com Swagger UI (http://localhost:8000/docs)

**FALA:**
> "Agora vou fazer uma previsão usando a documentação interativa do FastAPI."

**AÇÃO:** 
1. Clicar no endpoint POST `/predict`
2. Clicar em "Try it out"
3. Colar o JSON com 60 preços
4. Clicar em "Execute"

> "Estou enviando os últimos 60 preços de fechamento da PETR4..."

**AÇÃO:** Mostrar a resposta

> "E a API retornou! O modelo previu **R$ 38,03** para o próximo dia de fechamento."

> "Vejam que o tempo de processamento foi de apenas 12 milissegundos - muito rápido para uso em produção."

---

## [6:45 - 7:15] Demo: Terminal (alternativa)

**TELA:** Terminal

**FALA:**
> "Também podemos chamar via terminal com curl:"

**AÇÃO:** Executar o comando curl completo (ter pronto para colar)

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"prices": [36.5, 36.8, 37.1, 37.0, 36.9, 37.2, 37.5, 37.3, 37.6, 37.8, 38.0, 37.9, 38.2, 38.1, 38.4, 38.3, 38.6, 38.5, 38.8, 38.7, 39.0, 38.9, 39.2, 39.1, 39.4, 39.3, 39.6, 39.5, 39.8, 39.7, 40.0, 39.9, 40.2, 40.1, 40.4, 40.3, 40.6, 40.5, 40.8, 40.7, 41.0, 40.9, 41.2, 41.1, 41.4, 41.3, 41.6, 41.5, 41.8, 41.7, 42.0, 41.9, 42.2, 42.1, 42.4, 42.3, 42.6, 42.5, 42.8, 42.7]}'
```

> "Mesma resposta, mostrando que a API funciona tanto pelo Swagger quanto por linha de comando."

---

## [7:15 - 7:40] Monitoramento

**TELA:** Terminal com logs ou VS Code no app.py

**FALA:**
> "Para monitoramento em produção, implementamos:"
> - "O endpoint `/health` para verificar disponibilidade"
> - "O campo `processing_time_ms` em cada resposta"
> - "Health check do Docker que reinicia o container se a API parar"

**AÇÃO:** Mostrar logs do Docker se possível:
```bash
docker logs stock-api --tail 10
```

---

## [7:40 - 8:00] Conclusão

**TELA:** VS Code no PROGRESS.md ou nos gráficos

**FALA (Geovana):**
> "Para finalizar, nosso projeto entrega:"

**FALA (Cleiton entra):**
> - "Um modelo LSTM com precisão de 96% - MAPE de apenas 3,83%"
> - "Uma API REST documentada e funcional"
> - "Container Docker pronto para deploy"
> - "Todo o código-fonte documentado no GitHub"

**FALA (Geovana):**
> "Obrigado pela atenção!"

**FALA (Cleiton):**
> "Estamos à disposição para perguntas."

---

# 📝 Checklist de Preparação

## Janelas para Deixar Abertas
- [ ] VS Code com o projeto
- [ ] Terminal com Docker rodando
- [ ] Navegador em http://localhost:8000/docs

## Arquivos para Abrir Rapidamente
1. `PROGRESS.md`
2. `src/data_collection.py`
3. `src/preprocessing.py`
4. `src/model.py`
5. `src/app.py`
6. `Dockerfile`
7. `models/training_history.png`
8. `models/predictions_vs_actual.png`

## Verificar Antes de Gravar
- [ ] Container Docker rodando: `docker ps`
- [ ] API respondendo: `curl http://localhost:8000/health`
- [ ] Swagger abrindo: http://localhost:8000/docs
- [ ] Gráficos existem na pasta `models/`

## Comandos Prontos (salvar em um arquivo .txt)

```bash
# Verificar container
docker ps

# Health check
curl http://localhost:8000/health

# Logs do container
docker logs stock-api --tail 10

# Previsão completa
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"prices": [36.5, 36.8, 37.1, 37.0, 36.9, 37.2, 37.5, 37.3, 37.6, 37.8, 38.0, 37.9, 38.2, 38.1, 38.4, 38.3, 38.6, 38.5, 38.8, 38.7, 39.0, 38.9, 39.2, 39.1, 39.4, 39.3, 39.6, 39.5, 39.8, 39.7, 40.0, 39.9, 40.2, 40.1, 40.4, 40.3, 40.6, 40.5, 40.8, 40.7, 41.0, 40.9, 41.2, 41.1, 41.4, 41.3, 41.6, 41.5, 41.8, 41.7, 42.0, 41.9, 42.2, 42.1, 42.4, 42.3, 42.6, 42.5, 42.8, 42.7]}'
```

---

# 🎯 Dicas para Gravação

1. **Pratiquem a navegação** - saibam onde clicar sem hesitar
2. **Aumentem a fonte** do VS Code e terminal (Cmd/Ctrl + para zoom)
3. **Fechem notificações** do sistema antes de gravar
4. **Falem enquanto navegam** - evitem silêncios longos
5. **Movam o mouse devagar** - facilita acompanhar
6. **Se errarem, continuem** - pequenos erros são normais
7. **Ensaiem 2x antes** de gravar pra valer

---

# 📐 Ordem de Navegação (Cola)

### Cleiton:
1. PROGRESS.md (mostrar etapas)
2. Sidebar (estrutura pastas)
3. data_collection.py
4. data/data_PETR4_SA.csv (rápido)
5. preprocessing.py
6. model.py
7. models/training_history.png
8. models/predictions_vs_actual.png

### Geovana:
1. src/app.py
2. Dockerfile
3. Terminal: `docker ps`
4. Terminal: `curl health`
5. Navegador: Swagger /predict
6. Terminal: `curl predict` (opcional)
7. Terminal: `docker logs`
8. PROGRESS.md ou gráficos (fechamento)

---

---

# 📊 RESUMO VISUAL DO PROGRESSO (Cola Rápida)

> Use esta seção se precisar de uma referência rápida dos números durante a apresentação.

## Timeline das Etapas

| # | Etapa | Status | Entregável Principal |
|:-:|-------|:------:|----------------------|
| 1 | Setup Ambiente | ✅ | `requirements.txt` + `venv` |
| 2 | Coleta de Dados | ✅ | `data_PETR4_SA.csv` (1.487 linhas) |
| 3 | Pré-processamento | ✅ | Janelas 60 dias + 80/20 split |
| 4 | Modelo LSTM | ✅ | `StockLSTM` (31.051 params) |
| 5 | Treinamento | ✅ | 100 épocas, Loss: 0.0014 |
| 6 | Avaliação | ✅ | MAPE: 3.83%, RMSE: R$0.89 |
| 7 | Salvamento | ✅ | `.pth` + `.pkl` |
| 8 | API FastAPI | ✅ | `/predict` + `/health` |
| 9 | Docker | ✅ | `stock-api` container |

## Números Importantes para Mencionar

```
┌────────────────────────────────────────────────────────┐
│  📈 DADOS                                              │
│     • Ticker: PETR4.SA (Petrobras)                     │
│     • Período: 6 anos (2018-2024)                      │
│     • Registros: 1.487                                 │
│     • Preço min: R$ 3.24 | max: R$ 27.38              │
├────────────────────────────────────────────────────────┤
│  🧠 MODELO                                             │
│     • Arquitetura: 2 LSTM × 100 neurônios             │
│     • Dropout: 20%                                     │
│     • Janela: 60 dias                                  │
│     • Parâmetros: 31.051                               │
├────────────────────────────────────────────────────────┤
│  📊 TREINAMENTO                                        │
│     • Épocas: 100                                      │
│     • Tempo: 18.7 segundos                             │
│     • Train Loss: 0.001405                             │
│     • Val Loss: 0.002383                               │
├────────────────────────────────────────────────────────┤
│  🎯 MÉTRICAS FINAIS                                    │
│     • MAPE: 3.83% (precisão ~96%)                      │
│     • RMSE: R$ 0.89                                    │
│     • MAE: R$ 0.67                                     │
├────────────────────────────────────────────────────────┤
│  🚀 API                                                │
│     • Tempo resposta: ~12ms                            │
│     • Endpoints: 2 (/health, /predict)                 │
│     • Container: stock-api:8000                        │
└────────────────────────────────────────────────────────┘
```

## Decisões de Projeto (se perguntarem)

| Decisão | Escolha | Justificativa |
|---------|---------|---------------|
| Ação | PETR4.SA | Ação brasileira, alta liquidez |
| Período | 6 anos | Volume adequado sem dados antigos demais |
| Janela | 60 dias | ~3 meses, captura tendências |
| Split | 80/20 | Padrão da literatura |
| Camadas | 2 LSTM | Equilíbrio complexidade/eficiência |
| Dropout | 20% | Previne overfitting |
| Épocas | 100 | Convergência estável |

---

*Roteiro atualizado em: 2026-02-19*
