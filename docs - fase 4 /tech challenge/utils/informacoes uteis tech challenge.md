# Informações Úteis - Tech Challenge Fase 4

## 1. Viabilidade: 3 dias (16 a 19) é suficiente?

**Resposta curta: Sim, é viável**, mas com algumas ressalvas.

### Análise de Tempo

Baseando-se no guia criado, o projeto pode ser dividido em:

| Etapa | Tempo Estimado |
|-------|----------------|
| Setup do Ambiente | ~30 min |
| Coleta de Dados (yfinance) | ~30 min |
| Análise e Pré-processamento | ~45 min |
| Construção do Modelo LSTM | ~45 min |
| Treinamento e Ajuste | 1h - 2h |
| Avaliação (MAE/RMSE) | ~30 min |
| Deploy da API (FastAPI) | ~1h |
| Docker + Monitoramento | ~1h |
| Documentação | ~30 min - 1h |
| **Gravação do Vídeo** | ~1h |
| **Total** | **~8-10 horas de trabalho** |

### Considerações

- **3 dias são ~72 horas** - você precisa de apenas 8-10h de trabalho focado
- Se conseguir **3-4 horas por dia**, entrega tranquilo
- O maior risco é o **treinamento do modelo** - pode exigir ajustes iterativos
- O deploy em nuvem é **opcional** (pode rodar local com Docker)

**Recomendação:** Siga o cronograma do guia com as sessões de 45-60 min + pausas. É totalmente factível.

---

## 2. Informações Essenciais do `tech_challenge_descricao_fase4`

### Peso e Importância
- **Vale 90% da nota** de todas as disciplinas da fase
- É **obrigatório**
- Deve ser desenvolvido em **grupo** (mas a princípio)

### Objetivo Central
> Criar um modelo preditivo LSTM para prever o **valor de fechamento** de ações de uma empresa à sua escolha

### Os 5 Requisitos Obrigatórios

1. **Coleta e Pré-processamento**
   - Usar `yfinance` para dados históricos
   - Período sugerido: 2018-2024 (dados suficientes para treinar)

2. **Desenvolvimento do Modelo LSTM**
   - Construir, treinar e ajustar hiperparâmetros
   - Avaliar com métricas: **MAE, RMSE ou MAPE**

3. **Salvamento do Modelo**
   - Salvar em formato para inferência (`.pth` ou `.h5`)

4. **Deploy em API RESTful**
   - Usar **Flask ou FastAPI**
   - Endpoint que recebe dados históricos e retorna previsão

5. **Escalabilidade e Monitoramento**
   - Configurar ferramentas para rastrear performance
   - Tempo de resposta e utilização de recursos

### Entregáveis Finais

| Entregável | Obrigatório |
|------------|-------------|
| Código-fonte + documentação no GIT | Sim |
| Scripts ou containers Docker | Sim |
| Link da API em produção (nuvem) | Opcional |
| **Vídeo explicativo** | Sim |

### Dica Crítica

O vídeo é importante - deve mostrar **todo o funcionamento da API**. Não deixe para o último momento.

---

## 3. Checklist Rápido de Entrega

- [ ] Ambiente configurado (venv + dependências)
- [ ] Dados coletados via yfinance
- [ ] Dados pré-processados e normalizados
- [ ] Modelo LSTM construído e treinado
- [ ] Métricas de avaliação calculadas (MAE/RMSE/MAPE)
- [ ] Modelo salvo (.pth ou .h5)
- [ ] API criada com FastAPI/Flask
- [ ] Dockerfile configurado
- [ ] Monitoramento básico implementado
- [ ] Documentação do projeto no README
- [ ] Código commitado no repositório GIT
- [ ] Vídeo gravado e enviado
