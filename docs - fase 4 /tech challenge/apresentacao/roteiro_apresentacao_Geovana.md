API FastAPI

**TELA:** Abrir o arquivo `src/app.py`
**FALA:**> "Oi eu sou a Geovana e agora vou mostrar como transformamos esse modelo em uma aplicação de produção."
**FALA:**>> "Conforme é solicitado como um dos requisitos dessa desafio, a gente criou uma API REST com FastAPI, onde existem dois endpoints principais:"

**AÇÃO:** Scrollar mostrando o código:

**FALA:**>> "O `/health` que verifica se o modelo está carregado..."

**FALA:**>> "E o `/predict` que recebe uma lista de preços e retorna a previsão."

**AÇÃO:** Mostrar os schemas Pydantic:

**FALA:**>> "A gente decidiu usar o Pydantic para validar a entrada - entao o usuário precisa enviar pelo menos 60 preços."


----------------------------------------------------------------------------------------------------------------------------

 Docker

**TELA:** Abrir o arquivo `Dockerfile`

**FALA:**> "Para garantir que rode em qualquer ambiente, containerizamos com Docker."

**AÇÃO:** Mostrar o Dockerfile:

**FALA:**> "Usamos Python 3.10, instalamos as dependências, copiamos o modelo treinado e configuramos o health check automático."

**TELA:** Abrir o terminal

**FALA:**> "Vou mostrar que o container já está rodando."

**AÇÃO:** Digitar:
```bash
docker ps
```

**FALA:**> "Aqui está nosso container `stock-api` ativo na porta 8000."

----------------------------------------------------------------------------------------------------------------------------

Demo: Health Check

**TELA:** Terminal

**FALA:**> "Vamos testar a API. Primeiro, o health check:"

**AÇÃO:** Digitar e executar:
```bash
curl http://localhost:8000/health
```

**FALA:**> "Bom a resposta mostra que o modelo tá carregado como esperado, rodando em CPU, configurado para PETR4 com janela de 60 dias."

----------------------------------------------------------------------------------------------------------------------------

 Demo: Via Ter

**TELA:** postman

**FALA:**> "Também podemos chamar via postman com curl:"

**AÇÃO:** importal o curl completo (ter pronto para colar)

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"prices": [36.5, 36.8, 37.1, 37.0, 36.9, 37.2, 37.5, 37.3, 37.6, 37.8, 38.0, 37.9, 38.2, 38.1, 38.4, 38.3, 38.6, 38.5, 38.8, 38.7, 39.0, 38.9, 39.2, 39.1, 39.4, 39.3, 39.6, 39.5, 39.8, 39.7, 40.0, 39.9, 40.2, 40.1, 40.4, 40.3, 40.6, 40.5, 40.8, 40.7, 41.0, 40.9, 41.2, 41.1, 41.4, 41.3, 41.6, 41.5, 41.8, 41.7, 42.0, 41.9, 42.2, 42.1, 42.4, 42.3, 42.6, 42.5, 42.8, 42.7]}'
```

**FALA:**> "A resposta mostra o preço previsto de R$ 37,38 para o próximo dia. Vejam que a API também retorna informações do modelo utilizado: LSTM com 100 neurônios e 2 camadas. O tempo de processamento foi de apenas 8 milissegundos, demonstrando que a solução é rápida o suficiente para uso em produção."

**FALA:**> "Mesma resposta, mostrando que a API funciona tanto pelo Swagger quanto por linha de comando."

----------------------------------------------------------------------------------------------------------------------------

## [7:15 - 7:40] Monitoramento

**TELA:** Terminal com logs ou VS Code no app.py

**FALA:**> "Para monitoramento em produção, implementamos:"
**FALA:**> - "O endpoint `/health` para verificar disponibilidade"
**FALA:**> - "O campo `processing_time_ms` em cada resposta"
**FALA:**> - "Health check do Docker que reinicia o container se a API parar"

**AÇÃO:** Mostrar logs do Docker se possível:
```bash
docker logs stock-api --tail 10
```

----------------------------------------------------------------------------------------------------------------------------

## [7:40 - 8:00] Conclusão

**TELA:** VS Code no PROGRESS.md ou nos gráficos

**FALA:**> "Para finalizar, nosso projeto entrega:"

**FALA:**> - "Um modelo LSTM com precisão de 96% - MAPE de apenas 3,83%"
**FALA:**> - "Uma API REST documentada e funcional"
**FALA:**> - "Container Docker pronto para deploy"
**FALA:**> - "Todo o código-fonte documentado no GitHub"

**FALA**> "Obrigado pela atenção!"


