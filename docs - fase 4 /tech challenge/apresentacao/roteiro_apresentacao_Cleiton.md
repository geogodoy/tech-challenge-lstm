Cleiton

Introducao:
**FALA:**> "Olá! eu sou o Cleiton, e vou apresentar a primeira parte do desafio de Tech Challenge da Fase 4: um sistema de predição de preços de ações utilizando redes neurais LSTM."

----------------------------------------------------------------------------------------------------------------------------
Coleta de dados:

**TELA:** Abrir o arquivo `src/data_collection.py`

**FALA:**> "A primeira etapa foi a coleta de dados. Usamos a biblioteca **yfinance** para baixar o histórico de preços." 
**FALA:**> "Para o desafio escolhemos a a ação **PETR4** da Petrobras para demonstrar o modelo."
> **Por quê?** A PETR4 é uma das ações mais negociadas da B3, com alto volume e liquidez. Isso garante dados consistentes sem gaps. Além disso, por ser uma ação de commodity (petróleo), apresenta volatilidade interessante para testar a capacidade preditiva do modelo.

**FALA:**> "Coletamos 6 anos de dados históricos - cerca de 1.400 registros de preços de fechamento."
> **Por quê?** Redes LSTM precisam de volume suficiente de dados para aprender padrões. 6 anos capturam diferentes ciclos de mercado: alta, baixa e lateralização. Menos dados resultaria em underfitting; muito mais poderia incluir padrões obsoletos do mercado.

**AÇÃO:** Mostrar o código, destacando:
- O ticker PETR4.SA
- O período de 2018 a 2024

**TELA:** Abrir o arquivo `data/data_PETR4_SA.csv` brevemente

**FALA:**> "Os dados ficam salvos neste CSV para não precisarmos baixar toda vez."

----------------------------------------------------------------------------------------------------------------------------
Pré-processamento

**TELA:** Abrir o arquivo `src/preprocessing.py`

**FALA:**
> "No pré-processamento, fazemos três coisas importantes:"

**AÇÃO:** Scrollar pelo código enquanto explica:

> "Primeiro, **normalizamos** os dados entre 0 e 1 usando MinMaxScaler. Isso é essencial porque redes neurais funcionam melhor com valores pequenos."

> **Por quê?** Sem normalização, valores altos (como R$40) dominam os gradientes e dificultam o aprendizado. Com dados entre 0-1, os pesos da rede convergem mais rápido e de forma estável. Também evita problemas numéricos como vanishing/exploding gradients.

> "Segundo, criamos **janelas deslizantes de 60 dias**. O modelo usa os últimos 60 dias de preços para prever o dia seguinte."

> **Por quê?** 60 dias representa aproximadamente 3 meses de pregão, capturando tendências de curto/médio prazo. É um valor comum na literatura financeira. Janelas muito curtas perdem contexto; muito longas incluem ruído e aumentam o custo computacional.

> "Terceiro, dividimos em **80% treino e 20% teste**."

> **Por quê?** A proporção 80/20 é um padrão em machine learning que equilibra dados suficientes para treinar sem comprometer a validação. Para séries temporais, mantemos a ordem cronológica: treino com dados antigos, teste com dados recentes, simulando uso real.

----------------------------------------------------------------------------------------------------------------------------

Modelo LSTM

**TELA:** Abrir o arquivo `src/model.py`

**FALA:**> "Aqui está o coração do projeto: a classe **StockLSTM**."

**AÇÃO:** Mostrar a classe, destacando:

**FALA:** > "O LSTM  é ideal para séries temporais, e se encaixa perfeitamente com o que é proposto na regra de negócio, ou seja, a sequencia temporal importa na previsao mais acertiva. Diferente de RNNs comuns, a LSTM tem uma estrutura que suporta uma memória mais longa durante a inferencia e portões que controlam o que manter na memória."

**FALA:**> "Nossa arquitetura tem:"
**FALA:**> - "2 camadas LSTM com 100 neurônios cada"
> **Por quê?** Duas camadas permitem aprender padrões hierárquicos: a primeira camada captura padrões simples, a segunda combina esses padrões em representações mais complexas. 100 neurônios oferecem capacidade suficiente sem overfitting para nosso volume de dados.

**FALA:**> - "Dropout de 20% para evitar overfitting"
> **Por quê?** Dropout "desliga" aleatoriamente 20% dos neurônios durante o treino, forçando a rede a não depender de caminhos específicos. Isso previne memorização dos dados de treino e melhora a generalização. 20% é um valor conservador que regulariza sem prejudicar o aprendizado.

**FALA:**> - "Uma camada Linear que gera o preço previsto"
> **Por quê?** A LSTM extrai features temporais, mas sua saída tem 100 dimensões. A camada Linear (fully connected) projeta essas 100 dimensões em um único valor: o preço previsto. É a "cabeça de regressão" que transforma representações abstratas em predição concreta.

----------------------------------------------------------------------------------------------------------------------------
Treinamento e Resultados

**TELA:** Abrir a imagem `models/training_history.png`
**FALA:**> "Treinamos por 100 épocas usando o otimizador Adam. Neste gráfico vemos as curvas de loss de treino e validação convergindo, sem sinais de overfitting."
> **Por quê?** Adam combina os benefícios de AdaGrad e RMSprop, adaptando a taxa de aprendizado por parâmetro. É robusto e converge rápido. 100 épocas foi suficiente para a convergência sem overfit, evidenciado pelas curvas de loss que param de cair juntas.

**FALA:**> "Vale destacar o processo de otimização: começamos com a configuração padrão da literatura, 50 neurônios na camada oculta, e o modelo ficou 'bom', com erro de 6,74%. Mas queríamos chegar em 'excelente', menos de 5% de erro."

**FALA:**> "Fizemos 12 experimentos variando parâmetros como taxa de aprendizado, quantidade de épocas e tamanho da rede. A grande descoberta foi que o modelo estava 'apertado' demais - como tentar guardar um guarda-roupa inteiro numa mala pequena. Quando dobramos a capacidade da rede de 50 para 100 neurônios, o erro caiu 43%, chegando aos 3,83% que vemos aqui."
> **Resumo:** O modelo precisava de mais "espaço mental" para aprender os padrões de 6 anos de dados históricos. A otimização levou o resultado de "bom" para "excelente" com uma mudança simples, mas que só descobrimos testando sistematicamente.

**TELA:** Abrir a imagem `models/predictions_vs_actual.png`
**FALA:**> "E aqui temos o resultado: a linha azul são os valores reais e a vermelha são as previsões do modelo. Vejam como acompanha bem o padrão."
> **Por quê?** O gráfico visual é a prova concreta de que a LSTM capturou a dinâmica temporal dos preços. O modelo não apenas segue a tendência geral, mas também reage a variações de curto prazo, validando que aprendeu padrões reais e não apenas uma média.

**FALA:**> "Nossas métricas finais:"
**FALA:**> - "**MAPE de 3,83%** - o modelo erra em média menos de 4%"
**FALA:**> - "**RMSE de 89 centavos** - erro médio de menos de 1 real"
> **Por quê?** MAPE abaixo de 5% é considerado excelente em previsão financeira. RMSE de R$0,89 em ações que custam ~R$35 representa erro de ~2,5%. Essas métricas complementares confirmam que o modelo tem precisão prática para auxiliar decisões de investimento.

**FALA:**> "Agora passo para a Geovana mostrar a API funcionando."

