# Resumo do Treinamento - Para o Professor

Professor, sobre o treinamento do modelo LSTM para prever ações da Petrobras:

Comecei com a configuração padrão da literatura (50 neurônios na camada oculta) e o modelo ficou "bom", com erro de 6.74%. Mas queria chegar em "excelente" (menos de 5% de erro). Fiz 12 experimentos variando parâmetros como velocidade de aprendizado, quantidade de épocas e tamanho da rede. A grande descoberta foi que o modelo estava "apertado" demais - era como tentar guardar um guarda-roupa inteiro numa mala pequena. Quando dobrei a capacidade da rede (de 50 para 100 neurônios), o erro caiu 43%, chegando a 3.83%.

Resumindo: o modelo precisava de mais "espaço mental" para aprender os padrões de 6 anos de dados históricos. A otimização levou o resultado de "bom" para "excelente" com uma mudança simples, mas que só descobri testando sistematicamente.
