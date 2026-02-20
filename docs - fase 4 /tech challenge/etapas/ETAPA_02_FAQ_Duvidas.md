# ❓ FAQ - Dúvidas da Etapa 2: Coleta de Dados

> Documento complementar à [ETAPA_02_Coleta_Dados.md](./ETAPA_02_Coleta_Dados.md)

---

## 📚 Índice

1. [O que é Regressão?](#1-o-que-é-regressão)
2. [Configurações da Coleta Explicadas](#2-configurações-da-coleta-explicadas)
3. [Por que as Estatísticas Min, Max e Média?](#3-por-que-as-estatísticas-min-max-e-média)
4. [Como funciona a função load_stock_data()](#4-como-funciona-a-função-load_stock_data)
5. [O que são as colunas OHLCV?](#5-o-que-são-as-colunas-ohlcv)
6. [Por que usamos apenas a coluna Close?](#6-por-que-usamos-apenas-a-coluna-close)
7. [O que é yfinance?](#7-o-que-é-yfinance)
8. [Por que os dados têm menos dias que 6 anos?](#8-por-que-os-dados-têm-menos-dias-que-6-anos)
9. [O que é MultiIndex e por que aparece no código?](#9-o-que-é-multiindex-e-por-que-aparece-no-código)
10. [Por que salvar em CSV?](#10-por-que-salvar-em-csv)

---

## 1. O que é Regressão?

**Referência:** Linha 30 do documento principal

### Definição Simples

**Regressão** é um tipo de problema em Machine Learning onde queremos **prever um valor numérico contínuo**.

### Comparação: Regressão vs Classificação

| Tipo de Problema | O que prevê | Exemplos |
|------------------|-------------|----------|
| **Classificação** | Categoria/Classe | "É spam ou não?", "É gato ou cachorro?", "Vai subir ou descer?" |
| **Regressão** | Número contínuo | "Qual será o preço amanhã?", "Qual a temperatura às 15h?", "Quantos produtos venderemos?" |

### No Contexto do Projeto

Estamos fazendo **regressão** porque queremos prever o **preço exato** de uma ação:
- ✅ Regressão: "O preço será R$ 27.50"
- ❌ Classificação: "O preço vai subir"

A saída do modelo LSTM será um **número** (ex: 27.50), não uma **categoria** (subir/descer).

---

## 2. Configurações da Coleta Explicadas

**Referência:** Linhas 15-22 do código `data_collection.py`

```python
TICKER = "PETR4.SA"
START_DATE = "2018-01-01"
END_DATE = "2024-01-01"
DATA_DIR = Path(__file__).parent.parent / "data"
```

### 2.1 TICKER - O Código da Ação

**O que é:** O "apelido oficial" de uma empresa na bolsa de valores.

| Ticker | Empresa | Mercado |
|--------|---------|---------|
| `PETR4.SA` | Petrobras | Brasil (B3) |
| `VALE3.SA` | Vale | Brasil (B3) |
| `ITUB4.SA` | Itaú | Brasil (B3) |
| `AAPL` | Apple | EUA (NASDAQ) |
| `MSFT` | Microsoft | EUA (NASDAQ) |

**Decodificando PETR4.SA:**
- `PETR` = Abreviação de Petrobras
- `4` = Tipo de ação (preferencial tipo 4 - tem preferência nos dividendos)
- `.SA` = Sufixo que indica bolsa brasileira (São Paulo / B3)

**Por que escolhemos Petrobras?**

| Motivo | Explicação |
|--------|------------|
| **Alta liquidez** | Muitas pessoas compram/vendem = dados confiáveis |
| **Volatilidade** | Preço varia bastante = bom para a IA aprender padrões |
| **Dados consistentes** | Empresa grande, dados históricos sem "buracos" |
| **Relevância** | Uma das maiores empresas do Brasil |

### 2.2 START_DATE e END_DATE - O Período

```
2018-01-01  ──────────────────────────────────>  2024-01-01
            |←──────── 6 anos de dados ────────→|
```

**Por que 6 anos?**

| Razão | Explicação |
|-------|------------|
| **Volume de dados** | ~1.487 dias de negociação para treinar |
| **Eventos diversos** | Inclui pandemia (2020), eleições, crises - a IA aprende variações |
| **Padrões sazonais** | 6 anos capturam ciclos anuais (ex: alta no fim do ano) |
| **Requisito LSTM** | Redes neurais precisam de muitos dados para aprender |

### 2.3 DATA_DIR - Onde Salvar

```python
DATA_DIR = Path(__file__).parent.parent / "data"
```

**Tradução:**
```
Path(__file__)         → /caminho/para/src/data_collection.py
.parent                → /caminho/para/src/
.parent                → /caminho/para/
/ "data"               → /caminho/para/data/
```

**Resultado:** `tech-challenge-lstm/data/`

---

## 3. Por que as Estatísticas Min, Max e Média?

**Referência:** Linhas 65-69 do código

```python
print(f"   Mínimo: R$ {df[close_col].min():.2f}")   # R$ 3.24
print(f"   Máximo: R$ {df[close_col].max():.2f}")   # R$ 27.38
print(f"   Média:  R$ {df[close_col].mean():.2f}")  # R$ 10.17
```

### Propósito de Cada Estatística

| Estatística | Valor PETR4 | Para que serve |
|-------------|-------------|----------------|
| **Mínimo** | R$ 3.24 | Detectar erros (preço negativo = problema) |
| **Máximo** | R$ 27.38 | Ver amplitude total dos dados |
| **Média** | R$ 10.17 | Entender o "centro" dos dados |

### Por que isso importa?

1. **Validação de qualidade:**
   - Se mínimo fosse negativo → erro nos dados
   - Se máximo fosse R$ 1.000.000 → dado corrompido

2. **Entender a volatilidade:**
   - Variação de R$ 3 a R$ 27 = preço multiplicou por 9!
   - Alta volatilidade = bom para treinar a IA

3. **Preparar para normalização (Etapa 3):**
   - Na próxima etapa, usaremos esses valores para escalar entre 0 e 1
   - MinMaxScaler usa: `(valor - min) / (max - min)`

---

## 4. Como funciona a função load_stock_data()

**Referência:** Linhas 86-105 do código

### O que faz?

**Carrega os dados que já foram baixados**, sem precisar acessar a internet novamente.

### Código Linha por Linha

```python
def load_stock_data(ticker: str = TICKER) -> pd.DataFrame:
```
↳ Define função que recebe um ticker (padrão: PETR4.SA) e retorna um DataFrame

```python
    filepath = DATA_DIR / f"data_{ticker.replace('.', '_')}.csv"
```
↳ Monta o caminho: `data/data_PETR4_SA.csv`
↳ O `.replace('.', '_')` troca o ponto por underline (evita problemas com extensão)

```python
    if not filepath.exists():
        raise FileNotFoundError(f"❌ Arquivo não encontrado: {filepath}")
```
↳ Se o arquivo não existir, mostra erro claro (ao invés de erro genérico)

```python
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
```
↳ `index_col=0`: Usa a primeira coluna (Date) como índice
↳ `parse_dates=True`: Converte strings de data para objetos datetime

```python
    return df
```
↳ Retorna o DataFrame pronto para usar

### Por que essa função existe?

```
┌─────────────────────────────────┐    ┌─────────────────────────────────┐
│   SEM load_stock_data():        │    │   COM load_stock_data():        │
├─────────────────────────────────┤    ├─────────────────────────────────┤
│ • Sempre baixar da internet     │    │ • Baixou uma vez? Usa arquivo!  │
│ • Demora 5-10 segundos          │    │ • Carrega em ~0.1 segundos      │
│ • Depende de conexão            │    │ • Funciona offline              │
│ • Pode ter limite de requisições│    │ • Sem limite de uso             │
└─────────────────────────────────┘    └─────────────────────────────────┘
```

---

## 5. O que são as colunas OHLCV?

**Referência:** Linha 113 do documento principal

### Estrutura dos Dados Baixados

```
Date        Open      High      Low       Close     Volume
2018-01-02  4.31      4.40      4.31      4.40      33461800
```

### Significado de Cada Coluna

| Coluna | Nome Completo | O que representa |
|--------|---------------|------------------|
| **Date** | Data | Dia de negociação |
| **Open** | Abertura | Primeiro preço do dia (às 10h) |
| **High** | Máxima | Maior preço atingido no dia |
| **Low** | Mínima | Menor preço atingido no dia |
| **Close** | Fechamento | Último preço do dia (às 17h) |
| **Volume** | Volume | Quantidade de ações negociadas |

### Visualização de um Dia

```
Preço
  ↑
  │     ┌── High (máxima do dia)
  │     │
  │  ───┼─── Open (abertura às 10h)
  │     │
  │     │    [variações durante o dia]
  │     │
  │  ───┼─── Close (fechamento às 17h)
  │     │
  │     └── Low (mínima do dia)
  │
  └─────────────────────────────────> Tempo
       10h                        17h
```

---

## 6. Por que usamos apenas a coluna Close?

**Referência:** Linhas 125-127 do documento principal

### Motivo Principal

O preço de **fechamento (Close)** é o mais usado porque:

| Característica | Explicação |
|----------------|------------|
| **Representa o consenso** | É o preço que o mercado "concordou" no fim do dia |
| **Mais estável** | Menos sujeito a oscilações momentâneas |
| **Padrão da indústria** | Analistas e investidores usam Close como referência |
| **Base para indicadores** | Médias móveis, RSI, etc. são calculados sobre Close |

### Por que NÃO usamos as outras?

| Coluna | Por que não usar sozinha |
|--------|--------------------------|
| **Open** | Muito influenciada por overnight (notícias da noite) |
| **High/Low** | São extremos, não representam tendência geral |
| **Volume** | Não é preço, é quantidade (unidade diferente) |

### Nota Avançada

Em modelos mais sofisticados, podemos usar **todas as colunas OHLCV** como features. Mas para este projeto introdutório, focamos apenas no Close para simplificar.

---

## 7. O que é yfinance?

**Referência:** Linha 15 do documento principal

### Definição

`yfinance` é uma biblioteca Python que **baixa dados financeiros do Yahoo Finance** de forma gratuita e simples.

### Como funciona

```python
import yfinance as yf

# Baixa dados da Petrobras de 2018 a 2024
df = yf.download("PETR4.SA", start="2018-01-01", end="2024-01-01")
```

### Vantagens

| Vantagem | Descrição |
|----------|-----------|
| **Gratuito** | Não precisa pagar por API |
| **Simples** | Uma linha de código para baixar dados |
| **Confiável** | Dados do Yahoo Finance (fonte respeitada) |
| **Completo** | Inclui OHLCV, dividendos, splits, etc. |

### Limitações

| Limitação | Impacto |
|-----------|---------|
| Dados podem ter delay de 15min | Para trading real, não serve |
| Limite de requisições | Se baixar muito, pode bloquear temporariamente |
| Depende do Yahoo | Se Yahoo mudar API, biblioteca pode quebrar |

---

## 8. Por que os dados têm menos dias que 6 anos?

**Referência:** Linha 164 do documento principal

### O "Mistério" dos 1.487 dias

```
Período solicitado: 2018-01-01 até 2024-01-01 = 2.192 dias
Dados recebidos: 1.487 registros

Onde foram parar os outros 705 dias? 🤔
```

### Resposta: Mercado não abre todo dia!

A bolsa de valores **não funciona** em:
- Sábados e domingos (~104 dias/ano × 6 anos = ~624 dias)
- Feriados nacionais (~10-15 dias/ano × 6 anos = ~60-90 dias)

### Cálculo Aproximado

```
2.192 dias totais
-  624 fins de semana (104 × 6)
-   80 feriados aproximados
─────────────────────
≈ 1.488 dias úteis de mercado ✅
```

### Isso é um problema?

**Não!** É o comportamento esperado. Os dados estão corretos - temos um registro para cada dia que o mercado funcionou.

---

## 9. O que é MultiIndex e por que aparece no código?

**Referência:** Linhas 66, 76-77 do código

### O que é MultiIndex?

É quando um DataFrame tem **colunas com múltiplos níveis** de nome:

```python
# Coluna normal (um nível):
df['Close']

# Coluna MultiIndex (dois níveis):
df[('Close', 'PETR4.SA')]
```

### Por que o yfinance retorna MultiIndex?

Quando você baixa **múltiplas ações** de uma vez, o yfinance organiza assim:

```
              Close                  Volume
          PETR4.SA    VALE3.SA   PETR4.SA    VALE3.SA
Date
2018-01-02   4.40       12.50    33461800    15000000
```

Mas quando baixamos **uma única ação**, ele ainda pode vir com MultiIndex (comportamento da biblioteca).

### Como o código lida com isso

```python
# Linha 66: Detecta se é MultiIndex
close_col = ('Close', ticker) if isinstance(df.columns, pd.MultiIndex) else 'Close'

# Linhas 76-77: "Achata" o MultiIndex antes de salvar
if isinstance(df.columns, pd.MultiIndex):
    df.columns = [col[0] for col in df.columns]  # ('Close', 'PETR4.SA') → 'Close'
```

### Você precisa se preocupar com isso?

**Não!** O código já trata automaticamente. Mas é bom saber que existe caso veja algum erro relacionado.

---

## 10. Por que salvar em CSV?

**Referência:** Linhas 71-81 do código

### Motivos para Salvar

| Motivo | Explicação |
|--------|------------|
| **Velocidade** | Carregar de arquivo é ~50x mais rápido que baixar |
| **Offline** | Funciona sem internet |
| **Consistência** | Dados não mudam entre execuções |
| **Limite de API** | Evita bloqueio por muitas requisições |

### Por que CSV especificamente?

| Formato | Vantagem | Desvantagem |
|---------|----------|-------------|
| **CSV** | Universal, legível, leve | Mais lento para arquivos grandes |
| Parquet | Muito rápido, compacto | Menos legível |
| Excel | Interface visual | Pesado, lento |
| JSON | Flexível | Não ideal para tabelas |

Para ~1.500 linhas, CSV é perfeito - simples e suficiente.

---

## 🔗 Navegação

| Anterior | Próximo |
|----------|---------|
| [ETAPA 01 - Setup](./ETAPA_01_Setup_Ambiente.md) | [ETAPA 03 - Pré-processamento](./ETAPA_03_Preprocessamento.md) |

---

*Documento criado para esclarecer dúvidas comuns sobre a Etapa 2 do projeto LSTM.*
