# Predicción del Precio de Acciones usando Análisis de Sentimientos y Redes LSTM

### Proyecto Final de la Materia de Procesamiento de Lenguaje Natural

---

## Descripción

Este proyecto final tiene como objetivo predecir el precio de acciones utilizando un modelo de red neuronal LSTM (Long Short-Term Memory) entrenado con datos financieros históricos, noticias y análisis de sentimientos. El proyecto incluye el uso de datos extraídos de fuentes como Yahoo Finance, FRED, y análisis de sentimientos con la herramienta VADER.

---

**Autor**: [Tu Nombre]  
**Fecha**: Septiembre 2024  
**Universidad**: [Nombre de la Universidad]  

**Enlaces**:  
- [Repositorio completo en GitHub](https://github.com/tu-usuario/pln-lstm-stock-prediction)  
- [Notebook en Google Colab](#) *(si es necesario)*

---

![Predicción del Precio de Acciones](ruta/a/grafico_prediccion.png)

## Introducción

La predicción de precios de acciones es un desafío clave en las finanzas debido a la naturaleza compleja y volátil de los mercados. En este proyecto, se utiliza una combinación de datos financieros históricos y el análisis de sentimientos basado en noticias para mejorar la precisión de las predicciones.

El modelo principal es una red neuronal de tipo LSTM, adecuada para el análisis de series temporales. Además, se incorporan técnicas de procesamiento de lenguaje natural (PLN) para analizar el sentimiento de las noticias financieras y medir su impacto en los precios de las acciones.

Los principales objetivos del proyecto son:

1. **Predecir el precio de acciones** utilizando una red LSTM basada en datos históricos de precios y volúmenes.
2. **Incorporar análisis de sentimientos** de noticias relacionadas con los mercados financieros para mejorar la predicción.
3. **Comparar el rendimiento** de un modelo basado únicamente en datos financieros versus uno que también incluye datos de sentimientos.

### Datos Utilizados

Los siguientes datos se emplean en este estudio:

- **Datos financieros**: Extraídos de Yahoo Finance, incluyendo precios históricos, volúmenes, y otros indicadores técnicos.
- **Datos macroeconómicos**: Extraídos de FRED, incluyendo tasas de interés y otros factores macroeconómicos relevantes.
- **Noticias y análisis de sentimientos**: Análisis de noticias relacionadas con las empresas seleccionadas, usando la herramienta de análisis de sentimientos VADER.

![Datos de Acciones](ruta/a/imagen_datos_acciones.png)

## Preparación de Datos

El éxito de un modelo de predicción depende en gran medida de la calidad de los datos utilizados. En este proyecto, se recopilan datos financieros históricos y noticias de varias fuentes. A continuación, se describen las etapas de preparación de los datos.

### 1. Fuentes de Datos

- **Yahoo Finance**: Se utilizaron los precios históricos ajustados de las acciones de las empresas seleccionadas, así como otros indicadores técnicos relevantes como el volumen de transacciones.
- **FRED (Federal Reserve Economic Data)**: Se obtuvieron datos macroeconómicos que incluyen tasas de interés, inflación, entre otros, para entender el contexto general del mercado.
- **Noticias y Análisis de Sentimientos**: Las noticias relacionadas con las empresas seleccionadas fueron obtenidas y procesadas mediante la herramienta VADER para realizar un análisis de sentimientos. Este análisis se utilizó como una variable adicional en el modelo para evaluar su impacto en los precios de las acciones.

![Fuentes de Datos](ruta/a/imagen_fuentes_datos.png)

### 2. Extracción de Datos Financieros

Se utilizó la biblioteca `yahoo_fin` para extraer los datos financieros de las acciones seleccionadas (AMZN, AAPL, GOOGL). A continuación, se muestra un fragmento del código utilizado para la descarga de datos:

```python
import pandas as pd
from yahoo_fin import stock_info as si

# Definir los tickers de las acciones a analizar
tickers = ['AAPL', 'AMZN', 'GOOGL']

# Descargar los datos financieros históricos para cada ticker
precios = {ticker: si.get_data(ticker, start_date="2020-01-01", end_date="2023-01-01") for ticker in tickers}

# Mostrar los primeros registros de los datos de AAPL
precios['AAPL'].head()```

## Preparación de Datos

El éxito de un modelo de predicción depende en gran medida de la calidad de los datos utilizados. En este proyecto, se recopilan datos financieros históricos y noticias de varias fuentes. A continuación, se describen las etapas de preparación de los datos.

### 1. Fuentes de Datos

- **Yahoo Finance**: Se utilizaron los precios históricos ajustados de las acciones de las empresas seleccionadas, así como otros indicadores técnicos relevantes como el volumen de transacciones.
- **FRED (Federal Reserve Economic Data)**: Se obtuvieron datos macroeconómicos que incluyen tasas de interés, inflación, entre otros, para entender el contexto general del mercado.
- **Noticias y Análisis de Sentimientos**: Las noticias relacionadas con las empresas seleccionadas fueron obtenidas y procesadas mediante la herramienta VADER para realizar un análisis de sentimientos. Este análisis se utilizó como una variable adicional en el modelo para evaluar su impacto en los precios de las acciones.

![Fuentes de Datos](ruta/a/imagen_fuentes_datos.png)

### 2. Extracción de Datos Financieros

Se utilizó la biblioteca `yahoo_fin` para extraer los datos financieros de las acciones seleccionadas (AMZN, AAPL, GOOGL). A continuación, se muestra un fragmento del código utilizado para la descarga de datos:

```python
import pandas as pd
from yahoo_fin import stock_info as si

# Definir los tickers de las acciones a analizar
tickers = ['AAPL', 'AMZN', 'GOOGL']

# Descargar los datos financieros históricos para cada ticker
precios = {ticker: si.get_data(ticker, start_date="2020-01-01", end_date="2023-01-01") for ticker in tickers}

# Mostrar los primeros registros de los datos de AAPL
precios['AAPL'].head()```

## Preparación de Datos

En este proyecto se utilizaron tres tipos de datos principales: datos financieros históricos, datos macroeconómicos y análisis de sentimientos extraídos de noticias. A continuación, se detallan los pasos clave de la preparación de datos.

### 1. Fuentes de Datos

- **Yahoo Finance**: Precios históricos ajustados y volumen de transacciones de las acciones de AMZN, AAPL y GOOGL.
- **FRED**: Datos macroeconómicos relevantes como las tasas de interés.
- **Análisis de Sentimientos de Noticias**: Se utilizó VADER para analizar noticias relacionadas con las empresas, obteniendo una puntuación de sentimiento para cada noticia.

### 2. Extracción de Datos Financieros

Para extraer los datos financieros históricos, se utilizó la biblioteca `yahoo_fin`. A continuación, se muestra un ejemplo del código utilizado para obtener los datos:

```python
import pandas as pd
from yahoo_fin import stock_info as si

# Extraer datos financieros de las acciones seleccionadas
tickers = ['AAPL', 'AMZN', 'GOOGL']
precios = {ticker: si.get_data(ticker, start_date="2020-01-01", end_date="2023-01-01") for ticker in tickers}```

Los datos financieros incluyen precios de cierre ajustados, volúmenes y otros indicadores técnicos.

### 3. Análisis Exploratorio de Datos (EDA)
Se realizó un análisis exploratorio de los datos financieros. A continuación, se presenta un gráfico de los precios ajustados de las acciones a lo largo del tiempo:


### 4. Análisis de Sentimientos
Para incorporar el análisis de sentimientos, se utilizaron noticias financieras recientes, y se analizó el sentimiento utilizando la herramienta VADER. Aquí un ejemplo de una noticia extraída:

Noticia: "Apple's new iPhone sales exceeded expectations, boosting investor confidence."

El análisis de sentimientos generó el siguiente resultado:

```python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()
noticia = "Apple's new iPhone sales exceeded expectations, boosting investor confidence."
sentimiento = analyzer.polarity_scores(noticia)
print(sentimiento)```

Resultado:
{'neg': 0.0, 'neu': 0.461, 'pos': 0.539, 'compound': 0.7269}

### 5. Integración de Datos
Finalmente, los datos financieros y los datos de sentimientos se integraron en un solo conjunto de datos, que se utilizó para entrenar el modelo LSTM:

```python
# Unir los datos financieros y los de sentimiento
datos_combinados = pd.merge(precios['AAPL'], sentimientos_df, left_index=True, right_index=True)

# Mostrar los primeros registros de los datos combinados
datos_combinados.head()```

## Modelo de LSTM para la Predicción de Precios

El modelo utilizado en este proyecto es una red neuronal de tipo LSTM (Long Short-Term Memory), ideal para la predicción de series temporales debido a su capacidad de aprender dependencias a largo plazo. A continuación, se describen los detalles de la implementación del modelo.

### 1. Estructura del Modelo

El modelo LSTM se implementó utilizando `PyTorch Lightning` para facilitar la gestión del entrenamiento y el seguimiento de las métricas de rendimiento. La red está diseñada para predecir el precio de cierre ajustado de las acciones basadas en datos históricos y análisis de sentimientos.

```python
import torch
import torch.nn as nn
import pytorch_lightning as pl

class LSTMModel(pl.LightningModule):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h_lstm, _ = self.lstm(x)
        out = self.fc(h_lstm[:, -1, :])  # Tomamos solo la última salida
        return out```
## Modelo de LSTM para la Predicción de Precios

El modelo utilizado en este proyecto es una red neuronal de tipo LSTM (Long Short-Term Memory), ideal para la predicción de series temporales debido a su capacidad de aprender dependencias a largo plazo. A continuación, se describen los detalles de la implementación del modelo.

### 1. Estructura del Modelo

El modelo LSTM se implementó utilizando `PyTorch Lightning` para facilitar la gestión del entrenamiento y el seguimiento de las métricas de rendimiento. La red está diseñada para predecir el precio de cierre ajustado de las acciones basadas en datos históricos y análisis de sentimientos.

```python
import torch
import torch.nn as nn
import pytorch_lightning as pl

class LSTMModel(pl.LightningModule):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h_lstm, _ = self.lstm(x)
        out = self.fc(h_lstm[:, -1, :])  # Tomamos solo la última salida
        return out
## Modelo de LSTM para la Predicción de Precios

El modelo utilizado en este proyecto es una red neuronal de tipo LSTM (Long Short-Term Memory), ideal para la predicción de series temporales debido a su capacidad de aprender dependencias a largo plazo. A continuación, se describen los detalles de la implementación del modelo.

### 1. Estructura del Modelo

El modelo LSTM se implementó utilizando `PyTorch Lightning` para facilitar la gestión del entrenamiento y el seguimiento de las métricas de rendimiento. La red está diseñada para predecir el precio de cierre ajustado de las acciones basadas en datos históricos y análisis de sentimientos.

```python
import torch
import torch.nn as nn
import pytorch_lightning as pl

class LSTMModel(pl.LightningModule):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h_lstm, _ = self.lstm(x)
        out = self.fc(h_lstm[:, -1, :])  # Tomamos solo la última salida
        return out
## Modelo de LSTM para la Predicción de Precios

El modelo utilizado en este proyecto es una red neuronal de tipo LSTM (Long Short-Term Memory), ideal para la predicción de series temporales debido a su capacidad de aprender dependencias a largo plazo. A continuación, se describen los detalles de la implementación del modelo.

### 1. Estructura del Modelo

El modelo LSTM se implementó utilizando `PyTorch Lightning` para facilitar la gestión del entrenamiento y el seguimiento de las métricas de rendimiento. La red está diseñada para predecir el precio de cierre ajustado de las acciones basadas en datos históricos y análisis de sentimientos.

```python
import torch
import torch.nn as nn
import pytorch_lightning as pl

class LSTMModel(pl.LightningModule):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h_lstm, _ = self.lstm(x)
        out = self.fc(h_lstm[:, -1, :])  # Tomamos solo la última salida
        return out
## Modelo de LSTM para la Predicción de Precios

El modelo utilizado en este proyecto es una red neuronal de tipo LSTM (Long Short-Term Memory), ideal para la predicción de series temporales debido a su capacidad de aprender dependencias a largo plazo. A continuación, se describen los detalles de la implementación del modelo.

### 1. Estructura del Modelo

El modelo LSTM se implementó utilizando `PyTorch Lightning` para facilitar la gestión del entrenamiento y el seguimiento de las métricas de rendimiento. La red está diseñada para predecir el precio de cierre ajustado de las acciones basadas en datos históricos y análisis de sentimientos.

```python
import torch
import torch.nn as nn
import pytorch_lightning as pl

class LSTMModel(pl.LightningModule):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h_lstm, _ = self.lstm(x)
        out = self.fc(h_lstm[:, -1, :])  # Tomamos solo la última salida
        return out
## Modelo de LSTM para la Predicción de Precios

El modelo utilizado en este proyecto es una red neuronal de tipo LSTM (Long Short-Term Memory), ideal para la predicción de series temporales debido a su capacidad de aprender dependencias a largo plazo. A continuación, se describen los detalles de la implementación del modelo.

### 1. Estructura del Modelo

El modelo LSTM se implementó utilizando `PyTorch Lightning` para facilitar la gestión del entrenamiento y el seguimiento de las métricas de rendimiento. La red está diseñada para predecir el precio de cierre ajustado de las acciones basadas en datos históricos y análisis de sentimientos.

```python
import torch
import torch.nn as nn
import pytorch_lightning as pl

class LSTMModel(pl.LightningModule):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h_lstm, _ = self.lstm(x)
        out = self.fc(h_lstm[:, -1, :])  # Tomamos solo la última salida
        return out
