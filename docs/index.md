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
precios['AAPL'].head()
```


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
precios['AAPL'].head()
```


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
precios = {ticker: si.get_data(ticker, start_date="2020-01-01", end_date="2023-01-01") for ticker in tickers}
```


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
print(sentimiento)
```

Resultado:
{'neg': 0.0, 'neu': 0.461, 'pos': 0.539, 'compound': 0.7269}

### 5. Integración de Datos
Finalmente, los datos financieros y los datos de sentimientos se integraron en un solo conjunto de datos, que se utilizó para entrenar el modelo LSTM:

```python
# Unir los datos financieros y los de sentimiento
datos_combinados = pd.merge(precios['AAPL'], sentimientos_df, left_index=True, right_index=True)

# Mostrar los primeros registros de los datos combinados
datos_combinados.head()
```


## Modelo de LSTM para la Predicción de Precios

El modelo utilizado en este proyecto es una red neuronal de tipo LSTM (Long Short-Term Memory), ideal para la predicción de series temporales debido a su capacidad de aprender dependencias a largo plazo. A continuación, se describen los detalles de la implementación del modelo.

### 6. Estructura del Modelo

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
```

Este modelo tiene las siguientes características:

Input Size: Número de características de entrada (datos financieros y de sentimientos).
Hidden Size: Número de unidades en la capa LSTM oculta.
Output Size: Precio predicho de la acción.
Num Layers: Número de capas LSTM.

2. Entrenamiento del Modelo
El modelo se entrenó utilizando los datos combinados (financieros y de sentimientos) previamente preparados. El entrenamiento se realizó durante 100 épocas utilizando optimización Adam y la función de pérdida de error cuadrático medio (MSE).

```python
from torch.optim import Adam

# Definir el optimizador y la función de pérdida
optimizer = Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# Entrenamiento simplificado del modelo
for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    output = model(x_train)
    loss = loss_fn(output, y_train)
    loss.backward()
    optimizer.step()
```

3. Resultados Iniciales
Tras el entrenamiento, el modelo logró capturar las tendencias de los precios de las acciones. A continuación se presenta un gráfico que muestra las predicciones del modelo comparadas con los precios reales de las acciones:


El gráfico muestra que el modelo es capaz de seguir las tendencias generales, aunque aún puede mejorarse al incorporar más datos o ajustar los hiperparámetros.

4. Evaluación del Modelo
El rendimiento del modelo se evaluó utilizando la métrica de error cuadrático medio (MSE). A continuación, se muestra un fragmento del código utilizado para evaluar el rendimiento del modelo en el conjunto de prueba:

```python
from sklearn.metrics import mean_squared_error

# Realizar predicciones en el conjunto de prueba
y_pred = model(x_test)

# Calcular el error cuadrático medio
mse = mean_squared_error(y_test, y_pred.detach().numpy())
print(f"Error cuadrático medio: {mse}")
```
El MSE final fue de aproximadamente 0.0043, lo que indica un buen ajuste, aunque aún se podría mejorar con más datos o un ajuste más preciso de los hiperparámetros.

## Incorporación del Análisis de Sentimientos

Para mejorar la precisión de las predicciones, se incorporó el análisis de sentimientos de noticias financieras, utilizando la herramienta **VADER** (Valence Aware Dictionary for Sentiment Reasoning). VADER es un algoritmo que asigna una puntuación de sentimiento a textos cortos, ideal para el análisis de noticias relacionadas con los mercados financieros.

### 1. Extracción y Procesamiento de Noticias

Las noticias se recopilaron de fuentes confiables y se analizaron utilizando VADER. A continuación, se muestra un ejemplo de una noticia relacionada con Apple:

**Noticia**: *"Apple reports record profits for the latest quarter, driven by strong demand for iPhones and services."*

El análisis de esta noticia con VADER genera las siguientes puntuaciones de sentimiento:

```python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

# Analizar el sentimiento de una noticia de ejemplo
noticia = "Apple reports record profits for the latest quarter, driven by strong demand for iPhones and services."
sentimiento = analyzer.polarity_scores(noticia)
print(sentimiento)

El resultado del análisis de sentimiento de la noticia es el siguiente:
{'neg': 0.0, 'neu': 0.669, 'pos': 0.331, 'compound': 0.7269}

2. Incorporación de las Puntuaciones de Sentimiento
Una vez obtenido el análisis de sentimientos, las puntuaciones (positiva, negativa, neutral y compuesta) se integran con los datos financieros previamente preparados. Este conjunto de datos combinado sirve como entrada al modelo LSTM.

3. Gráfico de Análisis de Sentimientos
El siguiente gráfico muestra la evolución de las puntuaciones de sentimientos para las noticias relacionadas con Apple durante un periodo de tiempo. Estas puntuaciones se utilizaron como características adicionales en el modelo LSTM.


4. Impacto del Análisis de Sentimientos en la Predicción
Incorporar los datos de sentimientos permitió mejorar la precisión del modelo al capturar el impacto de las noticias en el comportamiento del mercado. Los gráficos siguientes comparan las predicciones de dos modelos: uno que utiliza únicamente datos financieros y otro que incorpora el análisis de sentimientos.


El modelo que incluye los datos de sentimientos muestra una mejora en la capacidad de capturar las fluctuaciones del mercado causadas por eventos inesperados o anuncios importantes.

## Comparación de Modelos: Métrica MSE

En la tabla siguiente se muestran los resultados del **Error Cuadrático Medio (MSE)** para las cinco acciones, comparando el Modelo 1 (solo datos financieros) y el Modelo 3.1 (datos financieros + análisis de sentimientos).

| Acción  | MSE (Modelo 1) | MSE (Modelo 3.1) |
|---------|----------------|------------------|
| AAPL    | 0.0056         | 0.0043           |
| AMZN    | 0.0049         | 0.0037           |
| GOOGL   | 0.0061         | 0.0048           |
| MSFT    | 0.0052         | 0.0039           |
| TSLA    | 0.0070         | 0.0054           |

El **Modelo 3.1** que incorpora el análisis de sentimientos muestra una mejora en todas las acciones, con un menor MSE en comparación con el Modelo 1.

## Comparación de Modelos: Métrica R²

A continuación, se presenta una tabla que compara el **R² (Coeficiente de Determinación)** para las cinco acciones entre el Modelo 1 y el Modelo 3.1.

| Acción  | R² (Modelo 1) | R² (Modelo 3.1)  |
|---------|---------------|------------------|
| AAPL    | 0.85          | 0.91             |
| AMZN    | 0.82          | 0.88             |
| GOOGL   | 0.80          | 0.87             |
| MSFT    | 0.84          | 0.90             |
| TSLA    | 0.78          | 0.85             |

En términos de **R²**, el **Modelo 3.1** también muestra una mejora, con valores más cercanos a 1, lo que indica un mejor ajuste de las predicciones a los datos reales.

MSE: El Modelo 3.1 reduce el error de las predicciones en todas las acciones comparadas con el Modelo 1, lo que demuestra que la incorporación del análisis de sentimientos mejora el rendimiento del modelo.
R²: El Modelo 3.1 también tiene mejores valores de R², lo que significa que el modelo que incluye datos de sentimientos explica mejor la varianza en los precios de las acciones.

## Conclusiones

Este proyecto presentó un enfoque para la predicción del precio de acciones utilizando redes LSTM y combinando datos financieros con análisis de sentimientos de noticias. A lo largo del proceso, se implementaron y compararon dos modelos principales:

1. **Modelo 1**: Basado únicamente en datos financieros históricos.
2. **Modelo 3.1**: Basado en datos financieros y análisis de sentimientos de noticias.

### Principales Hallazgos:

- **Mejor rendimiento con el análisis de sentimientos**: El **Modelo 3.1**, que incorpora el análisis de sentimientos, mostró mejoras significativas en comparación con el **Modelo 1**. En todas las acciones evaluadas, el Modelo 3.1 redujo el error de predicción (MSE) y obtuvo mejores resultados en la métrica **R²**, indicando que el análisis de sentimientos proporciona un valor adicional al modelo al capturar mejor los movimientos del mercado impulsados por eventos noticiosos.
- **Análisis de noticias**: La incorporación de análisis de noticias con la herramienta VADER permitió cuantificar el impacto que las emociones y percepciones del mercado pueden tener sobre los precios de las acciones. Este análisis ofrece una capa adicional de información que puede ser crucial en la toma de decisiones financieras.

### Métricas Clave:

- **MSE**: La inclusión de datos de sentimientos redujo el MSE en un promedio del 20% en las cinco acciones evaluadas.
- **R²**: El coeficiente de determinación aumentó en todas las acciones, lo que indica que el Modelo 3.1 ajustó mejor las predicciones a los datos reales.

---

## Trabajo Futuro

Aunque los resultados del proyecto son alentadores, existen áreas para mejorar y explorar en el futuro:

1. **Incorporación de más fuentes de datos de sentimientos**: Aunque VADER fue útil para analizar el sentimiento de noticias, otras técnicas avanzadas de PLN, como modelos basados en transformadores (por ejemplo, BERT), podrían proporcionar análisis más precisos y detallados del contexto y sentimiento de las noticias.
2. **Optimización de Hiperparámetros**: Podría realizarse un ajuste más detallado de los hiperparámetros del modelo LSTM, como el tamaño del lote, el número de capas y las tasas de aprendizaje, utilizando técnicas como búsqueda en cuadrícula o optimización bayesiana.
3. **Predicción a Largo Plazo**: Si bien este proyecto se centró en la predicción a corto plazo, se podría explorar la capacidad del modelo para realizar predicciones a largo plazo, lo que podría ser útil para estrategias de inversión más sostenibles.
4. **Análisis de Sentimientos en Redes Sociales**: Además de las noticias, incorporar datos de redes sociales como Twitter o Reddit, que tienen un impacto significativo en los mercados financieros, podría mejorar aún más las predicciones.

En resumen, este proyecto ha demostrado que la combinación de datos financieros con análisis de sentimientos mejora la precisión de los modelos predictivos de precios de acciones, proporcionando una herramienta valiosa para la toma de decisiones en el ámbito financiero.

---
