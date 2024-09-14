# Predicción del Precio de Acciones usando Análisis de Sentimientos y Redes LSTM

Este proyecto final de la materia de Procesamiento de Lenguaje Natural desarrolla un modelo para predecir el precio de acciones utilizando datos financieros, noticias y análisis de sentimientos.

## Introducción
El modelo se construye utilizando LSTM y datos extraídos de Yahoo Finance, FRED, y análisis de sentimientos con la herramienta VADER.

![Gráfico Predicción](graficos/prediccion_acciones.png)

## Fragmento de Código
```python
# Ejemplo de modelo LSTM
import torch.nn as nn
class LSTMModel(nn.Module):
    ...

