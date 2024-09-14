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

