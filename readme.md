# :thermometer: PyTorch Weather Prediction

## :memo: Descripción
Este proyecto utiliza una red neuronal LSTM en PyTorch para predecir temperaturas máximas y mínimas a partir de datos climáticos históricos. Se basa en datos recopilados diariamente durante el año 2024, que incluyen información sobre temperatura, precipitación, viento y humedad.

## 	:sparkles: Características
- Implementación de un modelo de **red neuronal LSTM** para la predicción de series temporales.
- Preprocesamiento de datos climáticos desde archivos CSV.
- Normalización de datos con **MinMaxScaler** de `sklearn`.
- Entrenamiento del modelo con PyTorch.
- Visualización de los resultados mediante gráficos.

## :pushpin: Requisitos
Para ejecutar este proyecto, es necesario tener instaladas las siguientes bibliotecas:

- Python 3.8+

- PyTorch

- Pandas

- Scikit-learn

- Matplotlib (opcional, para visualización)

Instala las dependencias con:

```bash
pip install -r requirements.txt
```

## 	:rocket: Uso

1. Coloca el archivo de datos en la carpeta data/.

2. Ejecuta el script principal:
```
python main.py
```

3. El modelo entrenará y generará predicciones de temperaturas máximas y mínimas.

## 	:bar_chart: Datos

Los datos meteorológicos provienen de registros diarios e incluyen:

- `t_max`: Temperatura máxima diaria

- `t_min`: Temperatura mínima diaria

- `precipitacion`: Nivel de precipitación (mm)

- `viento`: Velocidad del viento (km/h)

- `humedad`: Humedad relativa (%)

## :file_folder: Estructura del repositorio

```
📂 pytorch-weather-prediction
│── data/                         # Carpeta para almacenar los datos CSV
│   ├── clima_limpio_2024.csv     # Datos climáticos usados
│── 📄 main.ipynb    # Código completo del modelo en jupyter notebook
│── 📄 main.py    # Código completo del modelo
│── 📄 README.md    # Documentación del proyecto
│── 📄 requirements.txt    # Lista de dependencias

```

## :mag: Notas
* El modelo usa una ventana de 30 días para predecir las temperaturas.

* La normalización se aplica para mejorar el rendimiento del modelo.

## :chart_with_upwards_trend: Resultados
Tras el entrenamiento, el modelo logra aprender tendencias en la temperatura y genera predicciones razonables basadas en los datos históricos.

Ejemplo de gráfico de predicciones:

![Predicción de temperaturas](https://github.com/pcanadas/pytorch-weather-prediction/blob/main/Figure_1.png)

## :construction: Futuras mejoras
- Incluir más variables climáticas para mejorar la precisión.
- Optimizar los hiperparámetros del modelo.
- Implementar una API para consultar predicciones en tiempo real.

## 	:bust_in_silhouette: Autor
**Patricia Cañadas**  
Si tienes preguntas o sugerencias, no dudes en abrir un issue en el repositorio.

## 	:scroll: Licencia
Este proyecto está bajo la licencia MIT.

