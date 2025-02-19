# PyTorch Weather Prediction

## Descripción
Este proyecto utiliza una red neuronal LSTM en PyTorch para predecir temperaturas máximas y mínimas a partir de datos climáticos históricos. Se basa en datos recopilados diariamente durante el año 2024, que incluyen información sobre temperatura, precipitación, viento y humedad.

## Características
- Implementación de un modelo de **red neuronal LSTM** para la predicción de series temporales.
- Preprocesamiento de datos climáticos desde archivos CSV.
- Normalización de datos con **MinMaxScaler** de `sklearn`.
- Entrenamiento del modelo con PyTorch.
- Visualización de los resultados mediante gráficos.

## Requisitos
Para ejecutar este proyecto, es necesario tener instaladas las siguientes bibliotecas:

```bash
pip install torch numpy pandas scikit-learn matplotlib
```

Si estás usando un entorno con `conda`, puedes instalarlo con:

```bash
conda install pytorch torchvision torchaudio -c pytorch
conda install numpy pandas scikit-learn matplotlib
```

## Uso

1. **Cargar y preprocesar los datos:**  
   Se limpia el conjunto de datos eliminando unidades (°C, mm, Km/h, %), se convierte la fecha a formato datetime y se normalizan los valores numéricos.

2. **Crear secuencias de entrenamiento:**  
   Se generan ventanas deslizantes de `30` días como entrada para predecir el siguiente día.

3. **Entrenar el modelo:**  
   Se entrena una red LSTM de dos capas con 50 neuronas en cada capa.

4. **Evaluar y visualizar los resultados:**  
   Se generan predicciones y se comparan con los valores reales mediante gráficos.

## Estructura del repositorio

```
📂 pytorch-weather-prediction
│── 📄 README.md  # Documentación del proyecto
│── 📄 main.ipynb    # Código completo del modelo en jupyter notebook
│── 📄 main.py    # Código completo del modelo
│── 📄 requirements.txt  # Lista de dependencias
│── 📄 data/clima_limpio_2024.csv  # Datos climáticos usados

```

## Resultados
Tras el entrenamiento, el modelo logra aprender tendencias en la temperatura y genera predicciones razonables basadas en los datos históricos.

Ejemplo de gráfico de predicciones:

![Predicción de temperaturas](https://github.com/pcanadas/pytorch-weather-prediction/blob/main/Figure_1.png)

## Futuras mejoras
- Incluir más variables climáticas para mejorar la precisión.
- Optimizar los hiperparámetros del modelo.
- Implementar una API para consultar predicciones en tiempo real.

## Autor
**Patricia Cañadas**  
Si tienes preguntas o sugerencias, no dudes en abrir un issue en el repositorio.

## Licencia
Este proyecto está bajo la licencia MIT.

