import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime


# Cargar datos desde CSV
df = pd.read_csv("data/clima_limpio_2024.csv")

# Convertir la columna 'fecha' a formato datetime
df['fecha'] = pd.to_datetime(df['fecha'], format='%d/%m/%Y')

# Filtrar por una ciudad si es necesario (opcional)
df_ciudad = df[df['ciudad'] == 'Madrid']

# Ordenar los datos por fecha
df_ciudad = df_ciudad.sort_values('fecha')

# Imputar o eliminar valores faltantes
df_ciudad = df_ciudad.dropna()

# Eliminar unidades y convertir a números
df_ciudad['t_max'] = df_ciudad['t_max'].str.replace('°', '').astype(float)
df_ciudad['t_min'] = df_ciudad['t_min'].str.replace('°', '').astype(float)

# Limpiar precipitaciones, reemplazar comas por puntos y convertir a float
df_ciudad['precipitacion'] = df_ciudad['precipitacion'].str.replace(' mm', '').str.replace(',', '.').astype(float)

# Limpiar viento, eliminar ' Km/h' y convertir a float
df_ciudad['viento'] = df_ciudad['viento'].str.replace(' Km/h', '').astype(float)

# Limpiar humedad, eliminar '%' y convertir a float
df_ciudad['humedad'] = df_ciudad['humedad'].str.replace('%', '').astype(float)

# Verifica los cambios
print(df_ciudad.head())

# Seleccionar las columnas numéricas que usarás para la predicción
features = ['t_max', 't_min', 'precipitacion', 'viento', 'humedad']
data = df_ciudad[features].values

# Normalizar los datos
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# Convertir la columna 'fecha' a formato datetime
df['fecha'] = pd.to_datetime(df['fecha'], format='%d/%m/%Y')

# Filtrar por una ciudad (opcional)
df_ciudad = df[df['ciudad'] == 'Madrid']

# Ordenar los datos por fecha
df_ciudad = df_ciudad.sort_values('fecha')

# Imputar o eliminar valores faltantes
df_ciudad = df_ciudad.dropna()

# Eliminar unidades y convertir a números
df_ciudad['t_max'] = df_ciudad['t_max'].str.replace('°', '').astype(float)
df_ciudad['t_min'] = df_ciudad['t_min'].str.replace('°', '').astype(float)

# Limpiar precipitaciones, reemplazar comas por puntos y convertir a float
df_ciudad['precipitacion'] = df_ciudad['precipitacion'].str.replace(' mm', '').str.replace(',', '.').astype(float)

# Limpiar viento, eliminar ' Km/h' y convertir a float
df_ciudad['viento'] = df_ciudad['viento'].str.replace(' Km/h', '').astype(float)

# Limpiar humedad, eliminar '%' y convertir a float
df_ciudad['humedad'] = df_ciudad['humedad'].str.replace('%', '').astype(float)

# Verifica los cambios
print(df_ciudad.head())

# Seleccionar las columnas numéricas que usarás para la predicción
features = ['t_max', 't_min', 'precipitacion', 'viento', 'humedad']
data = df_ciudad[features].values

# Normalizar los datos
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# Crear secuencias de datos para LSTM
def create_sequences(data, seq_length):
    sequences = []
    targets = []
    dates = []
    
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
        targets.append(data[i+seq_length, :2])  # Predicción de t_max y t_min
        dates.append(df_ciudad.iloc[i + seq_length]['fecha'])  # Fecha correspondiente

    return np.array(sequences), np.array(targets), np.array(dates)

SEQ_LENGTH = 30  # Usamos los últimos 30 días para predecir el siguiente
X, y, fechas = create_sequences(data_scaled, SEQ_LENGTH)

# Separar en conjunto de entrenamiento y prueba
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
fechas_test = fechas[train_size:]

# Convertir a tensores de PyTorch
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)


print("Forma de X_train:", X_train.shape)  # (n_samples, 30, 5)
print("Forma de y_train:", y_train.shape)  # (n_samples, 2)
print("Forma de X_test:", X_test.shape)  # (n_samples, 30, 5)
print("Forma de y_test:", y_test.shape)  # (n_samples, 2)

# Definir el modelo LSTM
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Hiperparámetros
input_size = 5  # Número de características
hidden_size = 50  # Número de neuronas en la capa oculta
num_layers = 2  # Capas de LSTM
output_size = 2  # Predicción de t_max y t_min
learning_rate = 0.001
num_epochs = 100
batch_size = 16

# Crear el modelo
model = LSTMModel(input_size, hidden_size, num_layers, output_size)
print(model)

# Definir la función de pérdida y el optimizador
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Entrenamiento del modelo
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"Época [{epoch+1}/{num_epochs}], Pérdida: {loss.item():.6f}")

print("Entrenamiento completado.")

# Evaluación del modelo
model.eval()
with torch.no_grad():
    y_pred = model(X_test)

# Calcular el error MSE en el conjunto de prueba
test_loss = criterion(y_pred, y_test)
print(f"Pérdida en el conjunto de prueba: {test_loss.item():.6f}")

# Desnormalizar predicciones y valores reales
y_pred_actual = scaler.inverse_transform(np.hstack((y_pred.numpy(), np.zeros((y_pred.shape[0], data.shape[1] - 2)))))[:, :2]
y_test_actual = scaler.inverse_transform(np.hstack((y_test.numpy(), np.zeros((y_test.shape[0], data.shape[1] - 2)))))[:, :2]

# Graficar predicciones vs valores reales con fechas
plt.figure(figsize=(12, 6))
plt.plot(fechas_test, y_test_actual[:, 0], label="T_max Real", color='blue')
plt.plot(fechas_test, y_pred_actual[:, 0], label="T_max Predicción", color='red', linestyle='dashed')
plt.plot(fechas_test, y_test_actual[:, 1], label="T_min Real", color='green')
plt.plot(fechas_test, y_pred_actual[:, 1], label="T_min Predicción", color='orange', linestyle='dashed')
plt.xlabel("Fecha")
plt.ylabel("Temperatura (°C)")
plt.legend()
plt.title("Predicción de Temperaturas Máximas y Mínimas usando LSTM")
plt.xticks(rotation=45)
plt.show()