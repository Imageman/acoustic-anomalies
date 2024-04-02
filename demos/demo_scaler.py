from sklearn.preprocessing import StandardScaler
import numpy as np

# Создание объекта класса StandardScaler
scaler = StandardScaler(with_mean=False, with_std=False)

# Цикл по данным
for data_chunk in data_generator:
    # Обновление нормализации со статистическими значениями
    scaler.partial_fit(data_chunk)
    
# Вычисление статистических значений
mean = np.mean(scaler.mean_)
std = np.sqrt(np.mean(scaler.var_))

# Нормализация данных
for data_chunk in data_generator:
    scaled_data_chunk = (data_chunk - mean) / std
    # Продолжение обработки данных...
