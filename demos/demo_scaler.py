from sklearn.preprocessing import StandardScaler
import numpy as np

# �������� ������� ������ StandardScaler
scaler = StandardScaler(with_mean=False, with_std=False)

# ���� �� ������
for data_chunk in data_generator:
    # ���������� ������������ �� ��������������� ����������
    scaler.partial_fit(data_chunk)
    
# ���������� �������������� ��������
mean = np.mean(scaler.mean_)
std = np.sqrt(np.mean(scaler.var_))

# ������������ ������
for data_chunk in data_generator:
    scaled_data_chunk = (data_chunk - mean) / std
    # ����������� ��������� ������...
