import numpy as np
import matplotlib.pyplot as plt

# 데이터 생성
np.random.seed(0)
x_data = np.linspace(-10, 10, 100)  # 입력 데이터 (100개의 점)
y_data = (x_data > 0).astype(float)  # 0보다 크면 1, 작으면 0

print(x_data, y_data)