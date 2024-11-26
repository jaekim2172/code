import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


# 데이터 불러오기
data = pd.read_csv("logi_data.csv")
x_data = data['x_data'].values.reshape(-1, 1)  # 입력 데이터
y_data = data['y_data'].values                 # 출력 데이터

print(x_data.shape,y_data.shape)

# 로지스틱 회귀 모델 학습


# 모델 학습
model = LogisticRegression()
model.fit(x_data, y_data)

# 예측값 계산
y_pred = model.predict_proba(x_data)[:, 1]  # 클래스 1에 대한 확률 예측
print("y_pred", y_pred.shape)

# 데이터 산점도와 예측 결과 시각화
plt.scatter(x_data, y_data, color='blue', label="True Data")
plt.plot(x_data, y_pred, color='red', label="Model Prediction (Sigmoid)")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Sigmoid Function Fitting using Logistic Regression")
plt.legend()
plt.show()






# 예측값 계산용 데이터 생성 (0.01 간격)
x_plot = np.arange(x_data.min(), x_data.max(), 0.01).reshape(-1, 1)
y_pred = model.predict_proba(x_plot)[:, 1]  # 클래스 1에 대한 확률 예측

# 데이터 산점도와 예측 결과 시각화
plt.scatter(x_data, y_data, color='blue', label="True Data")
plt.plot(x_plot, y_pred, color='red', label="Model Prediction (Sigmoid)")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Sigmoid Function Fitting using Logistic Regression")
plt.legend()
plt.show()
