import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 데이터 생성



print(x_data, y_data)


# 데이터 산점도 그리기
plt.scatter(x_data, y_data, color='blue', label="True Data")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Scatter Plot of x_data and y_data")
plt.legend()
plt.show()

# 시그모이드 함수 정의
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 예측 함수
def predict(x, w, b):
    z = w * x + b
    return sigmoid(z)

# 손실 함수 (크로스 엔트로피) 계산
def cross_entropy_loss(y, y_pred):
    return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

# 모델 초기화
w = np.random.randn()  # 가중치 초기값
b = np.random.randn()  # 절편 초기값
learning_rate = 0.01  # 학습률
num_epochs = 1000  # 학습 반복 횟수

# 경사하강법을 사용한 학습
loss_values = []
for epoch in range(num_epochs):
    # 예측값 계산
    y_pred = predict(x_data, w, b)
    
    # 손실 계산
    loss = cross_entropy_loss(y_data, y_pred)
    loss_values.append(loss)
    
    # 미분을 통한 기울기 계산 (크로스 엔트로피의 w, b에 대한 편미분)
    dw = np.mean((y_pred - y_data) * x_data)
    db = np.mean(y_pred - y_data)
    
    # 가중치 및 절편 업데이트
    w = w - learning_rate * dw
    b = b - learning_rate * db
    
    # 100번마다 손실 출력
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}")

# 학습 결과 시각화
plt.plot(range(num_epochs), loss_values)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Cross Entropy Loss over Epochs")
plt.show()

# 예측 결과 시각화
y_pred_final = predict(x_data, w, b)
plt.scatter(x_data, y_data, label="True Data")
plt.plot(x_data, y_pred_final, color="red", label="Model Prediction")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Sigmoid Function Fitting using Cross Entropy and Gradient Descent")
plt.legend()
plt.show()
