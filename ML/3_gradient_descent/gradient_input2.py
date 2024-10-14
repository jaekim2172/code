import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# CSV 파일에서 데이터 읽기
data = pd.read_csv('data.csv') # ('data.csv', header=None) 첫번째 head의 설명이 없는 경우

# X와 y 변수로 분리 (첫 번째 열을 X, 두 번째 열을 y로 가정)
X = data.iloc[:, 0].values  # 첫 번째 열을 X 값으로 사용
y = data.iloc[:, 1].values  # 두 번째 열을 y 값으로 사용

# 경사 하강법 파라미터 설정
learning_rate = 0.01  # 더 작은 학습률 설정
n_iterations = 20000    # 최대 반복 횟수 설정
tolerance = 0.00001     # da와 db가 이 값보다 작아지면 중단

# 초기값 설정 (a = 기울기, b = 절편)
a = 0  # 기울기를 0으로 초기화
b = 0  # 절편을 0으로 초기화

# 경사 하강법
for iteration in range(n_iterations):
    # 예측값 계산
    y_pred = a * X + b
    
    # 비용 함수의 그래디언트 계산
    da = -(2) * np.sum((y - y_pred) * X)
    db = -(2) * np.sum(y - y_pred)
    
    # 파라미터 업데이트
    a = a - learning_rate * da
    b = b - learning_rate * db
    
    # da와 db가 임계값보다 작아지면 계산 종료
    if abs(da) < tolerance and abs(db) < tolerance:
        print(f"Converged after {iteration} iterations.")
        break

# 최적화된 a, b 출력
print(f"최적화된 기울기 a: {a}")
print(f"최적화된 절편 b: {b}")

# 회귀선 시각화
plt.scatter(X, y, color='blue', label='Data Points')  # 실제 데이터

# -2에서 5까지의 X 값을 사용하여 회귀선 그리기
X_ext = np.linspace(-2, 5, 100)  # -2에서 5까지의 X 값 생성
y_pred_ext = a * X_ext + b  # 확장된 구간의 회귀선 예측값 계산
plt.plot(X_ext, y_pred_ext, color='red', label='Regression Line')  # 확장된 회귀선

plt.xlim([-2, 5])  # X축 범위 설정
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression using Gradient Descent')
plt.legend()
plt.show()