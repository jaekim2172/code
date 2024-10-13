import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import math

# CSV 파일 로드를 함수로 분리하여 재사용성을 높임
def load_data(file_name):
    df = pd.read_csv(file_name)
    return np.array(df.math), np.array(df.cs)

# scikit-learn을 사용한 예측 함수
def predict_using_sklean(x, y):
    r = LinearRegression()
    r.fit(x.reshape(-1, 1), y)
    return r.coef_[0], r.intercept_

# Gradient Descent 함수
def gradient_descent(x, y, learning_rate=0.0002, iterations=1000000):
    m_curr = 0
    b_curr = 0
    n = len(x)
    cost_previous = float('inf')  # 초기값을 무한대로 설정

    for i in range(iterations):
        y_predicted = m_curr * x + b_curr
        cost = 0  # 초기 cost 값 설정
        for j in range(n):  # 리스트 내포 대신 for 루프 사용
            cost += (y[j] - y_predicted[j]) ** 2  # 제곱 오차 계산
            print("cost", j, cost)
        cost = cost / n  # 평균으로 나누기

        # 기울기 계산
        md = -(2/n) * sum(x * (y - y_predicted))
        bd = -(2/n) * sum(y - y_predicted)
        print("md, bd", md, bd)

        # 가중치 업데이트
        m_curr -= learning_rate * md
        b_curr -= learning_rate * bd

        # 수렴 조건
        if math.isclose(cost, cost_previous, rel_tol=1e-8):
            break
        cost_previous = cost

    return m_curr, b_curr

# 메인 함수
def main():
    x, y = load_data("test_scores.csv")  # 데이터 로드

    # Gradient Descent 사용
    m, b = gradient_descent(x, y)
    print(f"Using gradient descent function: Coef {m} Intercept {b}")

    # scikit-learn 사용
    m_sklearn, b_sklearn = predict_using_sklean(x, y)
    print(f"Using sklearn: Coef {m_sklearn} Intercept {b_sklearn}")

# 프로그램 시작점
if __name__ == "__main__":
    print("name",__name__)
    main()