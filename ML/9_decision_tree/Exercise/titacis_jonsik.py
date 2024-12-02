import pandas as pd

# titanic.csv 파일 읽기
df = pd.read_csv('titanic.csv')

# 데이터프레임 출력
print(df.head())  # 처음 5행 출력