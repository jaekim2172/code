# 필수 라이브러리 임포트
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets

# Iris 데이터셋 로드
iris = datasets.load_iris()


# 데이터를 DataFrame으로 변환하여 보기 쉽게 구성
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target  # 타겟 데이터를 추가 (species)
df['species'] = df['species'].apply(lambda x: iris.target_names[x])  # 숫자를 이름으로 변환

# 데이터 정보 출력
print("Iris 데이터셋 샘플:")
print(df.head(30))  # 상위 5개 샘플 출력
print("\n데이터셋 기본 정보:")
print(df.describe())  # 기본 통계 출력
# 데이터셋의 shape 확인
print("Iris 데이터셋의 shape:", df.shape)


# 데이터 시각화: 특성 간의 산점도
sns.pairplot(df, hue='species', markers=["o", "s", "D"])
plt.suptitle("Iris 데이터셋 특성 간 관계", y=1.02)
plt.show()
