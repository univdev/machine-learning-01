# 성적 데이터셋 (Exam-112.csv) 에서 공부 시간에 따른 성적 예측 시스템 구현
# 1. 성적 데이터를 불러온다.
# 2. 데이터를 훈련 데이터와 테스트 데이터로 분리한다.
# 3. 훈련 데이터를 사용하여 선형 회귀 모델을 학습한다.
# 4. 테스트 데이터를 사용하여 모델의 성능을 평가한다.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 데이터 불러오기
file_path = "Exam-112.csv"  # 파일 경로를 올바르게 설정하세요.
df = pd.read_csv(file_path)

# 결측값 제거
df_clean = df[['hours', 'score']].dropna()

# 입력(X)와 출력(y) 설정
X = df_clean[['hours']]
y = df_clean['score']

# 훈련 데이터와 테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 선형 회귀 모델 생성 및 학습
model = LinearRegression()
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# 모델 평가
mse = mean_squared_error(y_test, y_pred)
# 회귀 모델의 결정 계수
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R² Score): {r2}")

# 시각화
plt.scatter(X_test, y_test, color="blue", label="Actual Scores")
plt.plot(X_test, y_pred, color="red", linewidth=2, label="Regression Line")
plt.xlabel("Study Hours")
plt.ylabel("Score")
plt.title("Study Hours vs. Score Regression")
plt.legend()
plt.show()
