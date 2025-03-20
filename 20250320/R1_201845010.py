import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

# 데이터 로드
data = pd.read_csv("Exam-112.csv")

# 결측치 처리: 각 열의 평균값으로 대체
data['math'].fillna(data['math'].mean(), inplace=True)
data['eng'].fillna(data['eng'].mean(), inplace=True)
data['science'].fillna(data['science'].mean(), inplace=True)

# 이상치 제거: IQR 방법 사용
Q1 = data[['math', 'eng', 'science']].quantile(0.25)
Q3 = data[['math', 'eng', 'science']].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

filtered_data = data[~((data[['math', 'eng', 'science']] < lower_bound) | (data[['math', 'eng', 'science']] > upper_bound)).any(axis=1)]

# 1. 수학 점수로 과학 점수 예측 (KNN 회귀)
X_science = filtered_data[['math']]
y_science = filtered_data['science']

X_train_science, X_test_science, y_train_science, y_test_science = train_test_split(X_science, y_science, test_size=0.2, random_state=42)

knn_regressor = KNeighborsRegressor(n_neighbors=5)
knn_regressor.fit(X_train_science, y_train_science)

def predict_science(math_score):
    prediction = knn_regressor.predict([[math_score]])
    return prediction[0]

# 2. 수학, 영어, 과학 점수로 합격 여부 예측 (KNN 분류)
X_pass = filtered_data[['math', 'eng', 'science']]
y_pass = filtered_data['pass']

label_encoder = LabelEncoder()
y_pass_encoded = label_encoder.fit_transform(y_pass)

X_train_pass, X_test_pass, y_train_pass, y_test_pass = train_test_split(X_pass, y_pass_encoded, test_size=0.2, random_state=42)

knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train_pass, y_train_pass)

def predict_pass(math_score, eng_score, science_score):
    prediction = knn_classifier.predict([[math_score, eng_score, science_score]])
    return label_encoder.inverse_transform(prediction)[0]

# 테스트: 수학 점수가 75일 때 과학 점수 예측
predicted_science_score = predict_science(75)
print(f"예측된 과학 점수: {predicted_science_score:.2f}")

# 테스트: 수학 75, 영어 70, 과학 80일 때 합격 여부 예측
predicted_status = predict_pass(75, 70, 80)
print(f"예측된 합격 여부: {predicted_status}")
