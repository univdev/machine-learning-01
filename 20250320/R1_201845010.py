import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# 파일 경로 설정
file_path = "Exam-112.csv"

# 데이터 불러오기
df = pd.read_csv(file_path)

# 결측치 처리
for col in df.columns:
    if df[col].dtype == 'object':  # 범주형 데이터
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:  # 수치형 데이터
        df[col].fillna(df[col].mean(), inplace=True)

# 이상치 처리 (IQR 방식)
def remove_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

# 수치형 컬럼에 대해 이상치 제거
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    df = remove_outliers(df, col)

# 전처리된 데이터 저장
processed_file_path = "Processed_Exam-112.csv"
df.to_csv(processed_file_path, index=False)

# ------------------------- KNN 회귀 (math 점수 → science 점수 예측) -------------------------
def predict_science_score():
    if "math" in df.columns and "science" in df.columns:
        X = df[["math"]].values  # 입력 변수 (math 점수)
        y = df["science"].values  # 타겟 변수 (science 점수)

        # 데이터 분할 (80% 학습, 20% 테스트)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 데이터 스케일링 (KNN은 거리 기반이므로 정규화 필요)
        scaler_math = StandardScaler()
        X_train = scaler_math.fit_transform(X_train)
        X_test = scaler_math.transform(X_test)

        # KNN 모델 생성 및 학습
        knn_reg = KNeighborsRegressor(n_neighbors=5)
        knn_reg.fit(X_train, y_train)

        # 사용자 입력 받아 science 점수 예측
        math_score = float(input("예측할 math 점수를 입력하세요: "))
        math_score_scaled = scaler_math.transform([[math_score]])  # 입력값 스케일 변환
        predicted_science_score = knn_reg.predict(math_score_scaled)

        print(f"예측된 science 점수: {predicted_science_score[0]:.2f}")
    else:
        print("데이터셋에 'math' 또는 'science' 컬럼이 없습니다.")

# ------------------------- KNN 분류 (math, eng, science 점수 → 합격 여부 예측) -------------------------
def predict_pass_status():
    if {"math", "eng", "science", "pass"}.issubset(df.columns):
        X_classification = df[["math", "eng", "science"]].values  # 입력 변수
        y_classification = df["pass"].values  # 타겟 변수

        # 데이터 분할 (80% 학습, 20% 테스트)
        X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
            X_classification, y_classification, test_size=0.2, random_state=42
        )

        # 데이터 스케일링
        scaler_class = StandardScaler()
        X_train_c = scaler_class.fit_transform(X_train_c)
        X_test_c = scaler_class.transform(X_test_c)

        # KNN 분류 모델 생성 및 학습
        knn_clf = KNeighborsClassifier(n_neighbors=5)
        knn_clf.fit(X_train_c, y_train_c)

        # 사용자 입력 받아 합격 여부 예측
        math_score = float(input("math 점수를 입력하세요: "))
        english_score = float(input("eng 점수를 입력하세요: "))
        science_score = float(input("science 점수를 입력하세요: "))

        user_input_scaled = scaler_class.transform([[math_score, english_score, science_score]])
        predicted_result = knn_clf.predict(user_input_scaled)

        print(f"예측된 합격 여부: {'합격' if predicted_result[0] == 'Pass' else '불합격'}")
    else:
        print("데이터셋에 'math', 'eng', 'science', 'pass' 컬럼이 없습니다.")

# ------------------------- 사용자 선택 -------------------------
while True:
    print("\n===== 예측 시스템 =====")
    print("1. 수학 점수로 과학 점수 예측")
    print("2. 수학, 과학, 영어 점수로 합불 여부 예측")
    print("3. 종료")
    
    choice = input("원하는 기능의 번호를 입력하세요: ")
    
    if choice == "1":
        predict_science_score()
    elif choice == "2":
        predict_pass_status()
    elif choice == "3":
        print("프로그램을 종료합니다.")
        break
    else:
        print("올바른 번호를 입력하세요. (1, 2, 3)")
