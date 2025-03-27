# 성적 예측 시스템

이 프로젝트는 선형 회귀 모델을 사용하여 공부 시간에 따른 성적을 예측하는 시스템입니다. `Exam-112.csv` 데이터셋을 이용하여 학습하며, Scikit-Learn 라이브러리를 활용해 모델을 학습하고 평가합니다.

## 📂 프로젝트 구조

```
📁 프로젝트 폴더
│── Exam-112.csv  # 성적 데이터셋
│── LR_박찬영.py  # 성적 예측 코드
│── README.md  # 프로젝트 설명 파일
```

## 🛠️ 사용 라이브러리

이 프로젝트에서는 다음과 같은 Python 라이브러리를 사용합니다:

- `pandas` : 데이터 로드 및 전처리
- `numpy` : 수학 연산
- `matplotlib` : 데이터 시각화
- `sklearn` : 모델 학습 및 평가

## 🚀 실행 방법

1. 필요한 라이브러리를 설치합니다.
   ```bash
   pip install pandas numpy matplotlib scikit-learn
   ```
2. `predict_scores.py` 스크립트를 실행합니다.
   ```bash
   python predict_scores.py
   ```

## 📊 코드 설명 (`predict_scores.py`)

1. `Exam-112.csv` 데이터를 로드하고 결측값을 제거합니다.
2. 공부 시간(`hours`)을 입력, 성적(`score`)을 출력으로 설정합니다.
3. `train_test_split`을 사용하여 80%를 훈련 데이터, 20%를 테스트 데이터로 분리합니다.
4. `LinearRegression` 모델을 학습하고 예측을 수행합니다.
5. 모델의 MSE(평균 제곱 오차)와 R² 점수를 출력합니다.
6. 결과를 그래프로 시각화합니다.

## 📈 예제 결과

```plaintext
Mean Squared Error (MSE): 335.70
R-squared (R² Score): 0.734
```

## 📌 참고 사항

- `random_state=42`를 설정하여 실행할 때마다 같은 결과가 나오도록 설정하였습니다.
- 더 정밀한 분석을 위해 다중 선형 회귀 모델을 확장할 수도 있습니다.

## 📧 문의

프로젝트 관련 문의사항이 있다면 연락 주세요! 🚀
