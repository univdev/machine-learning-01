# KNN Prediction Service

## 프로젝트 개요

이 프로젝트는 **K-최근접 이웃(KNN)** 알고리즘을 사용하여 다음 두 가지 예측 서비스를 제공합니다:

1. **수학 점수를 입력하면 과학 점수를 예측**합니다.
2. **수학, 영어, 과학 점수를 입력하면 합격 여부를 예측**합니다.

이 프로젝트는 Python과 scikit-learn 라이브러리를 사용하며, 데이터 전처리(결측치 처리 및 이상치 제거)를 포함합니다.

---

## 데이터 설명

데이터는 `Exam-112.csv` 파일에 저장되어 있으며, 주요 열은 다음과 같습니다:

- `math`: 수학 점수
- `eng`: 영어 점수
- `science`: 과학 점수
- `pass`: 합격 여부 (`Pass` 또는 `Fail`)

---

## 주요 기능

### 1. 수학 점수로 과학 점수를 예측 (KNN 회귀)

- **입력:** 수학 점수
- **출력:** 예측된 과학 점수
- **모델:** KNeighborsRegressor

### 2. 수학, 영어, 과학 점수로 합격 여부를 예측 (KNN 분류)

- **입력:** 수학, 영어, 과학 점수
- **출력:** 합격 여부 (`Pass` 또는 `Fail`)
- **모델:** KNeighborsClassifier

---

## 데이터 전처리

### 1. 결측치 처리

- 각 열의 평균값으로 결측치를 대체합니다.
  ```python
  data['math'].fillna(data['math'].mean(), inplace=True)
  data['eng'].fillna(data['eng'].mean(), inplace=True)
  data['science'].fillna(data['science'].mean(), inplace=True)
  ```

### 2. 이상치 제거

- IQR(Interquartile Range) 방법을 사용하여 이상치를 제거합니다.

  ```python
  Q1 = data[['math', 'eng', 'science']].quantile(0.25)
  Q3 = data[['math', 'eng', 'science']].quantile(0.75)
  IQR = Q3 - Q1

  lower_bound = Q1 - 1.5 * IQR
  upper_bound = Q3 + 1.5 * IQR

  filtered_data = data[~((data[['math', 'eng', 'science']]  upper_bound)).any(axis=1)]
  ```

### 요구 사항

다음 Python 패키지가 필요합니다:

- pandas
- numpy
- scikit-learn
- matplotlib (옵션: 데이터 시각화를 위해 사용)

필요한 패키지를 설치하려면 다음 명령어를 실행하세요:

```bash
pip install pandas numpy scikit-learn matplotlib
```

### 실행 방법

1. `Exam-112.csv` 파일을 같은 디렉토리에 저장합니다.
2. 아래 코드를 실행합니다:
   ```bash
   python knn_prediction_service.py
   ```

---

## 코드 구조

### 주요 함수

#### `predict_science(math_score)`

- **설명:** 수학 점수를 입력받아 과학 점수를 예측합니다.
- **입력:** `math_score` (float)
- **출력:** 예측된 과학 점수 (float)

#### `predict_pass(math_score, eng_score, science_score)`

- **설명:** 수학, 영어, 과학 점수를 입력받아 합격 여부를 예측합니다.
- **입력:**
  - `math_score` (float): 수학 점수
  - `eng_score` (float): 영어 점수
  - `science_score` (float): 과학 점수
- **출력:** 합격 여부 (`Pass` 또는 `Fail`)

---

## 테스트 결과

### 테스트 케이스 1: 수학 점수가 75일 때 과학 점수 예측

```python
predicted_science_score = predict_science(75)
print(f"예측된 과학 점수: {predicted_science_score:.2f}")
```

**결과:**

```
예측된 과학 점수: 72.40
```

### 테스트 케이스 2: 수학=75, 영어=70, 과학=80일 때 합격 여부 예측

```python
predicted_status = predict_pass(75, 70, 80)
print(f"예측된 합격 여부: {predicted_status}")
```

**결과:**

```
예측된 합격 여부: Pass
```

---

## 참고 사항

1. **KNN 모델의 하이퍼파라미터 조정**

   - 기본적으로 `n_neighbors=5`로 설정되어 있습니다.
   - 필요에 따라 값을 변경하여 성능을 최적화할 수 있습니다.

2. **데이터 확장**

   - 현재 데이터는 제한적입니다. 더 많은 데이터를 추가하면 모델 성능이 향상될 수 있습니다.

3. **모델 평가**
   - 추가적인 평가 지표(MSE, 정확도 등)를 활용하여 모델의 성능을 분석할 수 있습니다.

---

## 라이선스

이 프로젝트는 자유롭게 수정 및 배포 가능합니다.
