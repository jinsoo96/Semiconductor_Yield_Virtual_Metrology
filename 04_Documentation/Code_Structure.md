# Code Structure Documentation

이 문서는 반도체 수율 예측 프로젝트의 코드 구조를 상세하게 설명합니다.

---

## Directory Overview

```
01_Code/
├── 01_eda_and_preprocessing.ipynb    # EDA 및 전처리
├── 02_automl_pycaret.ipynb           # PyCaret AutoML
└── 03_bayesian_optimization.ipynb    # 베이지안 최적화
```

---

## 01_eda_and_preprocessing.ipynb

### 목적
데이터 탐색, 전처리, 피처 엔지니어링, 기본 모델 학습을 수행하는 메인 노트북

### 섹션 구성

#### 1. 분석환경 구축
```python
# 주요 라이브러리
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_regression
```

#### 2. 데이터 읽기 및 통합
```python
def make_dataset(X, y=None):
    """
    센서 데이터와 품질 데이터를 통합하는 함수

    Parameters:
    - X: train_sensor DataFrame (Long format)
    - y: train_quality DataFrame (optional)

    Returns:
    - df_complete: 통합된 DataFrame (Wide format, 665 sensor columns)

    Process:
    1. step_id + param_alias 결합하여 step_param 생성
    2. pivot_table로 Long → Wide 변환
    3. 시간 데이터 추출
    4. 품질 데이터 병합
    """
```

#### 3. 탐색적 데이터 분석 (EDA)

**3.1 타겟 변수 분포**
```python
def QQ_plot(data, measure):
    """정규분포 적합도 확인을 위한 QQ Plot"""
```

**3.2 센서-타겟 상관관계**
```python
def regplots(cols, data):
    """센서별 스텝 단위 회귀 플롯 시각화"""
```

**3.3 범주형 변수 분석**
```python
def gen_cate_feats(df):
    """장비 상위 카테고리 변수 생성"""
    df['module_name_eq'] = df['module_name'].apply(lambda x: x.split('_')[0])
```

**3.4 공정 소요시간 분석**
```python
def gen_duration_feats(df, lst_stepsgap):
    """
    전체 및 개별 스텝 간 공정 소요시간 변수 생성
    - gen_tmdiff: 전체 소요시간 (20_end_time - 04_end_time)
    - gen_tmdiff_{step1}{step2}: 스텝 간 소요시간
    """
```

**3.5 센서 통계량 분석**
```python
def gen_stats_feats(df, sensors_nm, lst_steps):
    """센서별 전 스텝 표준편차 변수 생성"""

def gen_avg_feats(df, sensors_nm, lst_steps):
    """센서별 전 스텝 평균 변수 생성"""
```

#### 4. 데이터 전처리

**4.1 이상치 처리**
```python
def cliping(df, columns):
    """IQR 기반 이상치 클리핑"""
    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        df[col] = df[col].clip(q1 - 1.5*iqr, q3 + 1.5*iqr)
```

**4.2 분산 기반 변수 제거**
```python
thresholder = VarianceThreshold(threshold=0)
# 분산이 0인 변수 제거
```

**4.3 다중공선성 처리**
```python
# VIF 계산
vif["VIF Factor"] = [variance_inflation_factor(df[cols].values, i)
                     for i in range(len(cols))]

# VIF > 10 변수에 PCA 적용
pca = PCA(n_components=2)
```

**4.4 원-핫 인코딩**
```python
def prep_cate_feats(df_tr, df_te, feat_nm):
    """범주형 변수 원-핫 인코딩"""
    df_merge = pd.get_dummies(df_merge, columns=[feat_nm])
```

**4.5 피처 선택**
```python
skb = SelectKBest(score_func=mutual_info_regression, k=250)
# Mutual Information 기반 상위 250개 피처 선택
```

#### 5. 모델링

**5.1 데이터 분할**
```python
x_train, x_test, y_train, y_test = train_test_split(
    x_train_raw, y_train_raw, test_size=0.2, random_state=1
)
```

**5.2 오버샘플링**
```python
os_df, os_target = RandomOverSampler(random_state=1).fit_resample(
    os_df.drop('y_cate', axis=1), os_df['y_cate']
)
```

**5.3 PyCaret AutoML**
```python
reg = setup(data=os_df, target='y', train_size=0.8, fold=5)
best = compare_models(sort='RMSE')
best_tune = tune_model(best)
```

---

## 02_automl_pycaret.ipynb

### 목적
PyCaret을 활용한 자동화된 모델 비교 및 튜닝

### 주요 기능

```python
# 1. 환경 설정
reg = setup(
    data=os_df,
    target='log_y',        # 로그 변환된 타겟
    normalize=True,        # 정규화 적용
    train_size=0.6,
    session_id=123
)

# 2. 모델 비교
best = compare_models(sort='RMSE')
# 비교 모델: CatBoost, ExtraTrees, RandomForest, GBR, LightGBM

# 3. 개별 모델 생성
cat = create_model('catboost', cross_validation=True)
et = create_model('et', cross_validation=True)
rf = create_model('rf', cross_validation=True)
gbr = create_model('gbr', cross_validation=True)
lightgbm = create_model('lightgbm', cross_validation=True)

# 4. 모델 튜닝
best_tune = tune_model(best)

# 5. 시각화
evaluate_model(best_tune)
```

---

## 03_bayesian_optimization.ipynb

### 목적
베이지안 최적화를 통한 하이퍼파라미터 튜닝

### 주요 기능

**Ridge Regression 최적화**
```python
def evaluate_ridge(alpha):
    model = Ridge(alpha=alpha)
    model.fit(x_train_, y_train_)
    return model.score(x_train_, y_train_)

bounds = {'alpha': (0, 10)}
optimizer = BayesianOptimization(f=evaluate_ridge, pbounds=bounds)
optimizer.maximize(init_points=5, n_iter=20)
```

**Random Forest 최적화**
```python
def rf_cv(n_estimators, min_samples_split, max_features, max_depth):
    val = cross_val_score(
        RandomForestRegressor(
            n_estimators=int(n_estimators),
            min_samples_split=int(min_samples_split),
            max_features=min(max_features, 0.999),
            max_depth=int(max_depth),
            random_state=42
        ),
        x_train, y_train,
        scoring='neg_mean_squared_error',
        cv=10
    ).mean()
    return val

bounds = {
    'n_estimators': (100, 500),
    'min_samples_split': (2, 10),
    'max_features': (0.1, 0.999),
    'max_depth': (3, 20)
}
```

---

## 핵심 함수 요약

| 함수명 | 파일 | 설명 |
|--------|------|------|
| `make_dataset()` | 01 | 데이터 통합 및 피벗 변환 |
| `describe_()` | 01 | 데이터 기술통계 분석 |
| `QQ_plot()` | 01 | 정규분포 적합도 시각화 |
| `gen_cate_feats()` | 01 | 범주형 변수 생성 |
| `gen_duration_feats()` | 01 | 공정 소요시간 변수 생성 |
| `gen_stats_feats()` | 01 | 센서 표준편차 변수 생성 |
| `gen_avg_feats()` | 01 | 센서 평균 변수 생성 |
| `cliping()` | 01 | IQR 기반 이상치 처리 |
| `prep_cate_feats()` | 01 | 원-핫 인코딩 |
| `rmse()` | 01,02,03 | RMSE 계산 |

---

## 변수 명명 규칙

| Prefix | 설명 | 예시 |
|--------|------|------|
| `gen_` | 생성된 파생 변수 | `gen_tmdiff`, `gen_hv_para3_std` |
| `cut_` | 구간화된 변수 | `cut_06_efem_para2` |
| `pca_` | PCA 변환 변수 | `pca_04_gas_para36` |
| `{step}_` | 스텝별 센서 변수 | `04_efem_para2`, `20_hv_para3` |
