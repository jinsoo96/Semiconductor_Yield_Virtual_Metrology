# Analysis Workflow

이 문서는 반도체 수율 예측 프로젝트의 전체 분석 워크플로우를 단계별로 설명합니다.

---

## Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Analysis Pipeline Overview                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   [Raw Data]                                                                 │
│       │                                                                      │
│       ▼                                                                      │
│   [1. Data Integration] ─────────────────────────────────────────────────   │
│       │  - Pivot transformation (Long → Wide)                                │
│       │  - Feature column generation (665 sensors)                           │
│       ▼                                                                      │
│   [2. Exploratory Data Analysis] ───────────────────────────────────────    │
│       │  - Target distribution analysis                                      │
│       │  - Sensor-target correlation                                         │
│       │  - Equipment effect analysis                                         │
│       ▼                                                                      │
│   [3. Feature Engineering] ─────────────────────────────────────────────    │
│       │  - Duration features (21)                                            │
│       │  - Statistical features (190)                                        │
│       │  - Categorical features                                              │
│       ▼                                                                      │
│   [4. Data Preprocessing] ──────────────────────────────────────────────    │
│       │  - Outlier treatment                                                 │
│       │  - Multicollinearity (VIF + PCA)                                     │
│       │  - Feature selection                                                 │
│       ▼                                                                      │
│   [5. Model Training] ──────────────────────────────────────────────────    │
│       │  - AutoML (PyCaret)                                                  │
│       │  - Bayesian Optimization                                             │
│       ▼                                                                      │
│   [6. Evaluation & Prediction] ─────────────────────────────────────────    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Step 1: Data Integration

### Input Files
```
02_Data/raw/
├── train_sensor.csv    (Long format, ~24.5MB)
├── train_quality.csv   (Target values, ~25KB)
└── predict_sensor.csv  (Test data, ~10.5MB)
```

### Process

1. **Load Data**
   ```python
   train_sensor = pd.read_csv('02_Data/raw/train_sensor.csv')
   train_quality = pd.read_csv('02_Data/raw/train_quality.csv')
   ```

2. **Create Step-Param Column**
   ```python
   df['step_param'] = df['step_id'].astype(str).str.zfill(2) + '_' + df['param_alias']
   # Example: "17_EPD_para4"
   ```

3. **Pivot Transformation**
   ```python
   df_pivot = df.pivot_table(
       index=['module_name', 'key_val'],
       columns='step_param',
       values='mean_val',
       aggfunc='sum'
   )
   ```

4. **Extract Time Features**
   ```python
   df_time = df.pivot_table(
       index='key_val',
       columns='end_time_tmp',
       values='end_time',
       aggfunc=lambda x: max(x.unique())
   )
   ```

5. **Merge with Target**
   ```python
   df_complete = pd.concat([df_pivot, df_time, df_quality], axis=1)
   ```

### Output
- **Shape**: (611, 680+)
- **Columns**: module_name, key_val, y, 665 sensor columns, 8 time columns

---

## Step 2: Exploratory Data Analysis

### 2.1 Target Variable Analysis

```python
# Distribution check
QQ_plot(df['y'], 'y')
```

**Findings:**
- Mean: 1263.41
- Std: 8.20
- Distribution: Approximately Normal
- Outliers: y < 1240 존재

### 2.2 Sensor-Target Correlation

```python
# Correlation analysis
details = describe_(train, 'y')
details.sort_values(by='corr y', ascending=False)
```

**Top Correlated Sensors:**
| Rank | Sensor | Correlation |
|------|--------|-------------|
| 1 | gen_tmdiff | -0.XX |
| 2 | 06_efem_para2 | 0.XX |
| 3 | 17_hv_para3 | 0.XX |

### 2.3 Equipment Analysis

```python
# Equipment category effect
sns.boxplot(x='module_name_eq', y='y', data=df)
```

**Findings:**
- EQ7, EQ8: 상대적으로 낮은 품질 지수
- EQ1~EQ6: 유사한 분포

### 2.4 Process Duration Analysis

```python
# Duration scatter plot
sns.scatterplot(x='gen_tmdiff', y='y', hue='tmdiff_speed', data=df)
```

**Findings:**
- 1870초 기준으로 두 그룹 명확히 구분
- Early Group: EQ7, EQ8 (30.6분)
- Late Group: 기타 장비 (31.9분)

---

## Step 3: Feature Engineering

### 3.1 Duration Features (21개)

```python
# 전체 공정 소요시간
df['gen_tmdiff'] = (df['20_end_time'] - df['04_end_time']).dt.total_seconds()

# 개별 스텝 간 소요시간
for gap in lst_stepsgap:  # ['0406', '0612', ...]
    df[f'gen_tmdiff_{gap}'] = (df[f'{gap[2:]}_end_time'] - df[f'{gap[:2]}_end_time']).dt.total_seconds()
```

### 3.2 Statistical Features (190개)

```python
# 센서별 표준편차 (95개)
for sensor in sensors_nm:
    cols = [f'{step}_{sensor}' for step in lst_steps]
    df[f'gen_{sensor}_std'] = df[cols].std(axis=1)

# 센서별 평균 (95개)
for sensor in sensors_nm:
    cols = [f'{step}_{sensor}' for step in lst_steps]
    df[f'gen_{sensor}_mean'] = df[cols].mean(axis=1)
```

### 3.3 Categorical Features

```python
# 장비 카테고리
df['module_name_eq'] = df['module_name'].apply(lambda x: x.split('_')[0])

# 공정 속도 카테고리
df['tmdiff_speed'] = np.where(df['gen_tmdiff'] < 1870, 'E', 'L')
```

### 3.4 Binned Features

```python
# 연속형 변수 구간화
def cut_data(df, col):
    bins = [1400, 1450, 1520, 1570, 1660, 1750]
    labels = ['a', 'b', 'c', 'd', 'e']
    return pd.cut(df[col], bins, labels=labels)
```

---

## Step 4: Data Preprocessing

### 4.1 Outlier Treatment

```python
# Target clipping
df['y'] = df['y'].clip(1240, 1500)

# Feature clipping (IQR method)
for col in numerical_columns:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    df[col] = df[col].clip(q1 - 1.5*iqr, q3 + 1.5*iqr)
```

### 4.2 Variance Threshold

```python
from sklearn.feature_selection import VarianceThreshold

thresholder = VarianceThreshold(threshold=0)
thresholder.fit(df[numerical_columns])
cols_to_drop = df.columns[~thresholder.get_support()]
```

### 4.3 Multicollinearity Treatment

```python
# VIF 계산
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
vif['feature'] = df.columns

# VIF > 10 변수에 PCA 적용
high_vif_cols = vif[vif['VIF'] > 10]['feature'].tolist()
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df[high_vif_cols])
```

### 4.4 One-Hot Encoding

```python
# 범주형 변수 인코딩
df = pd.get_dummies(df, columns=['module_name_eq', 'tmdiff_speed'])
```

### 4.5 Feature Selection

```python
from sklearn.feature_selection import SelectKBest, mutual_info_regression

skb = SelectKBest(score_func=mutual_info_regression, k=250)
X_selected = skb.fit_transform(X, y)
selected_features = X.columns[skb.get_support()]
```

---

## Step 5: Model Training

### 5.1 Data Split

```python
# 80-20 split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)

# Validation split from train
X_train_, X_valid, y_train_, y_valid = train_test_split(
    X_train, y_train, test_size=0.2, random_state=1
)
```

### 5.2 Oversampling

```python
from imblearn.over_sampling import RandomOverSampler

# Target binning for oversampling
df['y_cate'] = pd.cut(df['y'], bins=[0, 1242, 1283, 1500], labels=['A', 'B', 'C'])

ros = RandomOverSampler(random_state=1)
X_resampled, y_resampled = ros.fit_resample(X, y_cate)
```

### 5.3 AutoML (PyCaret)

```python
from pycaret.regression import *

# Setup
reg = setup(
    data=train_df,
    target='y',
    normalize=True,
    train_size=0.8,
    fold=5,
    session_id=123
)

# Model comparison
best = compare_models(sort='RMSE')

# Tuning
best_tuned = tune_model(best)

# Evaluation
evaluate_model(best_tuned)
```

### 5.4 Bayesian Optimization

```python
from bayes_opt import BayesianOptimization

# Ridge optimization
def ridge_cv(alpha):
    model = Ridge(alpha=alpha)
    return cross_val_score(model, X, y, cv=5).mean()

optimizer = BayesianOptimization(
    f=ridge_cv,
    pbounds={'alpha': (0, 10)}
)
optimizer.maximize(init_points=5, n_iter=20)
```

---

## Step 6: Evaluation & Prediction

### Metrics

```python
from sklearn.metrics import mean_squared_error, r2_score

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Evaluate
print(f"RMSE: {rmse(y_test, y_pred):.4f}")
print(f"R2: {r2_score(y_test, y_pred):.4f}")
```

### Prediction

```python
# Load test data
predict = make_dataset(predict_sensor)

# Apply same preprocessing
predict = gen_cate_feats(predict)
predict = gen_duration_feats(predict, lst_stepsgap)
predict = gen_stats_feats(predict, sensors_nm, lst_steps)

# Predict
predictions = best_model.predict(predict[selected_features])
```

---

## Execution Order

```
[Recommended Order]

1. Run 01_eda_and_preprocessing.ipynb
   ├── Cell 1-20: Data loading & EDA
   ├── Cell 21-40: Feature engineering
   ├── Cell 41-60: Preprocessing
   └── Cell 61+: Basic modeling

2. Run 02_automl_pycaret.ipynb
   ├── Same preprocessing as Step 1
   └── AutoML model comparison

3. Run 03_bayesian_optimization.ipynb
   ├── Same preprocessing as Step 1
   └── Hyperparameter tuning
```

---

## Output Files

| File | Location | Description |
|------|----------|-------------|
| Predictions | 03_Results/tables/ | Final predictions CSV |
| Figures | 03_Results/figures/ | EDA visualizations |
| Model | 03_Results/ | Trained model pickle |
