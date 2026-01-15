# Semiconductor Yield Virtual Metrology

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-clean-brightgreen.svg)](https://github.com/psf/black)
[![Status](https://img.shields.io/badge/status-research-success.svg)](https://github.com)

> **반도체 센서 데이터를 활용한 공정 품질 지수 가상 계측(Virtual Metrology) 모델 개발**

SK-Planet 데이터 분석가 양성과정 프로젝트 (2023.01.03 ~ 2023.03.09)

---

## Overview

반도체 제조 공정에서 수집된 센서 데이터를 활용하여 **공정 품질 지수(Quality Index)를 예측**하는 머신러닝 모델을 개발하였습니다. 665개의 센서 변수와 7개의 공정 스텝 데이터를 기반으로 피처 엔지니어링, 다중공선성 처리, 하이퍼파라미터 최적화 등의 기법을 적용하였습니다.

### Key Features

- **Feature Engineering**: 665개 센서 변수 → 800+ 파생 변수 생성
- **Multicollinearity Treatment**: VIF > 10 변수에 대한 PCA 차원 축소
- **Feature Selection**: SelectKBest (Mutual Information) 기반 상위 250개 피처 선택
- **Bayesian Optimization**: Ridge, Random Forest 하이퍼파라미터 자동 튜닝
- **AutoML**: PyCaret을 활용한 다중 모델 비교 및 자동 튜닝

### Quick Stats

| Metric | Value |
|--------|-------|
| **Sample Size** | 611 observations (train) |
| **Sensor Variables** | 665 features |
| **Process Steps** | 7 steps (04, 06, 12, 13, 17, 18, 20) |
| **Sensor Types** | 95 sensor categories |
| **Target Variable** | Quality Index (mean ~1263) |
| **Generated Features** | 200+ engineered features |

---

## Sample Results

### Target Distribution Analysis
```
Mean: 1263.41
Variance: 67.16
Distribution: Approximately Normal
Outliers: y < 1240 (clipped during preprocessing)
```

### Process Duration Analysis
```
Total Process Time: 30.6 ~ 31.9 minutes
Critical Step Gap: Step 06 → Step 12 (longest duration)
Speed Classification: 1870 seconds threshold
  - Early (E): EQ7, EQ8 modules
  - Late (L): Other modules
```

### Model Performance (AutoML)
```
Best Model: Extra Trees Regressor
Evaluation Metric: RMSE
Cross-validation: 5-fold
```

---

## Quick Start

### Prerequisites

```bash
# Python 3.8 or higher
python --version

# Install dependencies
pip install -r requirements.txt
```

### Installation

```bash
# Clone the repository
git clone https://github.com/jinsoo96/T_academy_semiconductor_ai.git
cd T_academy_semiconductor_ai

# Install Python packages
pip install -r requirements.txt
```

### Run Analysis

**Full Pipeline:**
```bash
# Step 1: EDA and Data Preprocessing
jupyter notebook 01_Code/01_eda_and_preprocessing.ipynb

# Step 2: Modeling (AutoML + Bayesian Optimization)
jupyter notebook 01_Code/02_modeling.ipynb
```

**Note:** Common functions are centralized in `01_Code/utils.py` to avoid code duplication.

**Execution Time**: ~30-45 minutes for full pipeline

---

## Repository Structure

```
semiconductor_analysis/
│
├── 01_Code/
│   ├── utils.py                          # Common utility functions (shared module)
│   ├── 01_eda_and_preprocessing.ipynb    # EDA & Data Preprocessing
│   └── 02_modeling.ipynb                 # All modeling (AutoML + Bayesian Opt)
│
├── 02_Data/
│   ├── DATA_INFO.txt                     # Data documentation
│   └── raw/
│       ├── train_sensor.csv              # Training sensor data (~24.5MB)
│       ├── train_quality.csv             # Training quality labels (~25KB)
│       └── predict_sensor.csv            # Test sensor data (~10.5MB)
│
├── 03_Results/
│   ├── figures/                          # Generated plots
│   ├── tables/                           # Result tables
│   ├── preprocessed_data.pkl             # Preprocessed data (generated)
│   └── best_model.pkl                    # Trained model (generated)
│
├── 04_Documentation/
│   ├── final_presentation.pdf            # Final presentation slides
│   ├── Code_Structure.md                 # Detailed code guide
│   └── Analysis_Workflow.md              # Step-by-step workflow
│
├── archive/                              # Original development files
│
├── README.md                             # This file
├── requirements.txt                      # Python dependencies
├── LICENSE                               # MIT License
└── .gitignore                            # Git ignore rules
```

---

## Methodology

### 1. Data Collection & Preprocessing

- **Source**: SK HYNIX semiconductor manufacturing data
- **Period**: 2021 (October)
- **Sample**: 611 LOT observations with 665 sensor features

**Preprocessing Steps:**
1. Pivot transformation (Long → Wide format)
2. Feature generation from step_id + param_alias
3. Time feature extraction (process duration)
4. Missing value handling
5. Outlier treatment (IQR-based clipping)
6. Standardization (StandardScaler)

### 2. Feature Engineering

| Feature Type | Description | Count |
|--------------|-------------|-------|
| **Original Sensors** | Raw sensor measurements per step | 665 |
| **Duration Features** | Total and inter-step process time | 21 |
| **Statistical Features** | Sensor std/mean across steps | 190 |
| **Categorical Features** | Equipment category encoding | 8+ |
| **Binned Features** | Continuous variable discretization | 30+ |

### 3. Feature Selection Pipeline

```
665 Original Features
    ↓
+200 Generated Features (865 total)
    ↓
Variance Threshold (remove zero-variance)
    ↓
VIF Analysis (identify multicollinearity)
    ↓
PCA (reduce VIF>10 features)
    ↓
SelectKBest (top 250 by Mutual Information)
    ↓
Final Feature Set (~300 features)
```

### 4. Model Training

**Models Evaluated:**
1. Ridge Regression (Bayesian Optimized)
2. Random Forest Regressor (Bayesian Optimized)
3. Extra Trees Regressor
4. CatBoost Regressor
5. LightGBM Regressor
6. Gradient Boosting Regressor

**Training Strategy:**
- Train/Valid/Test split: 64% / 16% / 20%
- RandomOverSampler for class imbalance
- Log transformation on target (optional)
- 5-fold Cross-validation

---

## Key Findings & Insights

### 1. Target Variable Analysis

**Finding**: 타겟 변수 y는 평균 1263.41, 분산 67.16인 정규분포를 따름
- **Insight**: 정규분포 가정이 적절하므로 회귀 모델 적용에 유리
- **Action**: y < 1240 이하의 이상치를 clip하여 모델 안정성 확보

### 2. Equipment Effect (module_name_eq)

**Finding**: 장비(EQ1~EQ8)에 따라 품질 지수에 유의한 차이 존재
- **Insight**: 특정 장비(EQ7, EQ8)는 다른 장비 대비 낮은 품질 지수 경향
- **Action**: module_name_eq를 범주형 변수로 원-핫 인코딩하여 모델에 반영

### 3. Process Duration Impact

**Finding**: 전체 공정 소요시간이 1870초를 기준으로 두 그룹으로 명확히 구분
- **Early Group (E)**: ~30.6분, 주로 EQ7, EQ8 모듈
- **Late Group (L)**: ~31.9분, 대부분의 모듈
- **Insight**: 공정 속도와 품질 간의 상관관계 존재
- **Action**: tmdiff_speed 범주형 변수 생성하여 모델 피처로 활용

### 4. Critical Process Steps

**Finding**: Step 06 → Step 12 간 소요시간이 가장 길고 변동성이 높음
- **Insight**: 이 구간이 품질에 가장 큰 영향을 미치는 핵심 공정으로 추정
- **Action**: 개별 스텝 간 소요시간(gen_tmdiff_0612 등)을 파생 변수로 생성

### 5. Sensor Aggregation Value

**Finding**: 개별 스텝 센서값보다 전 스텝 집계값(std, mean)이 더 높은 예측력 보유
- **Insight**: 스텝 간 변동성이 품질 예측에 중요한 지표
- **Action**: 95개 센서별 표준편차(gen_{sensor}_std)와 평균(gen_{sensor}_mean) 변수 생성

### 6. Multicollinearity Issue

**Finding**: VIF > 10인 변수가 다수 존재 (다중공선성 문제)
- **Insight**: 센서 변수 간 높은 상관관계로 인한 모델 불안정성 위험
- **Action**: VIF > 10 변수들에 PCA 적용하여 2개 주성분으로 차원 축소

### 7. Feature Importance

**Finding**: Mutual Information 기반 상위 250개 피처 선택 시 성능 유지
- **Key Features**:
  - Process duration features (gen_tmdiff_*)
  - Aggregated sensor statistics (gen_*_std, gen_*_mean)
  - Equipment categorical features
- **Insight**: 파생 변수가 원본 센서 변수보다 예측력이 높음

### 8. Class Imbalance

**Finding**: 타겟 변수를 3개 구간(A: y<1242, B: 1242≤y≤1283, C: y>1283)으로 분류 시 불균형 발생
- **Distribution**: B 구간에 대부분의 샘플 집중
- **Action**: RandomOverSampler로 소수 클래스 오버샘플링

---

## Code Example

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.model_selection import train_test_split
from pycaret.regression import setup, compare_models, tune_model

# Load data
path = "./02_Data/raw/"
train_sensor = pd.read_csv(f'{path}train_sensor.csv')
train_quality = pd.read_csv(f'{path}train_quality.csv')

# Create dataset
train = make_dataset(train_sensor, train_quality)

# Feature engineering
train = gen_cate_feats(train)           # Equipment category
train = gen_duration_feats(train)       # Process duration
train = gen_stats_feats(train)          # Sensor statistics

# Feature selection
skb = SelectKBest(score_func=mutual_info_regression, k=250)
X_selected = skb.fit_transform(X, y)

# AutoML
reg = setup(data=train_df, target='y', normalize=True)
best = compare_models(sort='RMSE')
best_tuned = tune_model(best)
```

---

## Requirements

### Python Packages

```txt
pandas>=1.3.0
numpy>=1.21.0
scipy>=1.7.0
scikit-learn>=1.0.0
catboost>=1.0.0
lightgbm>=3.3.0
xgboost>=1.5.0
pycaret>=2.3.0
bayesian-optimization>=1.2.0
imbalanced-learn>=0.9.0
statsmodels>=0.13.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

### System Requirements

- **Python**: 3.8 or higher
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 500MB for code and data
- **OS**: Windows, macOS, or Linux

---

## Team Members

| Name | Role |
|------|------|
| 임유경 | Data Analysis & Modeling |
| 김진수 | Feature Engineering & Optimization |
| 이호진 | EDA & Visualization |
| 안승아 | Preprocessing & Documentation |

---

## Documentation

| Document | Description |
|----------|-------------|
| [README.md](README.md) | Project overview (this file) |
| [Code_Structure.md](04_Documentation/Code_Structure.md) | Detailed code documentation |
| [Analysis_Workflow.md](04_Documentation/Analysis_Workflow.md) | Step-by-step workflow |
| [final_presentation.pdf](04_Documentation/final_presentation.pdf) | Final presentation slides |

---

## References

- [SK HYNIX Project Notion](https://www.notion.so/SCV-SK-with-ASAC-2c4a1af25a594d1895b60ada4e7144ad)
- [PyCaret Documentation](https://pycaret.org/)
- [Bayesian Optimization](https://github.com/fmfn/BayesianOptimization)
- [Scikit-learn Feature Selection](https://scikit-learn.org/stable/modules/feature_selection.html)

---

## License

This project is for educational purposes as part of SK-Planet Data Analyst Training Program.

---

## Contact

- **Team**: SCV (SK with ASAC)
- **Notion**: https://www.notion.so/SCV-SK-with-ASAC-2c4a1af25a594d1895b60ada4e7144ad
