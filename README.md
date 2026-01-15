# Semiconductor Yield Virtual Metrology

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-clean-brightgreen.svg)](https://github.com/psf/black)
[![Status](https://img.shields.io/badge/status-research-success.svg)](https://github.com)

> **Machine Learning-based Quality Index Prediction Model for Semiconductor Manufacturing Process**

SK-Planet Data Analyst Training Program Project (2023.01.03 ~ 2023.03.09)

---

## Overview

This project develops a **Virtual Metrology (VM) model** that predicts the quality index of semiconductor manufacturing processes using sensor data. The model leverages 665 sensor variables across 7 process steps, applying feature engineering, multicollinearity treatment, and hyperparameter optimization techniques.

### Key Features

- **Feature Engineering**: 665 sensor variables → 800+ derived features
- **Multicollinearity Treatment**: PCA dimensionality reduction for VIF > 10 variables
- **Feature Selection**: SelectKBest (Mutual Information) based top 250 features
- **Bayesian Optimization**: Automated hyperparameter tuning for Ridge, Random Forest
- **AutoML**: Multi-model comparison and auto-tuning with PyCaret

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
git clone git@github.com:jinsoo96/Semiconductor_Yield_Virtual_Metrology.git
cd Semiconductor_Yield_Virtual_Metrology

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
Semiconductor_Yield_Virtual_Metrology/
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
- **Period**: October 2021
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

**Finding**: Target variable y follows approximately normal distribution (mean: 1263.41, variance: 67.16)
- **Insight**: Normal distribution assumption is valid, favorable for regression models
- **Action**: Clip outliers below y < 1240 for model stability

### 2. Equipment Effect (module_name_eq)

**Finding**: Significant quality index variation exists across equipment (EQ1~EQ8)
- **Insight**: EQ7, EQ8 show relatively lower quality indices compared to others
- **Action**: One-hot encode module_name_eq as categorical feature

### 3. Process Duration Impact

**Finding**: Total process duration clearly separates into two groups at 1870 seconds threshold
- **Early Group (E)**: ~30.6 min, mainly EQ7, EQ8 modules
- **Late Group (L)**: ~31.9 min, most modules
- **Insight**: Correlation exists between process speed and quality
- **Action**: Create tmdiff_speed categorical variable as model feature

### 4. Critical Process Steps

**Finding**: Step 06 → Step 12 shows longest duration and highest variability
- **Insight**: This interval is estimated to have the greatest impact on quality
- **Action**: Generate inter-step duration features (gen_tmdiff_0612, etc.)

### 5. Sensor Aggregation Value

**Finding**: Aggregated sensor statistics (std, mean) across steps show higher predictive power than individual step values
- **Insight**: Step-wise variability is a key indicator for quality prediction
- **Action**: Generate 95 sensor std (gen_{sensor}_std) and mean (gen_{sensor}_mean) features

### 6. Multicollinearity Issue

**Finding**: Multiple variables with VIF > 10 exist (multicollinearity problem)
- **Insight**: High correlation among sensor variables poses model instability risk
- **Action**: Apply PCA to VIF > 10 variables, reduce to 2 principal components

### 7. Feature Importance

**Finding**: Model performance maintained with top 250 features selected by Mutual Information
- **Key Features**:
  - Process duration features (gen_tmdiff_*)
  - Aggregated sensor statistics (gen_*_std, gen_*_mean)
  - Equipment categorical features
- **Insight**: Derived features show higher predictive power than original sensor variables

### 8. Class Imbalance

**Finding**: Target variable shows imbalance when binned into 3 categories (A: y<1242, B: 1242≤y≤1283, C: y>1283)
- **Distribution**: Most samples concentrated in B category
- **Action**: Apply RandomOverSampler for minority class oversampling

---

## Code Example

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.model_selection import train_test_split

# Import utility functions
from utils import (
    load_data, make_dataset,
    gen_cate_feats, gen_duration_feats, gen_stats_feats,
    LST_STEPS, LST_STEPSGAP
)

# Load data
path = "./02_Data/raw/"
train_sensor, train_quality, predict_sensor = load_data(path)

# Create dataset
train = make_dataset(train_sensor, train_quality)

# Feature engineering
train = gen_cate_feats(train)           # Equipment category
train = gen_duration_feats(train, LST_STEPSGAP)  # Process duration
train = gen_stats_feats(train, sensors_nm, LST_STEPS)  # Sensor statistics

# Feature selection
skb = SelectKBest(score_func=mutual_info_regression, k=250)
X_selected = skb.fit_transform(X, y)

# Train model
from pycaret.regression import setup, compare_models, tune_model
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
| Yukyung Lim | Data Analysis & Modeling |
| Jin Soo Kim | Feature Engineering & Optimization |
| Hojin Lee | EDA & Visualization |
| Seungah Ahn | Preprocessing & Documentation |

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

- [PyCaret Documentation](https://pycaret.org/)
- [Bayesian Optimization](https://github.com/fmfn/BayesianOptimization)
- [Scikit-learn Feature Selection](https://scikit-learn.org/stable/modules/feature_selection.html)
- [Virtual Metrology in Semiconductor Manufacturing](https://ieeexplore.ieee.org/)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License
Copyright (c) 2023 Jin Soo Kim
```

---

## Contact

- **Author**: Jin Soo Kim
- **GitHub**: [@jinsoo96](https://github.com/jinsoo96)
