"""
Semiconductor Yield Prediction - Utility Functions
===================================================

This module contains common functions for data preprocessing,
feature engineering, and visualization used across all notebooks.

Author: Jin Soo Kim
Date: 2023
"""

import warnings
warnings.filterwarnings('ignore')

# Data processing
import numpy as np
import pandas as pd
import scipy as sp
from datetime import datetime, date, timedelta

# Preprocessing
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_regression, VarianceThreshold
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew, norm, probplot

# Imbalanced learning
from imblearn.over_sampling import RandomOverSampler

# Itertools
from itertools import product


# =============================================================================
# DATA LOADING AND INTEGRATION
# =============================================================================

def load_data(path="./02_Data/raw/"):
    """
    Load training and prediction data.

    Parameters:
    -----------
    path : str
        Path to data directory

    Returns:
    --------
    tuple : (train_sensor, train_quality, predict_sensor)
    """
    train_sensor = pd.read_csv(f'{path}train_sensor.csv')
    train_quality = pd.read_csv(f'{path}train_quality.csv')
    predict_sensor = pd.read_csv(f'{path}predict_sensor.csv')

    return train_sensor, train_quality, predict_sensor


def make_dataset(X, y=None):
    """
    Transform sensor data from long to wide format and merge with quality data.

    Parameters:
    -----------
    X : DataFrame
        Sensor data in long format
    y : DataFrame, optional
        Quality data with target variable

    Returns:
    --------
    DataFrame : Integrated dataset in wide format with 665 sensor columns
    """
    # -----------------------------------
    # train_sensor (X argument)
    # -----------------------------------
    df_X = X.copy()
    # Zero-pad step_id if single digit
    df_X['step_id'] = df_X['step_id'].apply(lambda x: str(x).zfill(2))
    # Create step_param column combining step_id and param_alias
    df_X['step_param'] = df_X[['step_id', 'param_alias']].apply(lambda x: '_'.join(x), axis=1)
    print('Feature count: {}'.format(len(set(df_X['step_param']))))

    df_X_tmp = df_X.pivot_table(
        index=['module_name', 'key_val'],
        columns='step_param',
        values='mean_val',
        aggfunc='sum'
    )
    df_X_tmp = df_X_tmp.reset_index(level=[0, 1])
    df_X_tmp.set_index('key_val', inplace=True)

    # -----------------------------------
    # Time data
    # -----------------------------------
    df_X['end_time_tmp'] = df_X.apply(lambda x: x['step_id'] + '_end_time', axis=1)
    df_X['end_time'] = pd.to_datetime(df_X['end_time'])

    df_time_tmp = df_X.pivot_table(
        index=['key_val'],
        columns='end_time_tmp',
        values='end_time',
        aggfunc=lambda x: max(x.unique())
    )
    df_time_tmp = df_time_tmp.reset_index()
    df_time_tmp.set_index('key_val', inplace=True)

    # -----------------------------------
    # train_quality (y argument)
    # -----------------------------------
    if y is None:  # Prediction data
        col_target = []
        col_idx = ['module_name', 'key_val']
        df_complete = pd.concat([df_X_tmp, df_time_tmp], axis=1).reset_index()
        df_complete = df_complete.rename(columns={'index': 'key_val'})
    else:  # Training data
        df_y = y.copy()
        df_y.set_index('key_val', inplace=True)

        col_target = ['y']
        col_idx = ['module_name', 'key_val', 'end_dt_tm']

        df_complete = pd.concat([df_X_tmp, df_time_tmp, df_y], axis=1).reset_index()
        df_complete = df_complete.rename(columns={'index': 'key_val'})
        df_complete.rename(columns={'msure_val': 'y'}, inplace=True)

    # Sort columns
    col_feats = df_X['step_param'].unique().tolist()
    col_feats.sort()
    col_time = [s for s in df_complete.columns.tolist() if "_end_time" in s]
    col_all = col_idx + col_target + col_feats + col_time
    df_complete = df_complete[col_all]

    # Convert to lowercase
    df_complete.columns = df_complete.columns.str.lower()

    return df_complete


# =============================================================================
# EXPLORATORY DATA ANALYSIS
# =============================================================================

def describe_(df, pred=None):
    """
    Generate descriptive statistics for DataFrame.

    Parameters:
    -----------
    df : DataFrame
        Input data
    pred : str, optional
        Target variable name for correlation analysis

    Returns:
    --------
    DataFrame : Descriptive statistics
    """
    obs = df.shape[0]
    types = df.dtypes
    counts = df.apply(lambda x: x.count())
    uniques = df.apply(lambda x: x.unique())
    distincts = df.apply(lambda x: x.unique().shape[0])
    nulls = df.apply(lambda x: x.isnull().sum())
    nulls_ratio = (df.isnull().sum() / obs) * 100
    skewness = df.skew()
    kurtosis = df.kurt()

    print('Data shape:', df.shape)

    if pred is None:
        cols = ['types', 'counts', 'distincts', 'nulls', 'nulls_ratio', 'uniques', 'skewness', 'kurtosis']
        values_ = pd.concat([types, counts, distincts, nulls, nulls_ratio, uniques, skewness, kurtosis], axis=1)
    else:
        corr = df.corr()[pred]
        values_ = pd.concat([types, counts, distincts, nulls, nulls_ratio, uniques, skewness, kurtosis, corr], axis=1, sort=False)
        corr_col = 'corr ' + pred
        cols = ['types', 'counts', 'distincts', 'nulls', 'nulls_ratio', 'uniques', 'skewness', 'kurtosis', corr_col]

    values_.columns = cols
    print('___________________________\nData types:\n', values_.types.value_counts())
    print('___________________________')

    return values_


def QQ_plot(data, measure):
    """
    Create QQ plot and distribution histogram.

    Parameters:
    -----------
    data : Series
        Data to plot
    measure : str
        Variable name for labels
    """
    fig = plt.figure(figsize=(10, 4))

    # Histogram with KDE
    fig1 = fig.add_subplot(121)
    (mu, sigma) = norm.fit(data)
    sns.distplot(data, kde=True, fit=norm)
    fig1.legend(['KDE', f'N({round(mu, 2)},{round(sigma**2, 2)})'], loc='upper right')
    fig1.set_title(f'{measure} Distribution', loc='center')
    fig1.set_xlabel(f'{measure}')

    # QQ Plot
    fig2 = fig.add_subplot(122)
    res = probplot(data, plot=fig2)
    fig2.set_title(f'{measure} Probability Plot', loc='center')

    plt.tight_layout()
    plt.show()


def regplots(cols, data):
    """
    Create regression plots for multiple columns.

    Parameters:
    -----------
    cols : list
        Column names to plot
    data : DataFrame
        Data source
    """
    fig, axes = plt.subplots(nrows=2, ncols=4, sharey=True, figsize=(15, 8))
    fig.subplots_adjust(hspace=.4, wspace=.1)

    for i, ax in zip(range(len(cols)), axes.flat):
        sns.regplot(x=cols[i], y='y', data=data, ax=ax)
        ax.set_title(f'{cols[i].upper()}')
        ax.set_xlabel('')
        ax.set_ylabel('')

    axes.flat[-1].set_visible(False)
    plt.tight_layout()
    plt.show()


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def gen_cate_feats(df):
    """
    Generate equipment category feature.

    Parameters:
    -----------
    df : DataFrame
        Input data

    Returns:
    --------
    DataFrame : Data with module_name_eq column
    """
    df['module_name_eq'] = df['module_name'].apply(lambda x: x.split('_')[0])
    return df


def gen_duration_feats(df, lst_stepsgap):
    """
    Generate process duration features.

    Parameters:
    -----------
    df : DataFrame
        Input data
    lst_stepsgap : list
        List of step gap codes (e.g., ['0406', '0612', ...])

    Returns:
    --------
    DataFrame : Data with duration features
    """
    # Total process duration (seconds)
    df['gen_tmdiff'] = (df['20_end_time'] - df['04_end_time']).dt.total_seconds()

    # Individual step durations
    for stepgap in lst_stepsgap:
        df[f'gen_tmdiff_{stepgap}'] = (df[f'{stepgap[2:]}_end_time'] - df[f'{stepgap[:2]}_end_time']).dt.total_seconds()

    return df


def gen_stats_feats(df, sensors_nm, lst_steps):
    """
    Generate aggregated standard deviation features for each sensor.

    Parameters:
    -----------
    df : DataFrame
        Input data
    sensors_nm : list
        List of sensor names
    lst_steps : list
        List of step IDs

    Returns:
    --------
    DataFrame : Data with std features
    """
    for sensor_nm in sensors_nm:
        tmp_lst = list(map(lambda x: f'{x}_{sensor_nm}', lst_steps))
        df[f'gen_{sensor_nm}_std'] = df[tmp_lst].std(axis=1)
    return df


def gen_avg_feats(df, sensors_nm, lst_steps):
    """
    Generate aggregated mean features for each sensor.

    Parameters:
    -----------
    df : DataFrame
        Input data
    sensors_nm : list
        List of sensor names
    lst_steps : list
        List of step IDs

    Returns:
    --------
    DataFrame : Data with mean features
    """
    for sensor_nm in sensors_nm:
        tmp_lst = list(map(lambda x: f'{x}_{sensor_nm}', lst_steps))
        df[f'gen_{sensor_nm}_mean'] = df[tmp_lst].mean(axis=1)
    return df


def gen_avg_param(df, param_set):
    """Generate average features by parameter type."""
    for i in param_set:
        param = [x for x in df.columns if i in x and 'para' in x]
        df[f'gen_avg_{i}'] = df[param].mean(axis=1)
    return df


def gen_std_param(df, param_set):
    """Generate std features by parameter type."""
    for i in param_set:
        param = [x for x in df.columns if i in x and 'para' in x]
        df[f'gen_std_{i}'] = df[param].std(axis=1)
    return df


def gen_avg_step(df, lst_steps):
    """Generate average features by step."""
    for i in lst_steps:
        param_steps = [x for x in df.columns if i == x.split('_')[0] and 'para' in x]
        df[f'gen_avg_{i}'] = df[param_steps].mean(axis=1)
    return df


# =============================================================================
# DATA PREPROCESSING
# =============================================================================

def log_skew(df):
    """
    Apply power transformation to reduce skewness.

    Parameters:
    -----------
    df : DataFrame
        Input data

    Returns:
    --------
    DataFrame : Transformed data
    """
    numeric = [k for k in df.columns if df[k].dtype == 'float']
    numeric_lst = pd.DataFrame(abs(df[numeric].skew())).sort_values(by=0, ascending=False)[:100].index.tolist()

    for col in numeric_lst:
        if df[col].skew() < 0:
            df[col] = df[col] ** 2
        else:
            df[col] = df[col] ** 0.5
    return df


def cut_data(df, col):
    """Bin continuous variable into categories."""
    bins = [1400, 1450, 1520, 1570, 1660, 1750]
    labels = ['a', 'b', 'c', 'd', 'e']
    return pd.cut(df[col], bins, right=False, labels=labels)


def cut_epd_data(df, col):
    """Bin EPD variable into categories."""
    bins = [0, 50, 100]
    labels = ['a', 'b']
    return pd.cut(df[col], bins, right=False, labels=labels)


def clipping(df, columns):
    """
    Apply IQR-based clipping to handle outliers.

    Parameters:
    -----------
    df : DataFrame
        Input data
    columns : list
        Columns to clip

    Returns:
    --------
    DataFrame : Clipped data
    """
    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        low_lim = q1 - 1.5 * iqr
        high_lim = q3 + 1.5 * iqr
        df[col] = df[col].clip(low_lim, high_lim)
    return df


def prep_cate_feats(df_tr, df_te, feat_nm):
    """
    Apply one-hot encoding to categorical variable.

    Parameters:
    -----------
    df_tr : DataFrame
        Training data
    df_te : DataFrame
        Test data
    feat_nm : str
        Feature name to encode

    Returns:
    --------
    tuple : (encoded train, encoded test)
    """
    df_merge = pd.concat([df_tr, df_te])
    df_merge = pd.get_dummies(df_merge, columns=[feat_nm])
    df_tr = df_merge.iloc[:df_tr.shape[0], :].reset_index(drop=True)
    df_te = df_merge.iloc[df_tr.shape[0]:, :].reset_index(drop=True)
    return df_tr, df_te


def calculate_vif(df, columns):
    """
    Calculate Variance Inflation Factor for multicollinearity detection.

    Parameters:
    -----------
    df : DataFrame
        Input data
    columns : list
        Columns to analyze

    Returns:
    --------
    DataFrame : VIF values for each feature
    """
    vif = pd.DataFrame()
    vif["VIF Factor"] = [variance_inflation_factor(df[columns].values, i) for i in range(len(columns))]
    vif["features"] = columns
    return vif.sort_values(by="VIF Factor", ascending=False)


# =============================================================================
# MODEL EVALUATION
# =============================================================================

def rmse(y_true, y_pred):
    """
    Calculate Root Mean Squared Error.

    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values

    Returns:
    --------
    float : RMSE value
    """
    from sklearn.metrics import mean_squared_error
    return round(mean_squared_error(y_true, y_pred, squared=False), 4)


# =============================================================================
# CONSTANTS
# =============================================================================

# Process steps
LST_STEPS = ['04', '06', '12', '13', '17', '18', '20']

# Step gaps for duration calculation
LST_STEPSGAP = [
    '0406', '0412', '0413', '0417', '0418',
    '0612', '0613', '0617', '0618', '0620',
    '1213', '1217', '1218', '1220',
    '1317', '1318', '1320',
    '1718', '1720',
    '1820'
]


def get_sensor_columns(df):
    """Get sensor column names from DataFrame."""
    col_sensor = df.iloc[:, 4:-7].columns.tolist()
    return col_sensor


def get_time_columns(df):
    """Get time column names from DataFrame."""
    return df.filter(regex='end').columns.tolist()


def get_sensor_names(col_sensor, lst_steps):
    """Extract unique sensor names from column list."""
    lst_sensors = []
    for step in lst_steps:
        _ = [col for col in col_sensor if col[:2] == step]
        lst_sensors.append(_)
    sensors_nm = list(map(lambda x: x[3:], lst_sensors[0]))
    return sensors_nm, lst_sensors
