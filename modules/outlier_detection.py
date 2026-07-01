"""
outlier_detection.py

Detects outliers in a cleaned dataframe using three methods:
- IQR (interquartile range) - robust, simple, good default for skewed data
- Z-score - assumes roughly normal distribution, sensitive to std deviation
- Isolation Forest - multivariate ML method, catches outliers that only
  look anomalous when multiple columns are considered together

Assumes the dataframe has already been through the cleaning pipeline
(modules/data_cleaning.py), so missing values shouldn't be present in the
numeric columns being scanned. Defensive NaN-handling is still included
since this module may be reused standalone.
"""

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.ensemble import IsolationForest


def get_numeric_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of column names with numeric dtype."""
    return [col for col in df.columns if is_numeric_dtype(df[col])]


def detect_outliers_iqr(df: pd.DataFrame, column: str) -> dict:
    """
    Detect outliers in a single column using the IQR method.
    Flags any value below Q1 - 1.5*IQR or above Q3 + 1.5*IQR.
    """
    series = df[column].dropna()

    q1 = np.percentile(series, 25)
    q3 = np.percentile(series, 75)
    iqr = q3 - q1

    if iqr == 0:
        # Column is constant (or near-constant) among non-null values.
        # Every differing value would technically fall outside a zero-width
        # band, which produces noisy, low-value "outliers." Safer to report
        # none rather than flood the user with false positives.
        return {
            "method": "iqr",
            "column": column,
            "count": 0,
            "indices": [],
            "lower_bound": float(q1),
            "upper_bound": float(q3),
            "note": "Column has near-zero variance; IQR method skipped.",
        }

    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)

    outliers_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
    indices = df[outliers_mask].index.tolist()

    return {
        "method": "iqr",
        "column": column,
        "count": len(indices),
        "indices": indices,
        "lower_bound": float(lower_bound),
        "upper_bound": float(upper_bound),
    }


def detect_outliers_zscore(df: pd.DataFrame, column: str, threshold: float = 3) -> dict:
    """
    Detect outliers in a single column using the z-score method.
    Flags any value more than `threshold` standard deviations from the mean.
    """
    series = df[column].dropna()

    mean = series.mean()
    std_dev = series.std()

    if std_dev == 0 or pd.isna(std_dev):
        # Constant column (or too few values to compute std) - z-scores
        # would be undefined (division by zero) or meaningless.
        return {
            "method": "zscore",
            "column": column,
            "count": 0,
            "indices": [],
            "threshold": threshold,
            "note": "Column has zero variance; z-score method skipped.",
        }

    z_scores = (df[column] - mean) / std_dev
    outliers_mask = z_scores.abs() > threshold
    indices = df[outliers_mask].index.tolist()

    return {
        "method": "zscore",
        "column": column,
        "count": len(indices),
        "indices": indices,
        "threshold": threshold,
    }


def detect_outliers_isolation_forest(
    df: pd.DataFrame, columns: list[str] = None, contamination: float = 0.05
) -> dict:
    """
    Detect outliers using Isolation Forest across multiple numeric columns
    at once (multivariate). Unlike IQR/z-score, this can catch rows that
    look fine column-by-column but are anomalous in combination.

    `contamination` is the expected proportion of outliers in the data
    (default 5%) - it's a rough prior, not a hard cutoff.
    """
    if columns is None:
        columns = get_numeric_columns(df)

    if len(columns) == 0:
        return {
            "method": "isolation_forest",
            "columns": [],
            "count": 0,
            "indices": [],
            "note": "No numeric columns available for multivariate detection.",
        }

    subset = df[columns].dropna()

    if len(subset) < 10:
        # Isolation Forest needs a reasonable sample size to be meaningful.
        return {
            "method": "isolation_forest",
            "columns": columns,
            "count": 0,
            "indices": [],
            "note": "Not enough rows for reliable multivariate detection.",
        }

    model = IsolationForest(contamination=contamination, random_state=42)
    predictions = model.fit_predict(subset)  # -1 = outlier, 1 = normal

    outlier_indices = subset.index[predictions == -1].tolist()

    return {
        "method": "isolation_forest",
        "columns": columns,
        "count": len(outlier_indices),
        "indices": outlier_indices,
    }


def detect_all_outliers(df: pd.DataFrame, method: str = "iqr") -> dict:
    """
    Run outlier detection across every numeric column in the dataframe
    using the specified method ("iqr" or "zscore"), plus one multivariate
    Isolation Forest pass across all numeric columns combined.

    Returns:
    {
        "per_column": {col: result_dict, ...},
        "multivariate": result_dict,
        "total_unique_outlier_rows": int,
    }
    """
    numeric_cols = get_numeric_columns(df)

    if method == "iqr":
        detector = detect_outliers_iqr
    elif method == "zscore":
        detector = detect_outliers_zscore
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'iqr' or 'zscore'.")

    per_column_results = {col: detector(df, col) for col in numeric_cols}
    multivariate_result = detect_outliers_isolation_forest(df, numeric_cols)

    # Union of all row indices flagged by any single-column method
    all_flagged_indices = set()
    for result in per_column_results.values():
        all_flagged_indices.update(result["indices"])

    return {
        "per_column": per_column_results,
        "multivariate": multivariate_result,
        "total_unique_outlier_rows": len(all_flagged_indices),
    }