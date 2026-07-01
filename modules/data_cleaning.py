"""
data_cleaning.py

Handles loading a raw CSV and running it through a standard cleaning pipeline:
- type inference (numeric, datetime, categorical)
- missing value detection + imputation
- duplicate row detection
- basic column name normalization

Returns both the cleaned dataframe and a "report" dict summarizing what was
done, so the UI layer can show the user what changed.
"""

import pandas as pd
import numpy as np


def load_csv(uploaded_file) -> pd.DataFrame:
    """
    Load a CSV file (as passed in from Streamlit's file_uploader) into a
    pandas DataFrame. Raises a clear error if the file can't be parsed.
    """
    try:
        df = pd.read_csv(uploaded_file)
    except UnicodeDecodeError:
        # Retry with a more permissive encoding if UTF-8 fails
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, encoding="latin1")

    if df.empty:
        raise ValueError("The uploaded CSV is empty.")

    return df


def strip_and_nullify_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    """
    Many real-world CSVs have stray whitespace around values (e.g. " 29",
    "Bob "), and cells that are blank/whitespace-only rather than truly
    empty. Left alone, these break numeric/datetime type inference and
    inflate "non-null" counts with values that are really missing data.

    This strips whitespace from all string cells and converts any
    whitespace-only or empty string to a proper NaN.
    """
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == "object" or pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace(r"^\s*$", np.nan, regex=True)
            df[col] = df[col].replace("nan", np.nan)  # str(NaN) -> "nan"
    return df


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names: strip whitespace, lowercase, replace spaces
    with underscores. Keeps the dataset easier to work with programmatically.
    """
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(r"[^\w]", "", regex=True)
    )
    return df


def infer_and_convert_types(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Attempt to convert object columns to more useful types:
    - datetime if it parses cleanly
    - numeric if it parses cleanly
    - otherwise leave as categorical/text

    Returns the converted dataframe and a dict mapping column -> inferred type.
    """
    df = df.copy()
    inferred_types = {}

    for col in df.columns:
        # pandas 3.0+ defaults text columns to a native "str" dtype rather
        # than the legacy "object" dtype, so we check for both.
        if df[col].dtype == "object" or pd.api.types.is_string_dtype(df[col]):
            original_non_null_count = df[col].notna().sum()

            if original_non_null_count == 0:
                inferred_types[col] = "categorical"
                continue

            # Try datetime
            converted_dt = pd.to_datetime(df[col], errors="coerce")
            dt_success_rate = converted_dt.notna().sum() / original_non_null_count

            # Try numeric
            converted_num = pd.to_numeric(df[col], errors="coerce")
            num_success_rate = converted_num.notna().sum() / original_non_null_count

            # Success rate is measured against values that were ALREADY
            # non-null, not against total row count -- a column with 20%
            # legitimate missing data can still be 100% numeric among the
            # values that are actually present.
            if dt_success_rate >= 0.9:
                df[col] = converted_dt
                inferred_types[col] = "datetime"
            elif num_success_rate >= 0.9:
                df[col] = converted_num
                inferred_types[col] = "numeric"
            else:
                inferred_types[col] = "categorical"
        elif pd.api.types.is_numeric_dtype(df[col]):
            inferred_types[col] = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            inferred_types[col] = "datetime"
        else:
            inferred_types[col] = "categorical"

    return df, inferred_types


def handle_missing_values(df: pd.DataFrame, column_types: dict) -> tuple[pd.DataFrame, dict]:
    """
    Impute missing values based on column type:
    - numeric: median
    - categorical: mode (most frequent value)
    - datetime: left as-is (imputing dates is often misleading)

    Returns the imputed dataframe and a report of how many values were
    filled per column.
    """
    df = df.copy()
    report = {}

    for col in df.columns:
        missing_count = int(df[col].isna().sum())
        if missing_count == 0:
            continue

        col_type = column_types.get(col, "categorical")

        if col_type == "numeric":
            fill_value = df[col].median()
            df[col] = df[col].fillna(fill_value)
            report[col] = {
                "missing_count": missing_count,
                "strategy": "median",
                "fill_value": fill_value,
            }
        elif col_type == "categorical":
            mode_series = df[col].mode()
            fill_value = mode_series.iloc[0] if not mode_series.empty else "Unknown"
            df[col] = df[col].fillna(fill_value)
            report[col] = {
                "missing_count": missing_count,
                "strategy": "mode",
                "fill_value": fill_value,
            }
        else:  # datetime - leave as NaT, just report it
            report[col] = {
                "missing_count": missing_count,
                "strategy": "left_as_missing",
                "fill_value": None,
            }

    return df, report


def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Drop exact duplicate rows. Returns the deduplicated dataframe and the
    count of rows removed.
    """
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(df)
    return df, removed


def clean_dataframe(uploaded_file) -> dict:
    """
    Full cleaning pipeline. Takes a raw uploaded CSV file and runs it through
    every step above, returning everything the UI needs in one dict:

    {
        "raw_df": original dataframe (before cleaning),
        "clean_df": cleaned dataframe,
        "column_types": {col: "numeric"/"categorical"/"datetime"},
        "missing_value_report": {...},
        "duplicates_removed": int,
    }
    """
    raw_df = load_csv(uploaded_file)

    df = strip_and_nullify_whitespace(raw_df)
    df = normalize_column_names(df)
    df, column_types = infer_and_convert_types(df)
    df, missing_report = handle_missing_values(df, column_types)
    df, duplicates_removed = remove_duplicates(df)

    return {
        "raw_df": raw_df,
        "clean_df": df,
        "column_types": column_types,
        "missing_value_report": missing_report,
        "duplicates_removed": duplicates_removed,
    }
