"""
ai_summary.py

Generates a one-time AI summary of the uploaded dataset using the Gemini
API. Rather than sending the whole dataframe to the model (expensive, slow,
and unnecessary), this builds a compact statistical "profile" of the data
-- shape, column types, per-column stats, missing values, outlier counts --
and asks Gemini to turn that into a short, readable narrative.

This is a single API call per upload, not an open chat -- see README for
the reasoning behind that design choice (cost control + no hallucination
risk from unconstrained Q&A).
"""

import os

import pandas as pd
from google import genai
from google.genai import errors as genai_errors
from dotenv import load_dotenv

load_dotenv()

def _get_api_key(api_key: str | None = None) -> str | None:
    """
    Resolve the Gemini API key from (in order of priority):
    1. An explicitly passed argument
    2. The GEMINI_API_KEY environment variable (local dev, via .env)
    3. Streamlit secrets (when deployed on Streamlit Community Cloud)
    """
    if api_key:
        return api_key

    env_key = os.environ.get("GEMINI_API_KEY")
    if env_key:
        return env_key

    try:
        import streamlit as st
        return st.secrets.get("GEMINI_API_KEY")
    except Exception:
        return None


def build_data_profile(df: pd.DataFrame, column_types: dict, outlier_results: dict) -> str:
    """
    Build a compact text summary of the dataset's shape and statistics.
    This is what actually gets sent to Gemini -- not the raw data -- to
    keep the prompt small, cheap, and free of any row-level PII exposure.
    """
    lines = []
    lines.append(f"Dataset shape: {len(df)} rows, {len(df.columns)} columns.")
    lines.append("")
    lines.append("Column types:")
    for col, col_type in column_types.items():
        lines.append(f"  - {col}: {col_type}")
    lines.append("")

    numeric_cols = [c for c, t in column_types.items() if t == "numeric"]
    categorical_cols = [c for c, t in column_types.items() if t == "categorical"]

    if numeric_cols:
        lines.append("Numeric column statistics:")
        stats = df[numeric_cols].describe().T
        for col, row in stats.iterrows():
            lines.append(
                f"  - {col}: mean={row['mean']:.2f}, median={df[col].median():.2f}, "
                f"std={row['std']:.2f}, min={row['min']:.2f}, max={row['max']:.2f}"
            )
        lines.append("")

    if categorical_cols:
        lines.append("Categorical column top values:")
        for col in categorical_cols:
            top_values = df[col].value_counts().head(3)
            top_str = ", ".join(f"{val} ({count})" for val, count in top_values.items())
            lines.append(f"  - {col}: {top_str}")
        lines.append("")

    lines.append("Outlier detection results:")
    for col, result in outlier_results.get("per_column", {}).items():
        if result["count"] > 0:
            lines.append(f"  - {col}: {result['count']} outlier(s) flagged ({result['method']} method)")
    mv = outlier_results.get("multivariate", {})
    if mv.get("count", 0) > 0:
        lines.append(
            f"  - Multivariate (Isolation Forest): {mv['count']} row(s) flagged as anomalous "
            f"across {len(mv.get('columns', []))} numeric columns combined."
        )
    lines.append(f"Total unique outlier rows (single-column methods): {outlier_results.get('total_unique_outlier_rows', 0)}")

    return "\n".join(lines)


def generate_summary(
    df: pd.DataFrame,
    column_types: dict,
    outlier_results: dict,
    api_key: str | None = None,
) -> str:
    """
    Generate a short, readable AI summary of the dataset's key patterns
    and trends using Gemini. Returns a markdown-formatted string.

    Raises a RuntimeError with a user-friendly message if the API key is
    missing or the API call fails, rather than letting a raw exception
    bubble up to the Streamlit UI.
    """
    resolved_key = _get_api_key(api_key)
    if not resolved_key:
        raise RuntimeError(
            "No Gemini API key found. Set GEMINI_API_KEY in your .env file "
            "(local) or in Streamlit secrets (deployed)."
        )

    profile = build_data_profile(df, column_types, outlier_results)

    prompt = (
        "You are a data analyst assistant. Below is a statistical profile "
        "of a dataset a user just uploaded (not the raw data). Write a "
        "short summary (3-5 bullet points) highlighting the most notable "
        "patterns, trends, or data quality issues. Be specific and "
        "reference actual column names and numbers from the profile. "
        "Avoid generic statements that could apply to any dataset.\n\n"
        f"{profile}"
    )

    try:
        client = genai.Client(api_key=resolved_key)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
    except genai_errors.APIError as e:
        raise RuntimeError(f"Gemini API error ({e.code}): {e.message}") from e
    except Exception as e:
        raise RuntimeError(f"Failed to generate AI summary: {e}") from e

    if not response.text:
        raise RuntimeError("Gemini returned an empty response.")

    return response.text
