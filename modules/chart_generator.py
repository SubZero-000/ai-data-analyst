"""
chart_generator.py

Automatically generates appropriate Plotly charts based on column type:
- numeric      -> histogram (distribution)
- categorical  -> bar chart of value counts (top N, rest bucketed as "Other")
- datetime     -> record count over time, binned at an appropriate granularity

Also generates two "cross-column" charts when the data supports it:
- a correlation heatmap across all numeric columns (if there are >= 2)
- a time-series trend of a numeric column over a datetime column
  (if both exist)

All chart functions return a plotly.graph_objects.Figure, so the caller
(Streamlit) can just do st.plotly_chart(fig).
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def generate_numeric_chart(df: pd.DataFrame, column: str) -> go.Figure:
    """Histogram showing the distribution of a numeric column."""
    fig = px.histogram(
        df, x=column, nbins=30, title=f"Distribution of {column}"
    )
    fig.update_layout(bargap=0.05)
    return fig


def generate_categorical_chart(df: pd.DataFrame, column: str, top_n: int = 10) -> go.Figure | None:
    """
    Bar chart of value counts for a categorical column. If there are more
    than `top_n` distinct values, the smaller ones are bucketed into
    "Other" so the chart stays readable instead of showing 200 tiny bars.

    Returns None if the column looks like an identifier (e.g. a name or ID
    column where almost every value is unique) since a value-count bar
    chart of all-1s isn't a useful visualization.
    """
    non_null = df[column].dropna()

    if len(non_null) == 0:
        return None

    uniqueness_ratio = non_null.nunique() / len(non_null)
    if uniqueness_ratio > 0.9 and non_null.nunique() > top_n:
        return None

    value_counts = non_null.value_counts()

    if len(value_counts) > top_n:
        top_values = value_counts.iloc[:top_n]
        other_count = value_counts.iloc[top_n:].sum()
        top_values = pd.concat([top_values, pd.Series({"Other": other_count})])
        value_counts = top_values

    fig = px.bar(
        x=value_counts.index.astype(str),
        y=value_counts.values,
        title=f"Value counts for {column}",
        labels={"x": column, "y": "count"},
    )
    return fig


def generate_datetime_chart(df: pd.DataFrame, column: str) -> go.Figure:
    """
    Record count over time for a datetime column, binned at a granularity
    chosen based on the overall date range (daily/weekly/monthly).
    """
    series = df[column].dropna()

    if series.empty:
        fig = go.Figure()
        fig.update_layout(title=f"No valid dates in {column}")
        return fig

    date_range_days = (series.max() - series.min()).days

    if date_range_days <= 60:
        freq = "D"
        label = "day"
    elif date_range_days <= 730:
        freq = "W"
        label = "week"
    else:
        freq = "ME"
        label = "month"

    counts = series.dt.to_period(freq).value_counts().sort_index()
    counts.index = counts.index.astype(str)

    fig = px.line(
        x=counts.index,
        y=counts.values,
        title=f"Record count by {label} ({column})",
        labels={"x": column, "y": "count"},
        markers=True,
    )
    return fig


def generate_correlation_heatmap(df: pd.DataFrame, numeric_columns: list[str]) -> go.Figure | None:
    """
    Correlation heatmap across all numeric columns. Returns None if there
    are fewer than 2 numeric columns (a heatmap of 1 column is meaningless).
    """
    if len(numeric_columns) < 2:
        return None

    corr_matrix = df[numeric_columns].corr()

    fig = px.imshow(
        corr_matrix,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Correlation heatmap (numeric columns)",
    )
    return fig


def generate_time_series_chart(
    df: pd.DataFrame, datetime_column: str, numeric_column: str
) -> go.Figure | None:
    """
    Line chart of a numeric column's average value over time, using the
    given datetime column. Returns None if either column has no valid data.
    """
    subset = df[[datetime_column, numeric_column]].dropna()

    if subset.empty:
        return None

    date_range_days = (subset[datetime_column].max() - subset[datetime_column].min()).days
    freq = "D" if date_range_days <= 60 else ("W" if date_range_days <= 730 else "ME")

    grouped = (
        subset.set_index(datetime_column)[numeric_column]
        .resample(freq)
        .mean()
        .dropna()
    )

    if grouped.empty:
        return None

    fig = px.line(
        x=grouped.index.astype(str),
        y=grouped.values,
        title=f"{numeric_column} over time (by {datetime_column})",
        labels={"x": datetime_column, "y": f"avg {numeric_column}"},
        markers=True,
    )
    return fig


def generate_all_charts(df: pd.DataFrame, column_types: dict) -> dict:
    """
    Generate the full set of auto-charts for a cleaned dataframe.

    Returns:
    {
        "per_column": {col: plotly Figure, ...},   # one chart per column
        "correlation_heatmap": Figure or None,
        "time_series": Figure or None,              # first datetime x first numeric col found
    }
    """
    per_column_charts = {}
    numeric_cols = [col for col, t in column_types.items() if t == "numeric"]
    datetime_cols = [col for col, t in column_types.items() if t == "datetime"]

    for col, col_type in column_types.items():
        if col_type == "numeric":
            per_column_charts[col] = generate_numeric_chart(df, col)
        elif col_type == "categorical":
            chart = generate_categorical_chart(df, col)
            if chart is not None:
                per_column_charts[col] = chart
        elif col_type == "datetime":
            per_column_charts[col] = generate_datetime_chart(df, col)

    correlation_heatmap = generate_correlation_heatmap(df, numeric_cols)

    time_series = None
    if datetime_cols and numeric_cols:
        time_series = generate_time_series_chart(df, datetime_cols[0], numeric_cols[0])

    return {
        "per_column": per_column_charts,
        "correlation_heatmap": correlation_heatmap,
        "time_series": time_series,
    }