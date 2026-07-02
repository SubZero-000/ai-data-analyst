"""
app.py

Main Streamlit entry point for the AI Data Analyst app.

Flow:
1. User uploads a CSV
2. Data is cleaned (modules/data_cleaning.py)
3. Outliers are detected (modules/outlier_detection.py)
4. Charts are auto-generated (modules/chart_generator.py)
5. A one-time AI summary is generated (modules/ai_summary.py)
"""

import streamlit as st
import pandas as pd
from PIL import Image

from modules.data_cleaning import clean_dataframe
from modules.outlier_detection import detect_all_outliers
from modules.chart_generator import generate_all_charts

try:
    from modules.ai_summary import generate_summary
    AI_SUMMARY_AVAILABLE = True
except ImportError:
    AI_SUMMARY_AVAILABLE = False

im = Image.open("favicon512.png")
st.set_page_config(
    page_title="AI Data Analyst",
    page_icon=im,
    layout="wide",
)


col1, col2 = st.columns([1, 8])
with col1:
    st.image("favicon512.png", width=60) 

with col2:
    st.title("AI Data Analyst")
st.caption("Upload a CSV and get automatic cleaning, outlier detection, charts, and an AI-generated summary.")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is None:
    st.info("Upload a CSV file to get started.")
    st.stop()

st.success(f"File '{uploaded_file.name}' uploaded successfully!")
# ---------------------------------------------------------------------------
# 1. Cleaning
# ---------------------------------------------------------------------------
try:
    with st.spinner("Cleaning data..."):
        cleaning_result = clean_dataframe(uploaded_file)
except ValueError as e:
    st.error(f"Couldn't process this file: {e}")
    st.stop()

clean_df = cleaning_result["clean_df"]
raw_df = cleaning_result["raw_df"]
column_types = cleaning_result["column_types"]
missing_report = cleaning_result["missing_value_report"]
duplicates_removed = cleaning_result["duplicates_removed"]

tab_overview, tab_cleaning, tab_outliers, tab_charts, tab_summary = st.tabs(
    ["Overview", "Cleaning Report", "Outliers", "Charts", "AI Summary"]
)

# ---------------------------------------------------------------------------
# Overview tab
# ---------------------------------------------------------------------------
with tab_overview:
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", len(clean_df))
    col2.metric("Columns", len(clean_df.columns))
    col3.metric("Duplicates removed", duplicates_removed)

    st.subheader("Preview")
    st.dataframe(clean_df.head(20), use_container_width=True)

    csv_bytes = clean_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download cleaned dataset as CSV",
        data=csv_bytes,
        file_name=f"cleaned_{uploaded_file.name}",
        mime="text/csv",
    )

    st.subheader("Detected column types")
    type_df = pd.DataFrame(
        {"column": list(column_types.keys()), "type": list(column_types.values())}
    )
    st.dataframe(type_df, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Cleaning Report tab
# ---------------------------------------------------------------------------
with tab_cleaning:
    st.subheader("Missing values")
    if missing_report:
        report_rows = []
        for col, info in missing_report.items():
            report_rows.append(
                {
                    "column": col,
                    "missing_count": info["missing_count"],
                    "strategy": info["strategy"],
                    "fill_value": info.get("fill_value"),
                }
            )
        st.dataframe(pd.DataFrame(report_rows), use_container_width=True, hide_index=True)
    else:
        st.success("No missing values found.")

    st.subheader("Duplicate rows")
    if duplicates_removed > 0:
        st.write(f"Removed **{duplicates_removed}** exact duplicate row(s).")
    else:
        st.success("No duplicate rows found.")

# ---------------------------------------------------------------------------
# Outliers tab
# ---------------------------------------------------------------------------
with tab_outliers:
    method = st.radio(
        "Detection method",
        options=["iqr", "zscore"],
        format_func=lambda m: "IQR (robust, default)" if m == "iqr" else "Z-score (assumes normal distribution)",
        horizontal=True,
    )

    with st.spinner("Detecting outliers..."):
        outlier_results = detect_all_outliers(clean_df, method=method)

    st.metric("Total unique outlier rows (single-column methods)", outlier_results["total_unique_outlier_rows"])

    st.subheader("Per-column results")
    per_col = outlier_results["per_column"]
    if per_col:
        rows = []
        for col, result in per_col.items():
            rows.append(
                {
                    "column": col,
                    "outlier_count": result["count"],
                    "note": result.get("note", ""),
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        selected_col = st.selectbox("View flagged rows for column:", list(per_col.keys()))
        flagged_indices = per_col[selected_col]["indices"]
        if flagged_indices:
            st.dataframe(clean_df.loc[flagged_indices], use_container_width=True)
        else:
            st.info("No outliers flagged for this column.")
    else:
        st.info("No numeric columns available for outlier detection.")

    st.subheader("Multivariate outliers (Isolation Forest)")
    mv = outlier_results["multivariate"]
    if mv.get("note"):
        st.info(mv["note"])
    else:
        st.write(f"Flagged **{mv['count']}** row(s) as anomalous across {len(mv['columns'])} numeric columns combined.")
        if mv["count"] > 0:
            st.dataframe(clean_df.loc[mv["indices"]], use_container_width=True)

# ---------------------------------------------------------------------------
# Charts tab
# ---------------------------------------------------------------------------
with tab_charts:
    all_columns = list(column_types.keys())
    selected_columns = st.multiselect(
        "Select columns to chart",
        options=all_columns,
        default=all_columns[: min(5, len(all_columns))],
        help="Charts are only generated for selected columns to avoid clutter on wide datasets.",
    )

    col_a, col_b = st.columns(2)
    show_correlation = col_a.checkbox("Show correlation heatmap", value=True)
    show_time_series = col_b.checkbox("Show trend over time", value=True)

    if not selected_columns:
        st.info("Select at least one column above to generate charts.")
    else:
        with st.spinner("Generating charts..."):
            charts = generate_all_charts(
                clean_df,
                column_types,
                selected_columns=selected_columns,
                include_correlation=show_correlation,
                include_time_series=show_time_series,
            )

        if charts["correlation_heatmap"] is not None:
            st.subheader("Correlation heatmap")
            st.plotly_chart(charts["correlation_heatmap"], use_container_width=True)

        if charts["time_series"] is not None:
            st.subheader("Trend over time")
            st.plotly_chart(charts["time_series"], use_container_width=True)

        st.subheader("Per-column charts")
        if not charts["per_column"]:
            st.info("No charts to display for the selected columns (e.g. identifier-like columns are skipped).")
        for col, fig in charts["per_column"].items():
            st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# AI Summary tab
# ---------------------------------------------------------------------------
with tab_summary:
    if not AI_SUMMARY_AVAILABLE:
        st.info("AI summary module isn't wired in yet -- coming next.")
    else:
        if st.button("Generate AI summary"):
            try:
                with st.spinner("Asking Gemini to summarize this dataset..."):
                    summary = generate_summary(clean_df, column_types, outlier_results)
                st.markdown(summary)
            except RuntimeError as e:
                st.error(str(e))