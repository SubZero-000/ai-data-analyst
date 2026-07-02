# AI Data Analyst

Upload a CSV and get automatic data cleaning, outlier detection, auto-generated charts, and a one-time AI-generated summary of key trends — all in a Streamlit app you can run locally or deploy for free.

## Features

- **Automatic cleaning** — type inference (numeric/datetime/categorical), missing value imputation (median for numeric, mode for categorical), duplicate row removal, and column name normalization.
- **Outlier detection** — choose between IQR or Z-score for per-column detection, plus a multivariate Isolation Forest pass that catches anomalies spanning multiple columns at once.
- **Auto-generated charts** — histograms for numeric columns, bar charts for categorical columns (with automatic "Other" bucketing for high-cardinality columns, and identifier-like columns skipped automatically), time-binned trend charts for datetime columns, a correlation heatmap, and a time-series overlay. You choose which columns to chart so wide datasets don't get cluttered.
- **One-time AI summary** — a single Gemini API call turns a statistical profile of your dataset (not the raw rows) into a short, readable narrative of the most notable patterns.
- **Export** — download the cleaned dataset as a new CSV.
- **Free to run and deploy** — Streamlit Community Cloud hosting is free, and Gemini's free tier covers the AI summary for typical/demo usage.

## Project structure

```
ai-data-analyst/
├── app.py                     # Main Streamlit app (entry point)
├── requirements.txt
├── .env.example                # Template for local environment variables
├── .gitignore
├── .streamlit/
│   └── config.toml            # Light theme configuration
└── modules/
    ├── __init__.py
    ├── data_cleaning.py        # CSV loading + cleaning pipeline
    ├── outlier_detection.py    # IQR, Z-score, and Isolation Forest methods
    ├── chart_generator.py      # Auto chart selection + Plotly generation
    └── ai_summary.py           # Gemini API call for the one-time summary
```

## Setup (local)

1. **Clone the repo and install dependencies:**

   ```bash
   git clone <your-repo-url>
   cd ai-data-analyst
   pip install -r requirements.txt
   ```

2. **Get a free Gemini API key** from [Google AI Studio](https://aistudio.google.com/apikey).

3. **Set up your environment variables:**

   ```bash
   cp .env.example .env
   ```

   Then open `.env` and paste in your key:

   ```
   GEMINI_API_KEY=your_actual_key_here
   ```

4. **Run the app:**

   ```bash
   streamlit run app.py
   ```

   It should open automatically at `http://localhost:8501`.

## Deployment (Streamlit Community Cloud — free)

1. Push this repo to GitHub (make sure `.env` is **not** committed — it's already excluded via `.gitignore`).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app**, select this repo, branch, and set the main file path to `app.py`.
4. Before deploying, add your API key under **Advanced settings → Secrets**:

   ```toml
   GEMINI_API_KEY = "your_actual_key_here"
   ```

5. Click **Deploy**. Your app will be live at a public `*.streamlit.app` URL.

> **Note on rate limits:** Gemini's free tier has generous but finite rate limits (requests per minute/day). Fine for a portfolio demo or moderate personal use — if this app gets real traffic, you'll want to either monitor usage or add stricter request throttling.

## How the AI summary works

Rather than sending your raw data to an LLM (slow, costly at scale, and a privacy consideration), the app builds a compact statistical **profile** of the dataset — shape, column types, per-column stats, top categories, and outlier counts — and sends only that to Gemini. This keeps the AI summary to a single, cheap API call per upload rather than an open-ended chat, which also avoids hallucination risk from unconstrained Q&A over your data.

## Tech stack

- **Streamlit** — UI and app framework
- **pandas / numpy** — data cleaning and manipulation
- **scikit-learn** — Isolation Forest for multivariate outlier detection
- **Plotly** — interactive charts
- **Gemini API** (`google-genai`) — one-time AI-generated dataset summary

## Limitations / possible future improvements

- Currently supports CSV only (no Excel, JSON, etc.)
- Outlier detection and charts are univariate/bivariate by default — no support yet for grouped/segmented analysis (e.g. "show outliers within each category")
- The AI summary is a single fixed call per upload — no follow-up Q&A by design (see above), though this could be added later with request throttling if desired
- No persistent storage — each session is independent; nothing is saved between visits
