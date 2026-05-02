from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
PRED_PATH = ROOT / "outputs" / "predictions.csv"
PRICE_PATH = ROOT / "data" / "us_largecap_10y_daily.csv"
METRIC_PATH = ROOT / "outputs" / "metrics.json"

FEATURE_COLS = [
    "RetDenoised",
    "TrendLF",
    "TrendMF",
    "OscHF",
    "HilbertAmp",
    "SpecEntropy20",
    "VolZ20",
    "Velocity",
    "Acceleration",
    "FrictionGamma",
    "DiffusionSigma",
    "ForceProxy",
    "Potential",
    "Curvature",
    "EnergyImbalance",
]

BOOTSTRAP_TICKERS = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "META",
    "NVDA",
    "JPM",
    "XOM",
    "PG",
    "PEP",
    "COST",
    "MRK",
]


st.set_page_config(
    page_title="DSP + Physics Alpha Dashboard",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');
      :root {
        --bg: #f3f4ef;
        --card: #fcfcfa;
        --text: #1f2b24;
        --muted: #5f6f63;
        --accent: #007a5a;
        --accent-soft: #d8efe8;
        --border: #d8ded8;
      }
      .stApp {
        background: radial-gradient(circle at top right, #e4ede7 0%, var(--bg) 45%);
        color: var(--text);
        font-family: 'Space Grotesk', sans-serif;
      }
      .block-container {
        padding-top: 1.3rem;
        padding-bottom: 2rem;
      }
      div[data-testid="stMetric"] {
        background: linear-gradient(155deg, var(--card) 0%, #f6f8f4 100%);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 0.9rem 1rem;
      }
      div[data-testid="stMetricLabel"] {
        color: var(--muted);
        font-weight: 600;
      }
      div[data-testid="stMetricValue"] {
        color: var(--text);
      }
      .dash-card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1rem 1rem 0.4rem 1rem;
      }
      .stDataFrame {
        border-radius: 14px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def _artifacts_exist() -> bool:
    return PRED_PATH.exists() and PRICE_PATH.exists()


def _bootstrap_artifacts() -> None:
    import yfinance as yf

    from src.quant_dsp.backtest import walk_forward_backtest
    from src.quant_dsp.features import build_feature_table

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cache_dir = DATA_DIR / ".yfinance_tz_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(cache_dir.resolve()))

    frames: list[pd.DataFrame] = []
    for ticker in BOOTSTRAP_TICKERS:
        df = yf.download(ticker, period="5y", auto_adjust=False, progress=False)
        if df.empty:
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df = df.reset_index()
        if "Datetime" in df.columns and "Date" not in df.columns:
            df = df.rename(columns={"Datetime": "Date"})
        if "Adj Close" not in df.columns and "Close" in df.columns:
            df["Adj Close"] = df["Close"]
        df["Ticker"] = ticker
        frames.append(df[["Date", "Ticker", "Open", "High", "Low", "Close", "Adj Close", "Volume"]])

    if not frames:
        raise RuntimeError("Could not download market data for demo artifact generation.")

    panel = (
        pd.concat(frames, ignore_index=True)
        .dropna(subset=["Date", "Adj Close", "Volume"])
        .sort_values(["Date", "Ticker"])
    )
    panel.to_csv(PRICE_PATH, index=False)

    features = build_feature_table(panel)
    pred_df, metrics = walk_forward_backtest(
        features,
        feature_cols=FEATURE_COLS,
        train_days=252,
        test_days=63,
    )
    pred_df.to_csv(PRED_PATH, index=False)
    METRIC_PATH.write_text(json.dumps(metrics, indent=2))


def _ensure_artifacts() -> None:
    if _artifacts_exist():
        return

    st.warning(
        "Artifacts not found in this repo yet. Click below to generate a lightweight "
        "demo dataset and model outputs for the dashboard."
    )
    if st.button("Generate Demo Artifacts", type="primary"):
        with st.spinner("Downloading data and running DSP + physics model..."):
            try:
                _bootstrap_artifacts()
                st.success("Artifacts generated. Reloading dashboard...")
                st.rerun()
            except Exception as exc:
                st.error(f"Artifact generation failed: {exc}")
    st.stop()


_ensure_artifacts()


@st.cache_data(show_spinner=False)
def load_predictions() -> pd.DataFrame:
    if not PRED_PATH.exists():
        raise FileNotFoundError(f"Missing predictions file: {PRED_PATH}")
    df = pd.read_csv(PRED_PATH, parse_dates=["Date"])
    return df.sort_values(["Date", "Ticker"]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_prices() -> pd.DataFrame:
    if not PRICE_PATH.exists():
        raise FileNotFoundError(f"Missing prices file: {PRICE_PATH}")
    px_df = pd.read_csv(PRICE_PATH, parse_dates=["Date"])
    cols = ["Date", "Ticker", "Adj Close", "Volume"]
    return px_df[cols].sort_values(["Date", "Ticker"]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_metrics() -> dict:
    if not METRIC_PATH.exists():
        return {}
    return json.loads(METRIC_PATH.read_text())


def rank_ic(pred: pd.Series, target: pd.Series) -> float:
    mask = pred.notna() & target.notna()
    if mask.sum() < 5:
        return np.nan
    return pred[mask].rank().corr(target[mask].rank())


def compute_daily_stats(df: pd.DataFrame, quantile: float) -> pd.DataFrame:
    rows: list[dict] = []
    for dt, g in df.groupby("Date"):
        g = g.sort_values("Pred")
        n = len(g)
        if n < 8:
            continue
        k = max(1, int(n * quantile))
        short_ret = g.iloc[:k]["Target5"].mean()
        long_ret = g.iloc[-k:]["Target5"].mean()
        rows.append(
            {
                "Date": dt,
                "IC": rank_ic(g["Pred"], g["Target5"]),
                "LS": long_ret - short_ret,
            }
        )

    out = pd.DataFrame(rows).sort_values("Date")
    if out.empty:
        return out
    out["CumLS"] = (1.0 + out["LS"].fillna(0.0)).cumprod() - 1.0
    return out


def format_pct(x: float) -> str:
    return f"{x * 100:.2f}%"


pred = load_predictions()
prices = load_prices()
metrics = load_metrics()
merged = pred.merge(prices, on=["Date", "Ticker"], how="left")

st.title("DSP + Physics-Informed Alpha Dashboard")
st.caption("Interactive exploration of model predictions, cross-sectional alpha, and signal behavior.")

with st.sidebar:
    st.header("Filters")
    min_date = pred["Date"].min().date()
    max_date = pred["Date"].max().date()
    selected_dates = st.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    tickers = sorted(pred["Ticker"].unique().tolist())
    selected_tickers = st.multiselect(
        "Tickers",
        options=tickers,
        default=tickers,
    )

    quantile = st.slider("Long/Short quantile", 0.10, 0.40, 0.20, 0.05)
    top_n = st.slider("Top long/short rows", 5, 25, 10, 1)

if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
    start_date, end_date = selected_dates
else:
    start_date, end_date = (min_date, max_date)

flt = pred[
    (pred["Date"].dt.date >= start_date)
    & (pred["Date"].dt.date <= end_date)
    & (pred["Ticker"].isin(selected_tickers))
].copy()

if flt.empty:
    st.error("No rows match the active filters.")
    st.stop()

daily = compute_daily_stats(flt, quantile=quantile)
if daily.empty:
    st.error("Not enough cross-sectional names to compute long/short and IC series.")
    st.stop()

mean_ic = float(daily["IC"].mean())
ic_ir = float(daily["IC"].mean() / (daily["IC"].std() + 1e-9))
ls_mean = float(daily["LS"].mean())
ls_sharpe = float(daily["LS"].mean() / (daily["LS"].std() + 1e-9))

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Mean Rank IC", f"{mean_ic:.3f}")
with m2:
    st.metric("IC Information Ratio", f"{ic_ir:.3f}")
with m3:
    st.metric("Mean 5D L/S Return", format_pct(ls_mean))
with m4:
    st.metric("L/S Sharpe Proxy", f"{ls_sharpe:.3f}")

if metrics:
    with st.expander("Baseline run metrics", expanded=False):
        st.json(metrics)

c1, c2 = st.columns([1.6, 1])
with c1:
    st.markdown("<div class='dash-card'>", unsafe_allow_html=True)
    fig_cum = px.line(
        daily,
        x="Date",
        y="CumLS",
        template="plotly_white",
        title="Cumulative 5D Long-Short Return (Proxy)",
    )
    fig_cum.update_traces(line=dict(color="#007a5a", width=2.2))
    fig_cum.update_layout(height=360, margin=dict(l=12, r=10, t=40, b=10))
    st.plotly_chart(fig_cum, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with c2:
    st.markdown("<div class='dash-card'>", unsafe_allow_html=True)
    monthly_ic = daily.set_index("Date")["IC"].resample("ME").mean().reset_index()
    fig_ic = px.bar(
        monthly_ic,
        x="Date",
        y="IC",
        template="plotly_white",
        title="Monthly Mean Rank IC",
    )
    fig_ic.update_traces(marker_color="#2f9e78")
    fig_ic.update_layout(height=360, margin=dict(l=12, r=10, t=40, b=10))
    st.plotly_chart(fig_ic, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.subheader("Cross-Section Explorer")
selected_day = st.select_slider(
    "Inspection date",
    options=sorted(flt["Date"].dt.date.unique().tolist()),
    value=flt["Date"].max().date(),
)

day_slice = flt[flt["Date"].dt.date == selected_day].copy().sort_values("Pred")
if day_slice.empty:
    st.warning("No rows for the chosen inspection date.")
    st.stop()

left, right = st.columns(2)
with left:
    longs = day_slice.sort_values("Pred", ascending=False).head(top_n)
    st.markdown("**Top Long Candidates**")
    st.dataframe(
        longs[["Ticker", "Pred", "Target5"]].rename(columns={"Pred": "Signal", "Target5": "5D Forward"}),
        use_container_width=True,
        height=330,
    )

with right:
    shorts = day_slice.sort_values("Pred", ascending=True).head(top_n)
    st.markdown("**Top Short Candidates**")
    st.dataframe(
        shorts[["Ticker", "Pred", "Target5"]].rename(columns={"Pred": "Signal", "Target5": "5D Forward"}),
        use_container_width=True,
        height=330,
    )

s1, s2 = st.columns([1, 1.2])
with s1:
    fig_scatter = px.scatter(
        day_slice,
        x="Pred",
        y="Target5",
        hover_data=["Ticker"],
        template="plotly_white",
        title=f"Signal vs 5D Forward Return ({selected_day})",
    )
    fig_scatter.update_traces(marker=dict(size=10, color="#007a5a", opacity=0.75))
    fig_scatter.update_layout(height=360, margin=dict(l=12, r=10, t=40, b=10))
    st.plotly_chart(fig_scatter, use_container_width=True)

with s2:
    chosen_ticker = st.selectbox(
        "Ticker detail",
        options=sorted(day_slice["Ticker"].unique().tolist()),
        index=0,
    )
    tdf = merged[
        (merged["Ticker"] == chosen_ticker)
        & (merged["Date"].dt.date >= start_date)
        & (merged["Date"].dt.date <= end_date)
    ].copy()
    fig_detail = go.Figure()
    fig_detail.add_trace(
        go.Scatter(
            x=tdf["Date"],
            y=tdf["Adj Close"],
            name="Adj Close",
            line=dict(color="#1f2b24", width=2),
            yaxis="y1",
        )
    )
    fig_detail.add_trace(
        go.Scatter(
            x=tdf["Date"],
            y=tdf["Pred"],
            name="Signal",
            line=dict(color="#2f9e78", width=2, dash="dot"),
            yaxis="y2",
        )
    )
    fig_detail.update_layout(
        title=f"{chosen_ticker}: Price vs Model Signal",
        template="plotly_white",
        height=360,
        margin=dict(l=12, r=10, t=40, b=10),
        yaxis=dict(title="Adj Close"),
        yaxis2=dict(title="Signal", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_detail, use_container_width=True)

st.caption(
    "Note: returns shown are research proxies from rolling 5-day forward returns; "
    "they are not transaction-cost-adjusted live performance."
)
