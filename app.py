import os
import io
import json
import math
import requests
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Try ARIMA; fall back to no-forecast mode if statsmodels missing
try:
    from statsmodels.tsa.arima.model import ARIMA
    ARIMA_AVAILABLE = True
except Exception:
    ARIMA_AVAILABLE = False

# =============================================================================
# CONFIG
# =============================================================================
# If your data lives in a subfolder inside the repo (e.g., "data"), set it here:
DATA_SUBFOLDER = ""  # e.g., "data" if your files are in /data; keep "" if in repo root

# Exact filenames you provided
FILES = {
    "world_bank": "world_bank_data_with_scores_and_continent.csv",
    "sectors": "merged_sectors_data.csv",
    "destinations": "merged_destinations_data.csv",
    "capex_eda": "capex_EDA.xlsx",
}

# Optional: RAW GitHub fallback (only used if local files are missing)
# In Streamlit Cloud: Settings → Secrets → add RAW_BASE="https://raw.githubusercontent.com/<user>/<repo>/<branch>"
RAW_BASE = st.secrets.get("RAW_BASE", "").strip()

# Indicator weights (sum is arbitrary; normalized internally to present indicators)
# Include negative direction for e.g. Inflation (higher is worse)
INDICATOR_WEIGHTS = {
    # Economic
    "GDP growth (annual %)": 10,
    "GDP per capita, PPP (current international $)": 8,
    "Current account balance (% of GDP)": 6,
    "Foreign direct investment, net outflows (% of GDP)": 6,
    "Inflation, consumer prices (annual %)": {"weight": 5, "direction": "negative"},
    "Exports of goods and services (% of GDP)": 5,
    "Imports of goods and services (% of GDP)": 5,
    # Governance
    "Political Stability and Absence of Violence/Terrorism: Estimate": 12,
    "Government Effectiveness: Estimate": 10,
    "Control of Corruption: Estimate": 8,
    # Infrastructure/Readiness
    "Access to electricity (% of population)": 9,
    "Individuals using the Internet (% of population)": 8,
    "Total reserves in months of imports": 8,
}

# Aliases to harmonize columns
SCORE_ALIASES = ["Score", "composite_score", "viability_score", "score"]
YEAR_ALIASES = ["Year", "year", "YEAR"]
COUNTRY_ALIASES = ["Country", "country", "Country Name", "Country_Name", "CountryName"]
GRADE_ALIASES = ["Grade", "grade"]
CONTINENT_ALIASES = ["Continent", "continent", "Region"]

# =============================================================================
# HELPERS: paths & loading
# =============================================================================
def _compose_local_path(fname: str) -> str:
    base_dir = os.path.dirname(__file__)
    if DATA_SUBFOLDER:
        return os.path.join(base_dir, DATA_SUBFOLDER, fname)
    return os.path.join(base_dir, fname)

def _compose_raw_url(fname: str) -> str:
    if not RAW_BASE:
        return ""
    base = RAW_BASE.rstrip("/")
    prefix = (DATA_SUBFOLDER.strip("/") + "/") if DATA_SUBFOLDER else ""
    return f"{base}/{prefix}{fname}"

@st.cache_data(show_spinner=True)
def read_csv_local_or_raw(fname: str) -> pd.DataFrame:
    # 1) local
    p = _compose_local_path(fname)
    if os.path.exists(p):
        return pd.read_csv(p)
    # 2) RAW fallback
    url = _compose_raw_url(fname)
    if not url:
        raise FileNotFoundError(f"CSV not found locally and RAW_BASE not set.\nMissing file: {p}")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return pd.read_csv(io.BytesIO(r.content))

@st.cache_data(show_spinner=True)
def read_excel_local_or_raw(fname: str, sheet_name=0) -> pd.DataFrame:
    # 1) local
    p = _compose_local_path(fname)
    if os.path.exists(p):
        return pd.read_excel(p, sheet_name=sheet_name, engine="openpyxl")
    # 2) RAW fallback
    url = _compose_raw_url(fname)
    if not url:
        raise FileNotFoundError(f"Excel not found locally and RAW_BASE not set.\nMissing file: {p}")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return pd.read_excel(io.BytesIO(r.content), sheet_name=sheet_name, engine="openpyxl")

# =============================================================================
# HELPERS: schema harmonization & scoring
# =============================================================================
def first_present(df: pd.DataFrame, names: list[str]) -> str | None:
    for n in names:
        if n in df.columns:
            return n
    return None

def harmonize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename common variants to canonical: Country, Year, Score, Grade, Continent."""
    df = df.copy()
    # Country
    c = first_present(df, COUNTRY_ALIASES)
    if c and c != "Country":
        df.rename(columns={c: "Country"}, inplace=True)
    # Year
    y = first_present(df, YEAR_ALIASES)
    if y and y != "Year":
        df.rename(columns={y: "Year"}, inplace=True)
    # Score
    s = first_present(df, SCORE_ALIASES)
    if s and s != "Score":
        df.rename(columns={s: "Score"}, inplace=True)
    # Grade
    g = first_present(df, GRADE_ALIASES)
    if g and g != "Grade":
        df.rename(columns={g: "Grade"}, inplace=True)
    # Continent
    ct = first_present(df, CONTINENT_ALIASES)
    if ct and ct != "Continent":
        df.rename(columns={ct: "Continent"}, inplace=True)
    return df

def _minmax_per_group(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    mask = s.notna()
    if mask.sum() <= 1:
        return pd.Series(0.5, index=s.index)
    lo, hi = s[mask].min(), s[mask].max()
    if hi == lo:
        return pd.Series(0.5, index=s.index)
    return (s - lo) / (hi - lo)

def grade_by_year(df: pd.DataFrame, score_col: str = "Score", year_col: str = "Year") -> pd.Series:
    if score_col not in df.columns or year_col not in df.columns:
        return pd.Series(index=df.index, dtype="object")
    out = pd.Series(index=df.index, dtype="object")
    for y, g in df.groupby(year_col):
        pct = g[score_col].rank(pct=True)
        bins = [0, 0.25, 0.50, 0.75, 0.90, 1.0]
        labels = ["D", "C", "B", "A", "A+"]
        out.loc[g.index] = pd.cut(pct, bins=bins, labels=labels, include_lowest=True, right=True)
    return out

def compute_score_if_missing(df: pd.DataFrame, indicator_weights: dict, year_col="Year") -> pd.DataFrame:
    """
    If 'Score' missing, compute from available indicator columns using weights.
    Negative-direction indicators are inverted post min–max normalization.
    Normalization is performed per Year (if Year exists), otherwise across all rows.
    """
    if "Score" in df.columns:
        return df

    present = []
    for key, meta in indicator_weights.items():
        if key in df.columns:
            w = meta["weight"] if isinstance(meta, dict) else meta
            direction = meta.get("direction", "positive") if isinstance(meta, dict) else "positive"
            present.append((key, w, direction))
    if not present:
        return df  # nothing to compute with

    work = df.copy()

    # Min–max normalize each present indicator (per year if Year exists)
    for key, _, direction in present:
        norm_col = f"{key}__norm__"
        if year_col in work.columns:
            work[norm_col] = (
                work.groupby(year_col, group_keys=False)[key].transform(_minmax_per_group)
            )
        else:
            work[norm_col] = _minmax_per_group(work[key])

        if direction == "negative":
            work[norm_col] = 1 - work[norm_col]

    # Normalize weights to present indicators
    weights = np.array([w for _, w, _ in present], dtype=float)
    weights = weights / weights.sum()

    # Weighted sum into Score
    score = np.zeros(len(work), dtype=float)
    for (key, _, _), w in zip(present, weights):
        score += w * work[f"{key}__norm__"].fillna(0.5).to_numpy()
    work["Score"] = score

    return work

# =============================================================================
# LOAD EVERYTHING
# =============================================================================
@st.cache_data(show_spinner=True)
def load_all():
    wb = read_csv_local_or_raw(FILES["world_bank"])
    sectors = read_csv_local_or_raw(FILES["sectors"])
    dest = read_csv_local_or_raw(FILES["destinations"])
    capex = read_excel_local_or_raw(FILES["capex_eda"])  # sheet 0

    # Harmonize
    wb = harmonize_columns(wb)
    sectors = harmonize_columns(sectors)
    dest = harmonize_columns(dest)
    capex = harmonize_columns(capex)

    # Light column fixes for CAPEX (expect at least Year + CAPEX)
    if "Year" not in capex.columns:
        for alt in ["year", "YEAR", "Year "]:
            if alt in capex.columns:
                capex.rename(columns={alt: "Year"}, inplace=True)
                break
    if "CAPEX" not in capex.columns:
        for alt in ["capex", "Capex", "CAPEX ($B)", "Capex ($B)"]:
            if alt in capex.columns:
                capex.rename(columns={alt: "CAPEX"}, inplace=True)
                break

    # Add Score & Grade if missing
    wb = compute_score_if_missing(wb, INDICATOR_WEIGHTS, year_col="Year")
    if "Grade" not in wb.columns:
        wb["Grade"] = grade_by_year(wb, "Score", "Year")

    return wb, sectors, dest, capex

# =============================================================================
# CHARTS
# =============================================================================
def world_map(df: pd.DataFrame, year: int):
    if "Country" not in df.columns or "Score" not in df.columns or "Year" not in df.columns:
        st.info("Cannot render map (missing Country/Score/Year).")
        return
    d = df[df["Year"] == year].dropna(subset=["Country", "Score"]).copy()
    if d.empty:
        st.info("No data for selected year.")
        return
    fig = px.choropleth(
        d,
        locations="Country",
        locationmode="country names",
        color="Score",
        hover_name="Country",
        color_continuous_scale="Viridis",
        title=f"Country Viability Score — {year}",
    )
    fig.update_layout(margin=dict(l=0, r=0, t=50, b=0), height=520)
    st.plotly_chart(fig, use_container_width=True)

def grade_distribution(df: pd.DataFrame, year: int):
    if "Grade" not in df.columns or "Year" not in df.columns:
        st.info("Cannot render grade distribution (missing Grade/Year).")
        return
    d = df[df["Year"] == year]
    if d.empty:
        st.info("No data for selected year.")
        return
    counts = d["Grade"].value_counts().reindex(["A+","A","B","C","D"]).fillna(0)
    fig = go.Figure([go.Bar(x=counts.index, y=counts.values)])
    fig.update_layout(title=f"Grade distribution — {year}", xaxis_title="Grade", yaxis_title="# Countries")
    st.plotly_chart(fig, use_container_width=True)

def capex_line(capex_df: pd.DataFrame):
    if not {"Year","CAPEX"}.issubset(capex_df.columns):
        st.info("CAPEX sheet needs columns: Year, CAPEX.")
        st.dataframe(capex_df.head(10))
        return
    fig = px.line(capex_df.sort_values("Year"), x="Year", y="CAPEX", markers=True, title="Global CAPEX")
    st.plotly_chart(fig, use_container_width=True)

def simple_arima_forecast(series: pd.Series, horizon=5):
    if not ARIMA_AVAILABLE:
        return None
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 6:
        return None
    try:
        s.index = pd.RangeIndex(len(s))  # ensure 0..N-1
        model = ARIMA(s, order=(1,1,1))
        res = model.fit()
        f = res.get_forecast(steps=horizon)
        fc = f.predicted_mean
        ci = f.conf_int()
        lo, hi = ci.iloc[:, 0], ci.iloc[:, 1]
        return fc, lo, hi
    except Exception:
        return None

def capex_forecast_plot(capex_df: pd.DataFrame, horizon=5):
    if not {"Year","CAPEX"}.issubset(capex_df.columns):
        return
    out = simple_arima_forecast(capex_df.sort_values("Year")["CAPEX"], horizon=horizon)
    if out is None:
        st.info("Forecast unavailable (statsmodels not installed or insufficient data).")
        return
    fc, lo, hi = out
    base = capex_df.sort_values("Year")
    last_year = int(base["Year"].max())
    future_years = [last_year + i for i in range(1, horizon + 1)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=base["Year"], y=base["CAPEX"], mode="lines+markers", name="Actual"))
    fig.add_trace(go.Scatter(x=future_years, y=fc, mode="lines+markers", name="Forecast"))
    fig.add_trace(go.Scatter(x=future_years, y=lo, mode="lines", name="Lower", line=dict(dash="dash")))
    fig.add_trace(go.Scatter(x=future_years, y=hi, mode="lines", name="Upper", line=dict(dash="dash")))
    fig.update_layout(title="CAPEX Forecast (ARIMA 1,1,1)", xaxis_title="Year", yaxis_title="CAPEX")
    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# UI
# =============================================================================
st.set_page_config(page_title="FDI Analytics Dashboard", layout="wide")

st.sidebar.title("Data status")
for k, v in FILES.items():
    local_path = _compose_local_path(v)
    if os.path.exists(local_path):
        st.sidebar.success(f"{k}: found locally at `{os.path.basename(local_path)}`")
    else:
        if RAW_BASE:
            st.sidebar.warning(f"{k}: using RAW fallback")
        else:
            st.sidebar.error(f"{k}: NOT FOUND and no RAW_BASE set")

with st.sidebar.expander("RAW fallback (optional)"):
    st.write("Set `RAW_BASE` in secrets to fetch files if they are not local:")
    st.code('RAW_BASE="https://raw.githubusercontent.com/<user>/<repo>/<branch>"', language="toml")

st.title("FDI Analytics Dashboard")
st.caption("EDA • Viability Scoring • Forecasting • Comparisons • Sectors")

# Load all data
wb, sectors, dest, capex = load_all()

# Years list
years = sorted([int(y) for y in pd.to_numeric(wb.get("Year"), errors="coerce").dropna().unique()]) if "Year" in wb.columns else []
default_year = years[-1] if years else None

# Filters
col1, col2, col3 = st.columns([1,1,2])
with col1:
    year_sel = st.selectbox("Year", years, index=(len(years)-1) if years else 0, disabled=not years)
with col2:
    continents = ["All"] + sorted(wb["Continent"].dropna().unique()) if "Continent" in wb.columns else ["All"]
    cont_sel = st.selectbox("Continent", continents, index=0)
with col3:
    query = st.text_input("Search country (optional)")

# Apply filters to a working copy
wb_f = wb.copy()
if cont_sel != "All" and "Continent" in wb_f.columns:
    wb_f = wb_f[wb_f["Continent"] == cont_sel]
if query.strip() and "Country" in wb_f.columns:
    wb_f = wb_f[wb_f["Country"].str.contains(query.strip(), case=False, na=False)]

tabs = st.tabs(["Overview", "EDA", "Scoring", "Forecasting", "Compare", "Sectors"])

# ---------------- Overview ----------------
with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Countries tracked", f"{wb['Country'].nunique():,}" if "Country" in wb.columns else "—")
    with c2:
        st.metric("Years", f"{len(years)}")
    with c3:
        if years and "Score" in wb.columns and "Year" in wb.columns:
            med_score = wb.loc[wb["Year"] == year_sel, "Score"].median()
            st.metric("Median Score", f"{med_score:.2f}" if pd.notna(med_score) else "—")
        else:
            st.metric("Median Score", "—")
    with c4:
        if years and "Grade" in wb.columns and "Year" in wb.columns:
            g = wb.loc[wb["Year"] == year_sel, "Grade"]
            top_a = (g == "A").sum() + (g == "A+").sum()
            st.metric("A / A+ Countries", f"{int(top_a)}")
        else:
            st.metric("A / A+ Countries", "—")

    st.subheader("Global CAPEX Trend")
    capex_line(capex)

    st.subheader("World Map — Viability Score")
    if years:
        world_map(wb_f, year_sel)
    else:
        st.info("No Year column found in the world bank dataset.")

    c5, c6 = st.columns([2,1])
    with c5:
        st.subheader("Top Countries (sample)")
        if years and "Score" in wb_f.columns and "Year" in wb_f.columns:
            cols_show = ["Country", "Score", "Grade"]
            if "CAPEX" in wb_f.columns:  # optional
                cols_show.append("CAPEX")
            tmp = wb_f[wb_f["Year"] == year_sel].sort_values(["Score"], ascending=False)[cols_show].head(20)
            st.dataframe(tmp, use_container_width=True)
        else:
            st.info("Missing Year/Score for top table.")
    with c6:
        st.subheader("Grade Distribution")
        if years:
            grade_distribution(wb_f, year_sel)
        else:
            st.info("No Year column to compute distribution.")

# ---------------- EDA ----------------
with tabs[1]:
    st.subheader("Sector CAPEX breakdown (merged_sectors_data.csv)")
    sec = sectors.copy()
    # Harmonize common fields
    if "Year" not in sec.columns:
        for alt in ["year", "YEAR"]:
            if alt in sec.columns:
                sec.rename(columns={alt: "Year"}, inplace=True)
                break
    if "Sector" not in sec.columns:
        for alt in ["sector", "SECTOR", "Industry"]:
            if alt in sec.columns:
                sec.rename(columns={alt: "Sector"}, inplace=True)
                break
    # pick a numeric value column
    value_col = None
    for cand in ["CAPEX","Capex","Value","Share","Amount","capex_usd_b","capex_busd","capex"]:
        if cand in sec.columns:
            value_col = cand
            break
    if {"Year","Sector"}.issubset(sec.columns) and value_col:
        s = sec[sec["Year"] == year_sel] if years else sec
        fig = px.pie(s, names="Sector", values=value_col, title=f"Sectors — {year_sel}" if years else "Sectors")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Expected columns in sectors: Year, Sector, and a numeric value column like CAPEX.")
        st.dataframe(sec.head(25), use_container_width=True)

    st.divider()
    st.subheader("Destination patterns (merged_destinations_data.csv)")
    st.dataframe(dest.head(25), use_container_width=True)

# ---------------- Scoring ----------------
with tabs[2]:
    st.subheader("Indicator Weights (used if Score missing)")
    rows = []
    for k, meta in INDICATOR_WEIGHTS.items():
        w = meta["weight"] if isinstance(meta, dict) else meta
        direction = meta.get("direction", "positive") if isinstance(meta, dict) else "positive"
        rows.append({"Indicator": k, "Weight %": w, "Direction": "+" if direction=="positive" else "-", "Present": k in wb.columns})
    st.dataframe(pd.DataFrame(rows).sort_values("Weight %", ascending=False), use_container_width=True)

    st.divider()
    st.subheader(f"Country scores — {year_sel}" if years else "Country scores")
    cols_to_show = ["Country", "Score", "Grade"]
    extra_inds = [c for c in INDICATOR_WEIGHTS.keys() if c in wb.columns]
    cols_to_show += extra_inds
    df_scores = wb_f[wb_f["Year"] == year_sel][cols_to_show].sort_values("Score", ascending=False) if years else wb_f[cols_to_show]
    st.dataframe(df_scores, use_container_width=True)

# ---------------- Forecasting ----------------
with tabs[3]:
    st.subheader("CAPEX Forecast")
    capex_forecast_plot(capex, horizon=5)

# ---------------- Compare ----------------
with tabs[4]:
    st.subheader("Compare two countries")
    if "Country" in wb_f.columns:
        countries = sorted(wb_f["Country"].dropna().unique().tolist())
        c1, c2, c3 = st.columns(3)
        with c1:
            country_a = st.selectbox("Country A", countries, index=0 if countries else None)
        with c2:
            country_b = st.selectbox("Country B", countries, index=1 if len(countries) > 1 else (0 if countries else None))
        with c3:
            metric = st.selectbox("Metric", ["Score","Grade","GDP growth (annual %)","Inflation, consumer prices (annual %)","GDP per capita, PPP (current international $)"])
        if countries:
            if years:
                d = wb_f[(wb_f["Country"].isin([country_a, country_b])) & (wb_f["Year"] == year_sel)]
            else:
                d = wb_f[wb_f["Country"].isin([country_a, country_b])]
            if metric == "Grade":
                st.dataframe(d[["Country","Grade"]], use_container_width=True)
            else:
                cols = [c for c in [metric, "Score", "Grade"] if c in d.columns]
                st.dataframe(d[["Country"] + cols], use_container_width=True)
    else:
        st.info("No Country column to compare.")

# ---------------- Sectors ----------------
with tabs[5]:
    st.subheader("Sector trends (top lines)")
    sec = sectors.copy()
    if "Year" not in sec.columns:
        for alt in ["year", "YEAR"]:
            if alt in sec.columns:
                sec.rename(columns={alt: "Year"}, inplace=True)
                break
    if "Sector" not in sec.columns:
        for alt in ["sector", "SECTOR", "Industry"]:
            if alt in sec.columns:
                sec.rename(columns={alt: "Sector"}, inplace=True)
                break
    value_col = None
    for cand in ["CAPEX","Capex","Value","Share","Amount","capex_usd_b","capex_busd","capex"]:
        if cand in sec.columns:
            value_col = cand; break
    if {"Year","Sector"}.issubset(sec.columns) and value_col:
        top = sec.groupby("Sector")[value_col].mean(numeric_only=True).sort_values(ascending=False).head(5).index.tolist()
        s = sec[sec["Sector"].isin(top)]
        fig = px.line(s, x="Year", y=value_col, color="Sector", markers=True, title="Top sectors over time")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No numeric sector value column found or missing Year/Sector.")
