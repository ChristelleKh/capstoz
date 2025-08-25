
# app.py
import os
import io
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="FDI Analytics (Streamlit)", layout="wide")

# ---------------------- Helpers ----------------------
DATA_FILES = {
    "scores": "world_bank_data_with_scores_and_continent.csv",
    "sectors": "merged_sectors_data.csv",
    "destinations": "merged_destinations_data.csv",
    "capex_eda": "capex_EDA.xlsx",
}

DEFAULT_WEIGHTS = {
    # ---- Economic ----
    "GDP growth (annual %)": {"weight": 0.10, "direction": "+"},
    "GDP per capita, PPP (current international $)": {"weight": 0.08, "direction": "+"},
    "Current account balance (% of GDP)": {"weight": 0.06, "direction": "+"},
    "Foreign direct investment, net outflows (% of GDP)": {"weight": 0.06, "direction": "+"},
    "Inflation, consumer prices (annual %)": {"weight": 0.05, "direction": "-"},
    "Exports of goods and services (% of GDP)": {"weight": 0.05, "direction": "+"},
    "Imports of goods and services (% of GDP)": {"weight": 0.05, "direction": "+"},
    # ---- Governance ----
    "Political Stability and Absence of Violence/Terrorism: Estimate": {"weight": 0.12, "direction": "+"},
    "Government Effectiveness: Estimate": {"weight": 0.10, "direction": "+"},
    "Control of Corruption: Estimate": {"weight": 0.08, "direction": "+"},
    # ---- Infrastructure / Financial Readiness ----
    "Access to electricity (% of population)": {"weight": 0.09, "direction": "+"},
    "Individuals using the Internet (% of population)": {"weight": 0.08, "direction": "+"},
    "Total reserves in months of imports": {"weight": 0.08, "direction": "+"},
}

CATEGORY_WEIGHTS = {
    "Economic": 0.45,
    "Governance": 0.30,
    "Infra/Financial": 0.25,
}

# Map indicators to categories (edit as needed)
INDICATOR_CATEGORY = {
    "GDP growth (annual %)": "Economic",
    "GDP per capita, PPP (current international $)": "Economic",
    "Current account balance (% of GDP)": "Economic",
    "Foreign direct investment, net outflows (% of GDP)": "Economic",
    "Inflation, consumer prices (annual %)": "Economic",
    "Exports of goods and services (% of GDP)": "Economic",
    "Imports of goods and services (% of GDP)": "Economic",
    "Political Stability and Absence of Violence/Terrorism: Estimate": "Governance",
    "Government Effectiveness: Estimate": "Governance",
    "Control of Corruption: Estimate": "Governance",
    "Access to electricity (% of population)": "Infra/Financial",
    "Individuals using the Internet (% of population)": "Infra/Financial",
    "Total reserves in months of imports": "Infra/Financial",
}

def load_local_or_upload(label, path, kind="csv"):
    st.sidebar.markdown(f"**{label}**")
    if os.path.exists(path):
        st.sidebar.success(f"Found `{os.path.basename(path)}`")
        if kind == "csv":
            return pd.read_csv(path)
        elif kind == "excel":
            return pd.read_excel(path)
    else:
        st.sidebar.warning(f"Missing `{os.path.basename(path)}` – upload below")
        upl = st.sidebar.file_uploader(f"Upload {label} ({kind.upper()})", type=["csv","xlsx","xls"])
        if upl is not None:
            if kind == "csv" or upl.name.lower().endswith(".csv"):
                return pd.read_csv(upl)
            else:
                return pd.read_excel(upl)
    return None

def ensure_columns_case_insensitive(df):
    # Convenience: normalize columns to simple names for merging/selection
    df.columns = [c.strip() for c in df.columns]
    return df

def minmax_by_year(df, cols, year_col="year"):
    """MinMax normalize each indicator within each year for comparability."""
    out = df.copy()
    for col in cols:
        out[col+"_norm"] = np.nan
    for y, g in out.groupby(year_col):
        scaler = MinMaxScaler()
        sub = g[cols].astype(float)
        # Handle constant columns gracefully
        for c in cols:
            if sub[c].nunique(dropna=True) <= 1:
                out.loc[g.index, c + "_norm"] = 0.5  # midpoint when no variation
            else:
                vals = scaler.fit_transform(sub[[c]].values)
                out.loc[g.index, c + "_norm"] = vals.ravel()
    return out

def apply_direction(df, weights):
    """Invert normalized columns where direction is '-'."""
    for ind, meta in weights.items():
        norm_col = ind + "_norm"
        if norm_col in df.columns and meta.get("direction") == "-":
            df[norm_col] = 1 - df[norm_col]
    return df

def compute_scores(df, weights, indicator_category, category_weights, country_col="country", year_col="year"):
    indicators = [k for k in weights.keys() if k in df.columns]
    if not indicators:
        return df, [], "No matching indicator columns found. Please align column names."

    work = df[[country_col, year_col] + indicators].copy()
    work = minmax_by_year(work, indicators, year_col=year_col)
    work = apply_direction(work, weights)

    # Category sub-scores (simple average of normalized indicators within category)
    for cat in set(indicator_category.values()):
        cat_inds = [i for i in indicators if indicator_category.get(i) == cat]
        cat_norm_cols = [i + "_norm" for i in cat_inds if i + "_norm" in work.columns]
        if cat_norm_cols:
            work[f"{cat}_score"] = work[cat_norm_cols].mean(axis=1)
        else:
            work[f"{cat}_score"] = np.nan

    # Final composite as weighted sum of category scores
    work["composite_score"] = 0
    denom = sum(category_weights.values())
    for cat, w in category_weights.items():
        work["composite_score"] += w * work[f"{cat}_score"]
    work["composite_score"] = work["composite_score"] / denom

    # Yearly percentile grades
    def grade(p):
        if p >= 0.90: return "A+"
        if p >= 0.75: return "A"
        if p >= 0.50: return "B"
        if p >= 0.25: return "C"
        return "D"

    work["percentile"] = work.groupby(year_col)["composite_score"].rank(pct=True)
    work["grade"] = work["percentile"].apply(grade)

    used = indicators
    msg = f"Computed scores for {len(work[country_col].unique())} countries across {work[year_col].nunique()} years."
    return work, used, msg

def kpi_card(label, value, helptext=None):
    c = st.container()
    c.metric(label, value)
    if helptext:
        c.caption(helptext)
    return c

# ---------------------- Load Data ----------------------
st.sidebar.header("Data sources")
scores_df = load_local_or_upload("World Bank + Scoring (panel CSV)", DATA_FILES["scores"], kind="csv")
sectors_df = load_local_or_upload("Sectors CAPEX (CSV)", DATA_FILES["sectors"], kind="csv")
dest_df = load_local_or_upload("Destinations CAPEX (CSV)", DATA_FILES["destinations"], kind="csv")
capex_book = load_local_or_upload("CAPEX EDA (Excel)", DATA_FILES["capex_eda"], kind="excel")

# Column name hints for mapping
DEFAULT_COUNTRY_COL = "country"  # adjust if your CSV uses different names
DEFAULT_YEAR_COL = "year"        # adjust if your CSV uses different names

st.sidebar.header("Scoring weights")
weights_json = st.sidebar.text_area(
    "Edit weights JSON (sum within each category guided by 45/30/25 across cats).",
    value=json.dumps(DEFAULT_WEIGHTS, indent=2),
    height=320
)
try:
    WEIGHTS = json.loads(weights_json)
except Exception as e:
    st.sidebar.error(f"Invalid JSON: {e}")
    WEIGHTS = DEFAULT_WEIGHTS

# ---------------------- UI ----------------------
st.title("FDI Analytics — Streamlit App")
st.write("EDA • Viability Scoring • Forecasting • Comparisons • Sectors • Map")

tab_overview, tab_eda, tab_scoring, tab_forecast, tab_compare, tab_sectors, tab_map, tab_admin = st.tabs(
    ["Overview", "EDA", "Scoring", "Forecasting", "Compare", "Sectors", "Map", "Admin"]
)

# ---------------------- Overview ----------------------
with tab_overview:
    st.subheader("Overview")
    if scores_df is not None:
        scores_df = ensure_columns_case_insensitive(scores_df)
        # Try to infer country/year column names
        guess_country = DEFAULT_COUNTRY_COL if DEFAULT_COUNTRY_COL in scores_df.columns else scores_df.columns[0]
        guess_year = DEFAULT_YEAR_COL if DEFAULT_YEAR_COL in scores_df.columns else "year"
        if guess_year not in scores_df.columns:
            # Try common alternatives
            for alt in ["Year","YEAR","yr"]:
                if alt in scores_df.columns: guess_year = alt

        # KPIs
        n_countries = scores_df[guess_country].nunique()
        years_avail = sorted(scores_df[guess_year].dropna().unique())
        latest_year = int(years_avail[-1]) if len(years_avail) else None

        cols = st.columns(4)
        kpi_card("Countries tracked", n_countries)
        kpi_card("Years", len(years_avail))
        if "CAPEX_USD_B" in scores_df.columns:
            kpi_card("Global CAPEX ($B, latest)", f"{scores_df.loc[scores_df[guess_year]==latest_year, 'CAPEX_USD_B'].sum():,.0f}")
        else:
            kpi_card("Global CAPEX", "—", "Add CAPEX_USD_B column for this KPI")

        st.markdown("---")
        # Simple world map if composite_score present
        if "composite_score" in scores_df.columns and guess_year in scores_df.columns:
            latest = scores_df[scores_df[guess_year]==latest_year]
            fig = px.choropleth(
                latest, locations=guess_country, locationmode="country names",
                color="composite_score", hover_name=guess_country,
                color_continuous_scale="Viridis", title=f"Composite Score — {latest_year}"
            )
            fig.update_layout(coloraxis_colorbar_title="Score")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No composite scores yet. Go to **Scoring** tab to compute.")
    else:
        st.warning("Load the panel CSV in the sidebar to begin.")

# ---------------------- EDA ----------------------
with tab_eda:
    st.subheader("Exploratory Data Analysis")
    if sectors_df is not None:
        sectors_df = ensure_columns_case_insensitive(sectors_df)
        # Expect columns: country, year, sector, capex_usd_b (adjust if needed)
        c1, c2 = st.columns(2)
        yr_values = sorted(sectors_df["year"].dropna().unique()) if "year" in sectors_df.columns else []
        with c1:
            yr = st.selectbox("Year", yr_values, index=0 if len(yr_values)>0 else None)
        with c2:
            topn = st.slider("Top N sectors", 3, 12, 6)

        if "sector" in sectors_df.columns and "capex_usd_b" in sectors_df.columns and len(yr_values)>0:
            subset = sectors_df[sectors_df["year"]==yr]
            top = subset.groupby("sector", as_index=False)["capex_usd_b"].sum().sort_values("capex_usd_b", ascending=False).head(topn)
            fig = px.pie(top, names="sector", values="capex_usd_b", title=f"Sector Breakdown — {yr}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Expecting columns: sector, capex_usd_b (and year). Please align your CSV headers.")
    else:
        st.info("Load sectors CSV in sidebar to explore.")

# ---------------------- Scoring ----------------------
with tab_scoring:
    st.subheader("Viability Scoring")
    if scores_df is not None:
        scores_df = ensure_columns_case_insensitive(scores_df)
        country_col = DEFAULT_COUNTRY_COL if DEFAULT_COUNTRY_COL in scores_df.columns else scores_df.columns[0]
        year_col = DEFAULT_YEAR_COL if DEFAULT_YEAR_COL in scores_df.columns else "year"
        if year_col not in scores_df.columns:
            for alt in ["Year","YEAR","yr"]:
                if alt in scores_df.columns: year_col = alt
        result, used_inds, msg = compute_scores(scores_df, WEIGHTS, INDICATOR_CATEGORY, CATEGORY_WEIGHTS, country_col, year_col)
        st.caption(msg)
        if used_inds:
            st.success(f"Using indicators: {', '.join(used_inds)}")
            # Save merged result in session for other tabs
            st.session_state["scored"] = result
            # Grade distribution
            latest_year = int(sorted(result[year_col].unique())[-1])
            latest = result[result[year_col]==latest_year]
            dist = latest["grade"].value_counts().reindex(["A+","A","B","C","D"]).fillna(0).reset_index()
            dist.columns = ["grade","count"]
            fig = px.bar(dist, x="grade", y="count", title=f"Grade Distribution — {latest_year}")
            st.plotly_chart(fig, use_container_width=True)

            # Top countries table
            st.markdown("**Top Countries (latest year)**")
            top = latest.sort_values("composite_score", ascending=False)[[country_col,"composite_score","grade"]].head(25)
            st.dataframe(top, use_container_width=True)

            # Download
            csv = result.to_csv(index=False).encode("utf-8")
            st.download_button("Download scored dataset (CSV)", csv, file_name="scored_countries.csv", mime="text/csv")
        else:
            st.error("No indicator columns matched. Edit your weights keys to match CSV headers.")
    else:
        st.info("Load the panel CSV to compute scores.")

# ---------------------- Forecasting ----------------------
with tab_forecast:
    st.subheader("FDI Forecasting (ARIMAX-lite)")
    df = st.session_state.get("scored") if "scored" in st.session_state else scores_df
    if df is None:
        st.info("Compute scores first or load panel CSV.")
    else:
        df = ensure_columns_case_insensitive(df)
        country_col = DEFAULT_COUNTRY_COL if DEFAULT_COUNTRY_COL in df.columns else df.columns[0]
        year_col = DEFAULT_YEAR_COL if DEFAULT_YEAR_COL in df.columns else "year"

        # Candidate target columns for forecasting
        candidates = [c for c in df.columns if c.lower().startswith("capex") or c.lower().endswith("% of gdp")]
        if len(candidates)==0:
            candidates = [c for c in df.columns if c.lower().startswith("fdi") or "usd" in c.lower()]
        target_col = st.selectbox("Target series to forecast", candidates, index=0 if candidates else None)
        if target_col is None:
            st.warning("No numeric target column found. Add a CAPEX/FDI series to use forecasting.")
        else:
            country = st.selectbox("Country", sorted(df[country_col].unique()))
            exog_choices = [k+"_norm" for k in DEFAULT_WEIGHTS.keys() if k+"_norm" in df.columns]
            exog_sel = st.multiselect("Exogenous indicators (normalized)", exog_choices, default=exog_choices[:3])

            series = df[df[country_col]==country].sort_values(year_col)
            y = series[target_col].astype(float)
            if y.isna().sum() > 0 or len(y.dropna()) < 5:
                st.warning("Not enough data for forecasting this series/country. Choose another target or ensure at least 5 points.")
            else:
                try:
                    exog = series[exog_sel] if exog_sel else None
                    model = SARIMAX(y, order=(1,1,1), exog=exog, enforce_stationarity=False, enforce_invertibility=False)
                    res = model.fit(disp=False)
                    horizon = st.slider("Forecast horizon (years)", 1, 5, 3)
                    fcast = res.get_forecast(steps=horizon, exog=np.tile(exog.iloc[-1:].values, (horizon,1)) if exog_sel else None)
                    pred = fcast.predicted_mean
                    conf = fcast.conf_int(alpha=0.2)
                    fut_years = np.arange(series[year_col].max()+1, series[year_col].max()+1+horizon)
                    plot_df = pd.DataFrame({
                        "year": list(series[year_col]) + list(fut_years),
                        "value": list(y.values) + list(pred.values),
                        "type": ["Actual"]*len(y) + ["Forecast"]*horizon
                    })
                    fig = px.line(plot_df, x="year", y="value", color="type", title=f"{country} — {target_col} Forecast")
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("Model: SARIMAX(1,1,1) with optional exogenous variables (normalized indicators).")
                except Exception as e:
                    st.error(f"Forecasting error: {e}")

# ---------------------- Compare ----------------------
with tab_compare:
    st.subheader("Compare Countries")
    df = st.session_state.get("scored") if "scored" in st.session_state else scores_df
    if df is None:
        st.info("Compute scores first or load panel CSV.")
    else:
        df = ensure_columns_case_insensitive(df)
        country_col = DEFAULT_COUNTRY_COL if DEFAULT_COUNTRY_COL in df.columns else df.columns[0]
        year_col = DEFAULT_YEAR_COL if DEFAULT_YEAR_COL in df.columns else "year"
        countries = sorted(df[country_col].unique())
        c1, c2 = st.columns(2)
        if len(countries)==0:
            st.info("No countries in dataset.")
        else:
            a = c1.selectbox("Country A", countries, index=0)
            b = c2.selectbox("Country B", countries, index=1 if len(countries)>1 else 0)
            metric = st.selectbox("Metric", ["composite_score","grade"] + [k+"_norm" for k in DEFAULT_WEIGHTS.keys() if k+"_norm" in df.columns])
            sub = df[df[country_col].isin([a,b])]
            if metric == "grade":
                latest_years = sub.groupby(country_col)[year_col].max().reset_index()
                latest = latest_years.merge(sub, on=[country_col, year_col], how="left")
                st.dataframe(latest[[country_col, "grade","composite_score"]], use_container_width=True)
            else:
                fig = px.line(sub, x=year_col, y=metric, color=country_col, markers=True, title=f"{metric} over time")
                st.plotly_chart(fig, use_container_width=True)

# ---------------------- Sectors ----------------------
with tab_sectors:
    st.subheader("Sector Trends")
    if sectors_df is not None:
        sectors_df = ensure_columns_case_insensitive(sectors_df)
        if all(c in sectors_df.columns for c in ["year","sector","capex_usd_b"]):
            s_year = st.selectbox("Year", sorted(sectors_df["year"].unique()))
            fig = px.bar(sectors_df[sectors_df["year"]==s_year].groupby("sector", as_index=False)["capex_usd_b"].sum(),
                         x="sector", y="capex_usd_b", title=f"Sector CAPEX — {s_year}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Expected columns: year, sector, capex_usd_b")
    else:
        st.info("Load sectors CSV in sidebar to explore.")

# ---------------------- Map ----------------------
with tab_map:
    st.subheader("Interactive World Map (Composite Score)")
    df = st.session_state.get("scored") if "scored" in st.session_state else scores_df
    if df is None:
        st.info("Compute scores first or load panel CSV on the sidebar.")
    else:
        df = ensure_columns_case_insensitive(df)
        country_col = DEFAULT_COUNTRY_COL if DEFAULT_COUNTRY_COL in df.columns else df.columns[0]
        year_col = DEFAULT_YEAR_COL if DEFAULT_YEAR_COL in df.columns else "year"

        # Filters
        years = sorted(df[year_col].dropna().unique())
        if not years:
            st.warning("No year column found.")
        else:
            col1, col2 = st.columns([3,2])
            with col1:
                sel_year = st.slider("Year", int(min(years)), int(max(years)), int(max(years)))
            with col2:
                continent_values = ["All"]
                if "continent" in df.columns:
                    continent_values += sorted([c for c in df["continent"].dropna().unique()])
                sel_cont = st.selectbox("Continent", continent_values, index=0)

            plot_df = df[df[year_col]==sel_year].copy()
            if sel_cont != "All" and "continent" in plot_df.columns:
                plot_df = plot_df[plot_df["continent"]==sel_cont]

            # Ensure composite_score exists
            if "composite_score" not in plot_df.columns:
                st.info("No composite scores yet. Go to **Scoring** tab to compute.")
            else:
                # Choropleth by composite_score
                fig = px.choropleth(
                    plot_df,
                    locations=country_col,
                    locationmode="country names",
                    color="composite_score",
                    hover_name=country_col,
                    hover_data={
                        "composite_score": ':.3f',
                        "Economic_score": ':.3f' if "Economic_score" in plot_df.columns else False,
                        "Governance_score": ':.3f' if "Governance_score" in plot_df.columns else False,
                        "Infra/Financial_score": ':.3f' if "Infra/Financial_score" in plot_df.columns else False,
                        country_col: False
                    },
                    color_continuous_scale="Viridis",
                    range_color=(0,1),
                    title=f"Composite Score — {sel_year}" + (f" — {sel_cont}" if sel_cont!='All' else "")
                )
                fig.update_layout(coloraxis_colorbar_title="Score (0–1)")
                st.plotly_chart(fig, use_container_width=True)

                # Country picker to drill into time series
                st.markdown("**Drill-through: country time series**")
                c_list = sorted(plot_df[country_col].unique())
                if c_list:
                    chosen = st.selectbox("Country", c_list)
                    long = df[df[country_col]==chosen].sort_values(by=year_col)
                    if "composite_score" in long.columns:
                        ts = px.line(long, x=year_col, y="composite_score", markers=True, title=f"{chosen} — Composite Score over time")
                        st.plotly_chart(ts, use_container_width=True)
                    # Show top indicator contributors (if normalized cols exist)
                    norm_cols = [c for c in long.columns if c.endswith("_norm")]
                    if norm_cols:
                        latest = long[long[year_col]==sel_year][norm_cols].T.reset_index()
                        latest.columns = ["indicator","value"]
                        latest = latest.sort_values("value", ascending=False).head(10)
                        bar = px.bar(latest, x="indicator", y="value", title=f"{chosen} — Top normalized indicators ({sel_year})")
                        st.plotly_chart(bar, use_container_width=True)

# ---------------------- Admin ----------------------
with tab_admin:
    st.subheader("Admin & Utilities")
    st.write("**1) Export current weights JSON**")
    st.download_button("Download weights.json", data=json.dumps(WEIGHTS, indent=2), file_name="weights.json")

    st.write("**2) Notes**")
    st.markdown("""
    - Make sure your **panel CSV** has columns named exactly like the indicators in the weights JSON.
    - The app normalizes each indicator by year using Min–Max scaling, inverts negative indicators (e.g., inflation), then aggregates by category (45/30/25) into a composite score. Grades (A+..D) are assigned by yearly percentiles.
    - For forecasting, choose a target numeric series (e.g., CAPEX) and optional normalized indicators as exogenous regressors.
    """)

    st.write("**3) Column name alignment**")
    st.dataframe(pd.DataFrame({"expected_indicators": list(DEFAULT_WEIGHTS.keys())}))

    st.write("**4) Debug: Preview head of loaded dataframes**")
    with st.expander("scores_df (panel)"):
        st.dataframe(scores_df.head() if scores_df is not None else pd.DataFrame())
    with st.expander("sectors_df"):
        st.dataframe(sectors_df.head() if sectors_df is not None else pd.DataFrame())
    with st.expander("destinations_df"):
        st.dataframe(dest_df.head() if dest_df is not None else pd.DataFrame())
    with st.expander("capex_eda (excel)"):
        if isinstance(capex_book, pd.DataFrame):
            st.dataframe(capex_book.head())
        elif capex_book is not None:
            st.write("Workbook loaded. Select a sheet above to preview.")
        else:
            st.write("—")
