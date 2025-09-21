# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --------- Page setup ---------
st.set_page_config(page_title="Tourism Insights", layout="wide")
st.title("📊 Tourism Insights Dashboard")

CSV_PATH = "DataSource Demographic.csv"  # make sure this file is alongside app.py

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Standardize column names (strip spaces)
    df.columns = [c.strip() for c in df.columns]
    # Expected columns
    expected = {
        "Town": "Town",
        "Tourism Index": "Tourism Index",
        "Total number of hotels": "Total number of hotels",
        "Total number of cafes": "Total number of cafes",
        "Total number of restaurants": "Total number of restaurants",
    }
    # Validate presence
    missing = [k for k in expected if k not in df.columns]
    if missing:
        st.error(f"Missing columns in CSV: {', '.join(missing)}")
        st.stop()

    # Coerce numerics and clean
    num_cols = [c for c in df.columns if c != "Town"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Tourism Index"])  # must have index
    df["Town"] = df["Town"].astype(str)
    return df

df = load_data(CSV_PATH)

facility_cols = {
    "Hotels": "Total number of hotels",
    "Cafes": "Total number of cafes",
    "Restaurants": "Total number of restaurants",
}

# Derived metrics (per-town totals)
town_agg = (
    df.groupby("Town", as_index=False)[["Tourism Index", *facility_cols.values()]]
    .mean()  # if there are multiple rows per town, average index and counts
)
town_agg["Total Facilities"] = town_agg[list(facility_cols.values())].sum(axis=1)

# Normalized “facility intensity” (z-score across towns)
for k, v in facility_cols.items():
    mu, sd = town_agg[v].mean(), town_agg[v].std(ddof=0) or 1.0
    town_agg[f"{k} z"] = (town_agg[v] - mu) / sd
town_agg["Facility Intensity (z)"] = town_agg[[f"{k} z" for k in facility_cols]].mean(axis=1)

# --------- Sidebar controls ---------
st.sidebar.header("Controls")
metric_choice = st.sidebar.selectbox(
    "Facility metric",
    ["Total Facilities", *facility_cols.values()],
    index=0,
)

min_index = int(np.floor(town_agg["Tourism Index"].min()))
max_index = int(np.ceil(town_agg["Tourism Index"].max()))
idx_cut = st.sidebar.slider("Minimum Tourism Index", min_index, max_index, int(town_agg["Tourism Index"].median()))
top_n = st.sidebar.slider("Top N towns", 5, min(25, len(town_agg)), min(10, len(town_agg)))

town_pick = st.sidebar.multiselect(
    "Filter towns (optional)",
    options=sorted(town_agg["Town"].unique()),
    default=[],
)

# Apply filters
filt = town_agg.query("`Tourism Index` >= @idx_cut").copy()
if town_pick:
    filt = filt[filt["Town"].isin(town_pick)]

# --------- KPI row ---------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Avg Tourism Index", f"{filt['Tourism Index'].mean():.1f}")
k2.metric("Median Tourism Index", f"{filt['Tourism Index'].median():.1f}")
k3.metric("Total Facilities (filtered)", int(filt["Total Facilities"].sum()))
k4.metric("Towns (filtered)", len(filt))

# --------- Tabs ---------
tab1, tab2, tab3 = st.tabs(
    ["🏆 Rankings", "🧮 Correlations", "📊 Distribution"]
)

# ===== TAB 1: Rankings =====
with tab1:
    colA, colB = st.columns([2, 1], gap="large")

    with colA:
        st.subheader(f"Top {top_n} towns by {metric_choice}")
        top_df = filt.nlargest(top_n, metric_choice)
        fig_bar = px.bar(
            top_df,
            x="Town",
            y=metric_choice,
            color="Tourism Index",
            text=metric_choice,
            title=None,
        )
        fig_bar.update_traces(texttemplate="%{text:.0f}", textposition="outside", cliponaxis=False)
        fig_bar.update_layout(yaxis_title=metric_choice, xaxis_title=None, showlegend=True)
        st.plotly_chart(fig_bar, use_container_width=True)

    with colB:
        st.subheader("Share of facilities by type")
        share = (
            filt[["Town", *facility_cols.values()]]
            .melt(id_vars="Town", var_name="Type", value_name="Count")
            .groupby("Type", as_index=False)["Count"]
            .sum()
        )
        fig_pie = px.pie(share, names="Type", values="Count", hole=0.45)
        st.plotly_chart(fig_pie, use_container_width=True)

    st.caption("Tip: use the sidebar to raise the Index threshold and watch leaders reshuffle.")

# ===== TAB 2: Correlations =====
with tab2:
    st.subheader("Correlation Heatmap")
    corr_cols = ["Tourism Index", *facility_cols.values(), "Total Facilities"]
    corr = filt[corr_cols].corr(numeric_only=True).round(2)
    fig_heat = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu",
        zmin=-1, zmax=1,
        aspect="auto",
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # Top pairwise correlations with Tourism Index
    topcorr = corr["Tourism Index"].drop("Tourism Index").sort_values(ascending=False)
    st.write("**Correlation with Tourism Index (descending):**")
    st.dataframe(topcorr.to_frame("Correlation"))

# ===== TAB 3: Distribution =====
with tab3:
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.subheader("Tourism Index distribution")
        fig_hist = px.histogram(filt, x="Tourism Index", nbins=15, marginal="box")
        st.plotly_chart(fig_hist, use_container_width=True)
    with c2:
        st.subheader("Facility totals per town (stacked)")
        long = filt.melt(
            id_vars=["Town", "Tourism Index"],
            value_vars=list(facility_cols.values()),
            var_name="Type",
            value_name="Count",
        )
        fig_stack = px.bar(long, x="Town", y="Count", color="Type", barmode="stack")
        st.plotly_chart(fig_stack, use_container_width=True)

# --------- Utilities ---------
st.divider()
colx, coly = st.columns([3, 1])
with colx:
    st.caption("Download filtered dataset")
with coly:
    st.download_button(
        label="⬇️ CSV",
        data=filt.to_csv(index=False).encode("utf-8"),
        file_name="tourism_filtered.csv",
        mime="text/csv",
        use_container_width=True,
    )
