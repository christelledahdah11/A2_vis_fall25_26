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
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["🏆 Rankings", "📈 Relationships", "🧮 Correlations", "📊 Distribution", "🧠 Auto-Insights"]
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

# ===== TAB 2: Relationships =====
with tab2:
    st.subheader("Tourism Index vs Facilities")
    rel_cols = list(facility_cols.values())
    col1, col2 = st.columns(2, gap="large")

    with col1:
        sel_fac = st.selectbox("X-axis facility", rel_cols, index=rel_cols.index("Total number of restaurants"))
        fig_scatter = px.scatter(
            filt,
            x=sel_fac,
            y="Tourism Index",
            color="Town",
            size=filt[rel_cols].sum(axis=1),  # proxy size by total facilities for the row
            trendline="ols",
            hover_name="Town",
            title=None,
        )
        fig_scatter.update_layout(xaxis_title=sel_fac, yaxis_title="Tourism Index")
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col2:
        st.subheader("Facility mix vs Index (ternary)")
        tern = filt.rename(columns={
            "Total number of hotels": "Hotels",
            "Total number of cafes": "Cafes",
            "Total number of restaurants": "Restaurants",
        }).copy()

        # avoid negatives/NaNs and all-zero rows
        tern[["Hotels", "Cafes", "Restaurants"]] = tern[["Hotels", "Cafes", "Restaurants"]].clip(lower=0).fillna(0)
        tern = tern[(tern[["Hotels", "Cafes", "Restaurants"]].sum(axis=1) > 0)]

        fig_tern = px.scatter_ternary(
            tern,
            a="Hotels", b="Cafes", c="Restaurants",
            color="Tourism Index",
            hover_name="Town",
            size=tern[["Hotels", "Cafes", "Restaurants"]].sum(axis=1),
        )
        st.plotly_chart(fig_tern, use_container_width=True)

# ===== TAB 3: Correlations =====
with tab3:
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

# ===== TAB 4: Distribution =====
with tab4:
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

# ===== TAB 5: Auto-Insights =====
with tab5:
    st.subheader("Signal, not noise.")

    # Leaders & laggards by Tourism Index
    leaders = filt.nlargest(3, "Tourism Index")[["Town", "Tourism Index"]]
    laggards = filt.nsmallest(3, "Tourism Index")[["Town", "Tourism Index"]]

    # Which facility best aligns with Tourism Index (highest corr)
    corr_idx = corr["Tourism Index"].drop("Tourism Index").idxmax()
    corr_val = float(corr["Tourism Index"].loc[corr_idx])

    # Outliers by facility intensity (|z| >= 1.5)
    out = town_agg[(town_agg["Facility Intensity (z)"].abs() >= 1.5) &
                   (town_agg["Tourism Index"] >= idx_cut)]
    out = out.sort_values("Facility Intensity (z)", ascending=False)

    with st.container(border=True):
        st.markdown("**Leaders (Top 3 by Tourism Index)**")
        st.table(leaders.set_index("Town"))

    with st.container(border=True):
        st.markdown("**Laggards (Bottom 3 by Tourism Index)**")
        st.table(laggards.set_index("Town"))

    with st.container(border=True):
        st.markdown("**Best-aligned facility with Tourism Index**")
        st.write(f"- **{corr_idx}** shows the strongest positive alignment with Tourism Index (ρ = {corr_val:.2f}).")
        st.caption("Correlation is descriptive, not causal. Validate with domain context.")

    with st.container(border=True):
        st.markdown("**Outlier towns by facility intensity (z-score ≥ 1.5)**")
        if len(out):
            show = out[["Town", "Tourism Index", "Total Facilities", "Facility Intensity (z)"]]
            st.dataframe(show.sort_values("Facility Intensity (z)", ascending=False), use_container_width=True)
        else:
            st.write("No material outliers given current filters.")

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
