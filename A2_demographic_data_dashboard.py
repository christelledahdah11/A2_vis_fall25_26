# app.py
import streamlit as st
import pandas as pd
import plotly.express as px

# Load your CSV (make sure it's in the same folder as this app.py)
df = pd.read_csv("DataSource Demographic.csv")   # 👈 replace with your actual filename

st.set_page_config(page_title="Tourism Insights", layout="wide")
st.title("📊 Tourism Insights Dashboard")

st.write("This dashboard shows how tourism facilities (hotels, cafes, restaurants) "
         "relate to the Tourism Index across towns. Use the controls in the sidebar.")

# --- Sidebar interactivity ---
facility = st.sidebar.selectbox(
    "Choose facility type:",
    ["Total number of hotels", "Total number of cafes", "Total number of restaurants"]
)

top_n = st.sidebar.slider("Show top N towns", 5, 20, 10)

tourism_cut = st.sidebar.slider(
    "Minimum Tourism Index",
    min_value=int(df["Tourism Index"].min()),
    max_value=int(df["Tourism Index"].max()),
    value=int(df["Tourism Index"].median())
)

# --- Chart 1: Bar ---
st.subheader(f"Top {top_n} towns by {facility}")
top_towns = df.groupby("Town")[facility].sum().nlargest(top_n)
fig_bar = px.bar(
    top_towns,
    x=top_towns.index,
    y=facility,
    title=f"Top {top_n} towns by {facility}"
)
st.plotly_chart(fig_bar, use_container_width=True)

# --- Chart 2: Scatter ---
st.subheader(f"Restaurants vs Tourism Index (Index ≥ {tourism_cut})")
filtered = df[df["Tourism Index"] >= tourism_cut]
fig_scatter = px.scatter(
    filtered,
    x="Total number of restaurants",
    y="Tourism Index",
    size="Total number of cafes",
    color="Total number of hotels",
    hover_name="Town",
    title="Restaurants vs Tourism Index"
)
st.plotly_chart(fig_scatter, use_container_width=True)

# --- Insights ---
st.markdown("### Key Insights")
st.write(f"- {facility} is concentrated in a few towns, with {top_towns.index[0]} leading.")
st.write(f"- Filtering shows {len(filtered)} towns with Tourism Index ≥ {tourism_cut}.")
