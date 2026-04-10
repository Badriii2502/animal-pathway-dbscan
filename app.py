import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium

st.title("🐾 Animal Pathway Identification using DBSCAN")

# Sidebar
st.sidebar.header("DBSCAN Parameters")
eps = st.sidebar.slider("Epsilon (eps)", 0.1, 1.0, 0.3)
min_samples = st.sidebar.slider("Min Samples", 2, 20, 5)

# Upload
uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

# ✅ FIXED CONDITION
if uploaded_file is not None:

    st.success("Dataset uploaded successfully!")

    df = pd.read_csv(uploaded_file)

    st.write("### Original Columns:", df.columns)

    # Rename safely (handles different datasets)
    df.columns = df.columns.str.lower()

    if "location-lat" in df.columns and "location-long" in df.columns:
        df = df.rename(columns={
            "location-lat": "Latitude",
            "location-long": "Longitude"
        })
    elif "latitude" in df.columns and "longitude" in df.columns:
        pass
    else:
        st.error("Dataset must contain Latitude and Longitude columns!")
        st.stop()

    # Keep only required columns
    df = df[['Latitude', 'Longitude']].dropna().drop_duplicates()
    if len(df) > 500:
        df = df.sample(500, random_state=42)

    st.write("### Cleaned Data Preview")
    st.dataframe(df.head())

    # Scale data
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(df[['Latitude', 'Longitude']])

    # DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples)
    df['Cluster'] = db.fit_predict(coords_scaled)

    st.write("### Cluster Visualization")

    fig, ax = plt.subplots()
    ax.scatter(df['Longitude'], df['Latitude'], c=df['Cluster'])
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    st.pyplot(fig)

    # Map
    st.write("### Map Visualization")

    center = [df['Latitude'].mean(), df['Longitude'].mean()]
    m = folium.Map(location=center, zoom_start=5, tiles="CartoDB positron")

    for _, row in df.iterrows():
        color = "red" if row['Cluster'] == -1 else "blue"

        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=3,
            color=color,
            fill=True
        ).add_to(m)

    st_folium(m, width=700, height=500)

    # Heatmap
    st.write("### Heatmap")

    heat_map = folium.Map(location=center, zoom_start=5, tiles="CartoDB positron")
    HeatMap(df[['Latitude', 'Longitude']].values.tolist()).add_to(heat_map)

    st_folium(heat_map, width=700, height=500)

else:
    st.info("Please upload a dataset to begin.")