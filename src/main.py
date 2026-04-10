import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import folium
from folium.plugins import HeatMap
import os

# -----------------------------
# 1. LOAD & FILTER DATA
# -----------------------------
def load_data():
    print("Loading dataset...")

    df = pd.read_csv("../data/animal_data.csv")

    print("Columns:", df.columns)

    # Rename columns (adjust if needed)
    df = df.rename(columns={
        "location-lat": "Latitude",
        "location-long": "Longitude",
        "individual-local-identifier": "animal_id"
    })

    # Keep required columns
    df = df[['Latitude', 'Longitude', 'animal_id']]

    # Clean data
    df = df.dropna()
    df = df.drop_duplicates()

    # ✅ IMPORTANT: Select ONE animal only
    selected_animal = df['animal_id'].iloc[0]
    df = df[df['animal_id'] == selected_animal]

    print("Using animal ID:", selected_animal)

    # Optional: limit size for clarity
    if len(df) > 1500:
        df = df.sample(1500, random_state=42)

    print("Filtered data points:", len(df))

    return df


# -----------------------------
# 2. APPLY DBSCAN (FIXED)
# -----------------------------
def apply_dbscan(df, eps=0.25, min_samples=5):
    coords = df[['Latitude', 'Longitude']]

    # Scale coordinates (VERY IMPORTANT)
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(coords)

    db = DBSCAN(eps=eps, min_samples=min_samples)
    df['Cluster'] = db.fit_predict(coords_scaled)

    return df


# -----------------------------
# 3. ANALYSIS
# -----------------------------
def analyze_clusters(df):
    total_points = len(df)
    noise_points = len(df[df['Cluster'] == -1])
    clusters = len(set(df['Cluster'])) - (1 if -1 in df['Cluster'] else 0)

    print("\n--- ANALYSIS ---")
    print(f"Total Points: {total_points}")
    print(f"Clusters Found: {clusters}")
    print(f"Noise Points: {noise_points}")


# -----------------------------
# 4. GRAPH VISUALIZATION
# -----------------------------
def plot_clusters(df):
    plt.figure()
    plt.scatter(df['Longitude'], df['Latitude'], c=df['Cluster'])
    plt.title("Animal Pathway Clusters (Filtered Real Data)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    os.makedirs("../outputs", exist_ok=True)
    plt.savefig("../outputs/clusters.png")
    plt.show()


# -----------------------------
# 5. MAP VISUALIZATION
# -----------------------------
def create_map(df):
    print("Creating map...")

    center = [df['Latitude'].mean(), df['Longitude'].mean()]
    m = folium.Map(
        location=center,
        zoom_start=6,
        tiles="CartoDB positron"
    )

    for _, row in df.iterrows():
        color = "red" if row['Cluster'] == -1 else "blue"

        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=3,
            color=color,
            fill=True
        ).add_to(m)

    os.makedirs("../outputs", exist_ok=True)
    m.save("../outputs/map.html")

    print("Map saved as map.html")


# -----------------------------
# 6. HEATMAP
# -----------------------------
def create_heatmap(df):
    print("Creating heatmap...")

    center = [df['Latitude'].mean(), df['Longitude'].mean()]
    m = folium.Map(
        location=center,
        zoom_start=6,
        tiles="CartoDB positron"
    )

    heat_data = df[['Latitude', 'Longitude']].values.tolist()
    HeatMap(heat_data).add_to(m)

    os.makedirs("../outputs", exist_ok=True)
    m.save("../outputs/heatmap.html")

    print("Heatmap saved as heatmap.html")


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    df = load_data()

    print("Applying DBSCAN...")
    df = apply_dbscan(df)

    print("Analyzing results...")
    analyze_clusters(df)

    print("Generating visualizations...")
    plot_clusters(df)
    create_map(df)
    create_heatmap(df)

    print("\nDone! Check outputs folder.")