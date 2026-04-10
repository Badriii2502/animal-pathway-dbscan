import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("data/animal_data.csv")

# Rename columns if needed
df.columns = df.columns.str.lower()

if "location-lat" in df.columns and "location-long" in df.columns:
    df = df.rename(columns={
        "location-lat": "Latitude",
        "location-long": "Longitude"
    })

# Clean
df = df[['Latitude', 'Longitude']].dropna().drop_duplicates()

# Scale
scaler = StandardScaler()
coords_scaled = scaler.fit_transform(df[['Latitude', 'Longitude']])

# DBSCAN
db = DBSCAN(eps=0.3, min_samples=5)
df['Cluster'] = db.fit_predict(coords_scaled)

# Plot
plt.scatter(df['Longitude'], df['Latitude'], c=df['Cluster'])
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Animal Pathway Clusters")
plt.show()

print("Clusters:", len(set(df['Cluster'])) - (1 if -1 in df['Cluster'] else 0))