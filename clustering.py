import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd

# Step 1: Load preprocessed data for clustering
X_clustering_scaled = np.load('X_clustering_scaled.npy', allow_pickle=True)

# Step 2: Inspect loaded data
print("Type of X_clustering_scaled:", type(X_clustering_scaled))
print("Shape of X_clustering_scaled:", X_clustering_scaled.shape if hasattr(X_clustering_scaled, 'shape') else "No shape attribute")

# Step 3: Handle 0D numpy arrays 
if isinstance(X_clustering_scaled, np.ndarray) and X_clustering_scaled.shape == ():
    print("X_clustering_scaled is a 0D array, unwrapping the inner object...")
    X_clustering_scaled = X_clustering_scaled.item()

# Re-inspect after unwrapping
print("Type of X_clustering_scaled (after unwrapping):", type(X_clustering_scaled))
print("Shape of X_clustering_scaled (after unwrapping):", X_clustering_scaled.shape if hasattr(X_clustering_scaled, 'shape') else "No shape attribute")

# Step 4: Determine optimal number of clusters using the elbow method
inertia = []
k_range = range(1, 11)  # Test k from 1 to 10
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_clustering_scaled)
    inertia.append(kmeans.inertia_)

# Plot the elbow curve
plt.figure(figsize=(8, 6))
plt.plot(k_range, inertia, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.grid(True)
plt.savefig('elbow_plot.png')
plt.close()
print("Elbow plot saved as 'elbow_plot.png'")

# Step 5: Apply K-Means with chosen k
# Choose k=4 (based on number of genres, adjust if elbow plot suggests otherwise)
k = 4
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_clustering_scaled)

# Step 6: Save cluster labels
np.save('kmeans_cluster_labels.npy', cluster_labels)

# Step 7: Basic output to verify clustering
print("\nCluster Distribution:")
print(pd.Series(cluster_labels).value_counts().sort_index())