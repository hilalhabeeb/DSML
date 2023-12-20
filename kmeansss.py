from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = load_iris()
X = iris.data  # Features

# Initialize KMeans with k=3
kmeans = KMeans(n_clusters=3, random_state=42)

# Fit KMeans to the dataset
kmeans.fit(X)

# Getting the cluster labels and centroids
cluster_labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Visualizing the clusters and marking centroids (considering only the first two features for simplicity)
plt.scatter(X[:, 0], X[:, 1], c=cluster_labels, cmap='viridis', edgecolor='k', label='Data Points')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=200, c='red', label='Centroids')
plt.title('KMeans Clustering of Iris Dataset with Centroids')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
