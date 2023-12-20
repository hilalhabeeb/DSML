from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

iris=load_iris()
x=iris.data


kmeans=KMeans(n_clusters=3,random_state=42)
kmeans.fit(x)
cluster_labels=kmeans.labels_
print(cluster_labels)
centroids=kmeans.cluster_centers_
print(centroids)

plt.scatter(x[:,0],x[:,1],c=cluster_labels,cmap='viridis',marker='o',edgecolors='black')
plt.scatter(centroids[:,0],centroids[:,1],marker='*',s=200,c="red",label='centroids')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.legend()
plt.show()