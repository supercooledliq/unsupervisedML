# Clustering Techniques
# K-Means Clustering Example with Elbow Method
# This script demonstrates how to perform K-Means clustering on a dataset and use the elbow method to determine the optimal number of clusters.
# # SPDX-License-Identifier: MIT
# # SPDX-FileCopyrightText: 2023 


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Sample data
df = pd.DataFrame({
    'Age': [25, 34, 22, 27, 45, 52, 23, 43, 36, 29],
    'Spending Score': [77, 62, 88, 71, 45, 20, 85, 40, 50, 65]
})

# Scale the data
scaler = StandardScaler()
scaled_df = scaler.fit_transform(df)

# Elbow method
wcss = []  #wcss- within-clusters sum of squares(It's a metric to measure how tight the clusters are.) Lower WCSS = better clustering (but too many clusters = overfitting).
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42) #trying to find an optimal number of k clusters
    kmeans.fit(scaled_df)
    wcss.append(kmeans.inertia_)

# Plot WCSS vs k
plt.plot(range(1, 11), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.grid(True)
plt.show()