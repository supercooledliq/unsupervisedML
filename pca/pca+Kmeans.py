import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans 
import seaborn as sns 


# Simulated high-dimensional data
df_pca = pd.DataFrame({
    'Feature1': [2, 4, 1, 3, 5],
    'Feature2': [8, 10, 6, 7, 9],
    'Feature3': [0.5, 0.7, 0.3, 0.4, 0.6]
})

# Standardize and apply PCA
scaled = StandardScaler().fit_transform(df_pca)
pca = PCA(n_components=2)
reduced = pca.fit_transform(scaled)

#clustering through Kmeans
kmeans = KMeans(n_clusters=4, random_state=42) 
clusters = kmeans.fit_predict(reduced) 

# Visualize PCA results
plt.scatter(reduced[:, 0], reduced[:, 1])
plt.title("PCA Projection (2D)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.show()

 #--- Step 3: Visualize Clusters --- 
plt.figure(figsize=(5,3)) 
sns.scatterplot(x=reduced[:, 0],y= reduced[:, 1], hue='Cluster', 
palette='Set2', s=100) 
plt.title('Feature Segmentation via K-Means', fontsize=14) 
plt.xlabel('PCA Component 1') 
plt.ylabel('PCA Component 2') 
plt.grid(True) 
plt.legend(title="feature Segments") 
plt.show() 