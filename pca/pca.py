import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#You are taking a small dataset with 3 features and using PCA to compress it into 2 dimensions
# so you can visualize it — while still keeping most of the important variation in the data.

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

# Visualize PCA results
plt.scatter(reduced[:, 0], reduced[:, 1])
plt.title("PCA Projection (2D)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.show()