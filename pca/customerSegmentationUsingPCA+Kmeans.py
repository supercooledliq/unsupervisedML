# Convert transactions to binary matrix

# Apply PCA(Principal component analysis) to reduce dimensionality(Takes many features (like age, income, frequency) 
# Combines them into fewer components that still retain the most variance (important information)

# Use KMeans to cluster similar basket behaviors

# Apply Apriori within each cluster for more targeted recommendations

# Output: More personalized bundle rules like: "Customers in Cluster 1 who buy A → likely to buy B and C"


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Load Data
df = pd.read_csv("Amazon Sale Report.csv")

# --- Step 1: Prepare Data for Segmentation ---
segmentation_df = df[['Order ID', 'Qty', 'Amount']].dropna()
segmentation_df = segmentation_df.groupby('Order ID').agg({'Qty': 'sum','Amount': 'sum'}).reset_index()

# --- Step 2: Scale, PCA & K-Means ---
scaler = StandardScaler()
scaled_data = scaler.fit_transform(segmentation_df[['Qty', 'Amount']])
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(pca_result)
segmentation_df['Cluster'] = clusters
segmentation_df['PCA1'] = pca_result[:, 0]
segmentation_df['PCA2'] = pca_result[:, 1]

# --- Step 3: Visualize Clusters ---
plt.figure(figsize=(10,6))
sns.scatterplot(data=segmentation_df, x='PCA1', y='PCA2', hue='Cluster',
palette='Set2', s=100)
plt.title('E-commerce Customer Segmentation via K-Means', fontsize=14)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.grid(True)
plt.legend(title="Customer Segments")
plt.show()