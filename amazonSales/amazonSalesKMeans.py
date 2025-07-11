# Clustering Techniques
# K-Means Clustering Example
# This script demonstrates how to perform K-Means clustering on a dataset  
#group customers based on:
# how much they spend
#how often they shop
#target each group with unique offers

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load data
df = pd.read_csv("customer_purchase.csv")
X = df[["total_spent", "purchase_frequency"]]
# Apply KMeans
kmeans = KMeans(n_clusters=3)
df['cluster'] = kmeans.fit_predict(X)

# Scatter Plot
plt.scatter(df['total_spent'], df['purchase_frequency'], c=df['cluster'], cmap='viridis')
plt.xlabel("Total Spent")
plt.ylabel("Purchase Frequency")
plt.title("Customer Segments")
plt.show()

# #bargraph
# # Create spending bins
# df['spending_range'] = pd.cut(df['total_spent'], bins=[0, 500, 1000, 2000, 3000, 5000], labels=['0–500', '500–1000', '1000–2000', '2000–3000', '3000–5000'])
# grouped = df.groupby('spending_range')['purchase_frequency'].sum()
# plt.bar(grouped.index, grouped.values, color='skyblue')

# # Define frequency bins
# df['frequency_range'] = pd.cut(df['purchase_frequency'], bins=[0, 100, 300, 600, 1000, 2000, float('inf')],labels=['0–100', '101–300', '301–600', '601–1000', '1001–2000', '2000+'])

# grouped = df.groupby(['spending_range', 'frequency_range']).size().reset_index(name='customer_count')

# plt.figure(figsize=(12, 6))
# sns.barplot(data=grouped, x='spending_range', y='customer_count', hue='frequency_range', palette='viridis')

# # Add number on top of each bar
# for i in range(len(grouped)):
#     x = i % len(grouped['spending_range'].unique())
#     y = grouped['customer_count'][i]
#     plt.text(x=x, y=y + 0.5, s=str(y), ha='center', fontsize=9)

# plt.title("Customer Segments by Spending and Frequency")
# plt.xlabel("Total Spending Range")
# plt.ylabel("Number of Customers")
# plt.legend(title="Purchase Frequency Range")
# plt.tight_layout()
# plt.show()