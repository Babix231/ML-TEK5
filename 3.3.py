"""
BREAST CANCER CLUSTERING ANALYSIS BASED ON SIZE
============================================

This analysis focuses on clustering breast cancer data based on size-related features
(radius, perimeter, area) and analyzing how these clusters correlate with other
geometric and texture features.

Size Features used for clustering:
- Radius (mean, SE, worst)
- Perimeter (mean, SE, worst)
- Area (mean, SE, worst)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("="*80)
print("STEP 1: LOADING AND PREPROCESSING DATA")
print("="*80)

# Load the dataset
df = pd.read_csv('breast_cancer_data.csv')

# Define size-related features
size_features = [col for col in df.columns if any(x in col for x in ['radius', 'perimeter', 'area'])]
other_features = [col for col in df.columns if col not in size_features 
                 and col not in ['id', 'diagnosis', 'Unnamed: 32']]

# Prepare features
X_size = df[size_features].copy()
X_other = df[other_features].copy()
diagnosis = df['diagnosis'].map({'M': 1, 'B': 0})

print("\nSize features used for clustering:")
for feature in size_features:
    print(f"- {feature}")

print(f"\nShape of size features data: {X_size.shape}")

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X_size_imputed = imputer.fit_transform(X_size)
X_other_imputed = imputer.fit_transform(X_other)

# Scale the features
scaler = StandardScaler()
X_size_scaled = scaler.fit_transform(X_size_imputed)
X_other_scaled = scaler.fit_transform(X_other_imputed)

print("\n" + "="*80)
print("STEP 2: DETERMINING OPTIMAL NUMBER OF CLUSTERS")
print("="*80)

# Elbow method and silhouette analysis
inertias = []
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_size_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_size_scaled, kmeans.labels_))

# Plot elbow curve and silhouette scores
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Elbow curve
ax1.plot(k_range, inertias, 'o-')
ax1.set_xlabel('Number of Clusters (k)')
ax1.set_ylabel('Inertia')
ax1.set_title('Elbow Method')

# Silhouette scores
ax2.plot(k_range, silhouette_scores, 'o-')
ax2.set_xlabel('Number of Clusters (k)')
ax2.set_ylabel('Silhouette Score')
ax2.set_title('Silhouette Analysis')

plt.tight_layout()
plt.savefig('size_clustering_optimization.png', dpi=300, bbox_inches='tight')
plt.close()

# Select optimal k based on silhouette score
optimal_k = k_range[np.argmax(silhouette_scores)]
print(f"\nOptimal number of clusters (k) based on silhouette score: {optimal_k}")

print("\n" + "="*80)
print("STEP 3: PERFORMING CLUSTERING")
print("="*80)

# Perform K-means clustering with optimal k
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
cluster_labels = kmeans.fit_predict(X_size_scaled)

# Calculate feature importance for size-based clustering
size_importance = pd.DataFrame({
    'Feature': size_features,
    'Importance': np.abs(kmeans.cluster_centers_).mean(axis=0)
}).sort_values('Importance', ascending=False)

# Analyze correlation between clusters and other features
cluster_df = pd.DataFrame(X_other_scaled, columns=other_features)
cluster_df['Cluster'] = cluster_labels
cluster_df['Diagnosis'] = diagnosis

# Calculate mean values of other features for each cluster
cluster_means = cluster_df.groupby('Cluster')[other_features].mean()

print("\nCluster Sizes:")
print(pd.Series(cluster_labels).value_counts().sort_index())

print("\nTop Size Features Importance:")
print(size_importance)

print("\nMean values of other features by cluster:")
print(cluster_means)

# Correlation between size clusters and diagnosis
diagnosis_correlation = pd.crosstab(cluster_labels, diagnosis)
print("\nSize Clusters vs Diagnosis:")
print(diagnosis_correlation)

# Create visualizations
plt.figure(figsize=(20, 15))

# PCA for size features
pca_size = PCA(n_components=2)
X_size_pca = pca_size.fit_transform(X_size_scaled)

# Plot clusters based on size
plt.subplot(2, 2, 1)
scatter = plt.scatter(X_size_pca[:, 0], X_size_pca[:, 1], c=cluster_labels, cmap='viridis')
plt.title('Clusters Based on Size Features', fontsize=12)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.colorbar(scatter, label='Cluster')

# Plot diagnosis distribution
plt.subplot(2, 2, 2)
scatter = plt.scatter(X_size_pca[:, 0], X_size_pca[:, 1], c=diagnosis, cmap='Set1')
plt.title('Actual Diagnosis', fontsize=12)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.colorbar(scatter, label='Diagnosis (0=Benign, 1=Malignant)')

# Plot size feature importance
plt.subplot(2, 2, 3)
sns.barplot(data=size_importance, x='Importance', y='Feature')
plt.title('Size Features Importance', fontsize=12)
plt.xlabel('Importance Score')

# Plot cluster characteristics heatmap
plt.subplot(2, 2, 4)
sns.heatmap(cluster_means, cmap='RdYlBu_r', center=0)
plt.title('Cluster Characteristics\n(Other Features)', fontsize=12)
plt.xlabel('Features')
plt.ylabel('Cluster')

plt.tight_layout()
plt.savefig('size_clustering_results.png', dpi=300, bbox_inches='tight')
plt.close()

# Create correlation matrix visualization
plt.figure(figsize=(15, 12))
# Create DataFrame with proper column names
size_df = pd.DataFrame(X_size_scaled, columns=size_features)
other_df = pd.DataFrame(X_other_scaled, columns=other_features)
# Concatenate the DataFrames
combined_df = pd.concat([size_df, other_df], axis=1)
# Calculate correlation matrix
correlation_matrix = combined_df.corr()
# Create heatmap
sns.heatmap(correlation_matrix, cmap='RdYlBu_r', center=0)
plt.title('Correlation Matrix: Size vs Other Features', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('size_correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nAnalysis complete! Check the generated visualization files:")
print("1. size_clustering_optimization.png - Elbow curve and silhouette analysis")
print("2. size_clustering_results.png - Main clustering results and analysis")
print("3. size_correlation_matrix.png - Correlation matrix between size and other features")
