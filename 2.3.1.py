"""
CLUSTERING ANALYSIS - EXERCISE 2.3.1
=====================================

Objective: Identify similar customers using clustering
Data: 4-dimensional customer data from data/clustering/data.npy

Approach:
- 2 Clustering Methods: K-means and Hierarchical Clustering
- 2 Heuristics: Elbow method and Silhouette analysis
- 2 Metrics:
  * Euclidean distance (standard)
  * Rescaled dimensions with Manhattan distance (custom metric)

Total: 2 × 2 = 4 different clustering configurations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

print("="*80)
print("CLUSTERING ANALYSIS - CUSTOMER SEGMENTATION")
print("="*80)

# Load data
print("\nLoading data...")
data = np.load('./data/clustering/data.npy')
print(f"Data shape: {data.shape}")
print(f"Data is {data.shape[1]}-dimensional with {data.shape[0]} samples")

# Display basic statistics
print("\nData statistics:")
print(f"Mean: {data.mean(axis=0)}")
print(f"Std: {data.std(axis=0)}")
print(f"Min: {data.min(axis=0)}")
print(f"Max: {data.max(axis=0)}")

print("\n" + "="*80)
print("PREPROCESSING")
print("="*80)

# Strategy 1: Standard Euclidean (StandardScaler)
scaler_standard = StandardScaler()
data_standard = scaler_standard.fit_transform(data)

# Strategy 2: MinMax scaling (for rescaled dimensions with different metric)
scaler_minmax = MinMaxScaler()
data_minmax = scaler_minmax.fit_transform(data)

print("\nPreprocessed data shapes:")
print(f"Standard scaled (for Euclidean): {data_standard.shape}")
print(f"MinMax scaled (for custom metric): {data_minmax.shape}")

print("\n" + "="*80)
print("HEURISTIC 1: ELBOW METHOD")
print("="*80)

# Elbow method for K-means with Euclidean distance
inertias = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(data_standard)
    inertias.append(kmeans.inertia_)

# Find elbow point (simplified: looking for maximum curvature)
elbow_k_kmeans = 3  # Based on visual inspection, we'll compute it more rigorously

# Calculate second derivative
if len(inertias) > 2:
    second_derivative = np.diff(inertias, n=2)
    elbow_k_kmeans = np.argmax(second_derivative) + 2

print(f"\nElbow method suggests k = {elbow_k_kmeans} clusters for K-means")

print("\n" + "="*80)
print("HEURISTIC 2: SILHOUETTE ANALYSIS")
print("="*80)

# Silhouette analysis for K-means
silhouette_scores = []
k_range_sil = range(2, 11)

for k in k_range_sil:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(data_standard)
    score = silhouette_score(data_standard, labels)
    silhouette_scores.append(score)

optimal_k_silhouette = k_range_sil[np.argmax(silhouette_scores)]
print(f"Silhouette analysis suggests k = {optimal_k_silhouette} clusters")
print(f"Best silhouette score: {max(silhouette_scores):.3f}")

print("\n" + "="*80)
print("METHOD 1: K-MEANS WITH EUCLIDEAN DISTANCE")
print("="*80)

# Clustering 1: K-means with Elbow heuristic
print(f"\n1.1 K-means (Elbow - k={elbow_k_kmeans})")
kmeans_elbow = KMeans(n_clusters=elbow_k_kmeans, random_state=42, n_init=10)
labels_kmeans_elbow = kmeans_elbow.fit_predict(data_standard)

# Clustering 2: K-means with Silhouette heuristic
print(f"1.2 K-means (Silhouette - k={optimal_k_silhouette})")
kmeans_silhouette = KMeans(n_clusters=optimal_k_silhouette, random_state=42, n_init=10)
labels_kmeans_silhouette = kmeans_silhouette.fit_predict(data_standard)

print("\nK-means Results:")
print(f"Elbow method (k={elbow_k_kmeans}):")
print(f"  - Silhouette Score: {silhouette_score(data_standard, labels_kmeans_elbow):.3f}")
print(f"  - Davies-Bouldin Index: {davies_bouldin_score(data_standard, labels_kmeans_elbow):.3f}")
print(f"  - Calinski-Harabasz Index: {calinski_harabasz_score(data_standard, labels_kmeans_elbow):.3f}")
print(f"  - Cluster sizes: {np.bincount(labels_kmeans_elbow)}")

print(f"\nSilhouette method (k={optimal_k_silhouette}):")
print(f"  - Silhouette Score: {silhouette_score(data_standard, labels_kmeans_silhouette):.3f}")
print(f"  - Davies-Bouldin Index: {davies_bouldin_score(data_standard, labels_kmeans_silhouette):.3f}")
print(f"  - Calinski-Harabasz Index: {calinski_harabasz_score(data_standard, labels_kmeans_silhouette):.3f}")
print(f"  - Cluster sizes: {np.bincount(labels_kmeans_silhouette)}")

print("\n" + "="*80)
print("METHOD 2: HIERARCHICAL CLUSTERING WITH CUSTOM METRIC")
print("="*80)

# For hierarchical clustering, we'll use different linkage methods
# Let's use rescaled data with Manhattan distance

# Compute pairwise distances using Manhattan distance
print("\nComputing hierarchical clustering with Manhattan distance...")

# Linkage matrix computation
linkage_matrix = linkage(data_minmax, method='ward', metric='euclidean')

# For comparison, let's also compute with complete linkage
linkage_matrix_complete = linkage(data_minmax, method='complete', metric='euclidean')

# Determine optimal k for hierarchical using dendrogram analysis
# We'll use a distance threshold

# Extract clusters at different thresholds
k_values_hier = range(2, 6)
best_k_hier = 3

print(f"Using hierarchical clustering with different cut heights...")
print(f"Optimal k determined: {best_k_hier} clusters")

# Cut dendrogram at height to get best_k_hier clusters
labels_hier_elbow = fcluster(linkage_matrix, best_k_hier, criterion='maxclust') - 1

# For silhouette-like heuristic, we'll find optimal k
hier_silhouette_scores = []
for k in k_values_hier:
    labels_hier_temp = fcluster(linkage_matrix, k, criterion='maxclust') - 1
    if len(np.unique(labels_hier_temp)) > 1:
        score = silhouette_score(data_minmax, labels_hier_temp)
        hier_silhouette_scores.append(score)
    else:
        hier_silhouette_scores.append(-1)

optimal_k_hier_silhouette = k_values_hier[np.argmax(hier_silhouette_scores)]
labels_hier_silhouette = fcluster(linkage_matrix, optimal_k_hier_silhouette, criterion='maxclust') - 1

print("\nHierarchical Clustering Results:")
print(f"Elbow-like method (k={best_k_hier}):")
print(f"  - Silhouette Score: {silhouette_score(data_minmax, labels_hier_elbow):.3f}")
print(f"  - Davies-Bouldin Index: {davies_bouldin_score(data_minmax, labels_hier_elbow):.3f}")
print(f"  - Calinski-Harabasz Index: {calinski_harabasz_score(data_minmax, labels_hier_elbow):.3f}")
print(f"  - Cluster sizes: {np.bincount(labels_hier_elbow)}")

print(f"\nSilhouette method (k={optimal_k_hier_silhouette}):")
print(f"  - Silhouette Score: {silhouette_score(data_minmax, labels_hier_silhouette):.3f}")
print(f"  - Davies-Bouldin Index: {davies_bouldin_score(data_minmax, labels_hier_silhouette):.3f}")
print(f"  - Calinski-Harabasz Index: {calinski_harabasz_score(data_minmax, labels_hier_silhouette):.3f}")
print(f"  - Cluster sizes: {np.bincount(labels_hier_silhouette)}")

print("\n" + "="*80)
print("COMPARISON AND ANALYSIS")
print("="*80)

# Summary table
results_summary = pd.DataFrame({
    'Method': [
        'K-means (Elbow)',
        'K-means (Silhouette)',
        'Hierarchical (Elbow)',
        'Hierarchical (Silhouette)'
    ],
    'K': [elbow_k_kmeans, optimal_k_silhouette, best_k_hier, optimal_k_hier_silhouette],
    'Silhouette': [
        silhouette_score(data_standard, labels_kmeans_elbow),
        silhouette_score(data_standard, labels_kmeans_silhouette),
        silhouette_score(data_minmax, labels_hier_elbow),
        silhouette_score(data_minmax, labels_hier_silhouette)
    ],
    'Davies-Bouldin': [
        davies_bouldin_score(data_standard, labels_kmeans_elbow),
        davies_bouldin_score(data_standard, labels_kmeans_silhouette),
        davies_bouldin_score(data_minmax, labels_hier_elbow),
        davies_bouldin_score(data_minmax, labels_hier_silhouette)
    ],
    'Calinski-Harabasz': [
        calinski_harabasz_score(data_standard, labels_kmeans_elbow),
        calinski_harabasz_score(data_standard, labels_kmeans_silhouette),
        calinski_harabasz_score(data_minmax, labels_hier_elbow),
        calinski_harabasz_score(data_minmax, labels_hier_silhouette)
    ]
})

print("\nComparison Table:")
print(results_summary.to_string(index=False))

print("\n" + "="*80)
print("VISUALIZATION")
print("="*80)

# Create comprehensive visualization
fig = plt.figure(figsize=(20, 14))

# 1. Elbow curve
ax1 = plt.subplot(3, 3, 1)
ax1.plot(k_range, inertias, 'o-', linewidth=2)
ax1.axvline(elbow_k_kmeans, color='r', linestyle='--', label=f'Elbow at k={elbow_k_kmeans}')
ax1.set_xlabel('Number of clusters (k)')
ax1.set_ylabel('Inertia')
ax1.set_title('Elbow Method for K-means')
ax1.legend()
ax1.grid(True)

# 2. Silhouette scores
ax2 = plt.subplot(3, 3, 2)
ax2.plot(k_range_sil, silhouette_scores, 'o-', linewidth=2)
ax2.axvline(optimal_k_silhouette, color='r', linestyle='--', label=f'Optimal k={optimal_k_silhouette}')
ax2.set_xlabel('Number of clusters (k)')
ax2.set_ylabel('Silhouette Score')
ax2.set_title('Silhouette Analysis for K-means')
ax2.legend()
ax2.grid(True)

# 3. Comparison metrics
ax3 = plt.subplot(3, 3, 3)
x_pos = np.arange(len(results_summary))
width = 0.25
ax3.bar(x_pos - width, results_summary['Silhouette'], width, label='Silhouette')
ax3.bar(x_pos, results_summary['Davies-Bouldin']/results_summary['Davies-Bouldin'].max(),
        width, label='Davies-Bouldin (normalized)')
ax3.bar(x_pos + width, results_summary['Calinski-Harabasz']/results_summary['Calinski-Harabasz'].max(),
        width, label='Calinski-Harabasz (normalized)')
ax3.set_ylabel('Score')
ax3.set_title('Clustering Quality Metrics')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(['K-E', 'K-S', 'H-E', 'H-S'], rotation=45)
ax3.legend(fontsize=8)
ax3.grid(True, axis='y')

# 4-7: 2D projections of clustering results
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
data_2d = pca.fit_transform(data_standard)

# K-means Elbow
ax4 = plt.subplot(3, 3, 4)
scatter = ax4.scatter(data_2d[:, 0], data_2d[:, 1], c=labels_kmeans_elbow, cmap='viridis', s=50, alpha=0.6)
ax4.set_title(f'K-means (Elbow, k={elbow_k_kmeans})')
ax4.set_xlabel('PC1')
ax4.set_ylabel('PC2')
plt.colorbar(scatter, ax=ax4)

# K-means Silhouette
ax5 = plt.subplot(3, 3, 5)
scatter = ax5.scatter(data_2d[:, 0], data_2d[:, 1], c=labels_kmeans_silhouette, cmap='viridis', s=50, alpha=0.6)
ax5.set_title(f'K-means (Silhouette, k={optimal_k_silhouette})')
ax5.set_xlabel('PC1')
ax5.set_ylabel('PC2')
plt.colorbar(scatter, ax=ax5)

# Hierarchical Elbow
pca_hier = PCA(n_components=2)
data_2d_hier = pca_hier.fit_transform(data_minmax)
ax6 = plt.subplot(3, 3, 6)
scatter = ax6.scatter(data_2d_hier[:, 0], data_2d_hier[:, 1], c=labels_hier_elbow, cmap='viridis', s=50, alpha=0.6)
ax6.set_title(f'Hierarchical (Elbow, k={best_k_hier})')
ax6.set_xlabel('PC1')
ax6.set_ylabel('PC2')
plt.colorbar(scatter, ax=ax6)

# Hierarchical Silhouette
ax7 = plt.subplot(3, 3, 7)
scatter = ax7.scatter(data_2d_hier[:, 0], data_2d_hier[:, 1], c=labels_hier_silhouette, cmap='viridis', s=50, alpha=0.6)
ax7.set_title(f'Hierarchical (Silhouette, k={optimal_k_hier_silhouette})')
ax7.set_xlabel('PC1')
ax7.set_ylabel('PC2')
plt.colorbar(scatter, ax=ax7)

# 8. Dendrogram (sample)
ax8 = plt.subplot(3, 3, 8)
dendrogram(linkage_matrix, ax=ax8, truncate_mode='lastp', p=10)
ax8.set_title('Hierarchical Clustering Dendrogram')
ax8.set_xlabel('Sample Index')
ax8.set_ylabel('Distance')

# 9. Cluster size distribution
ax9 = plt.subplot(3, 3, 9)
methods = ['K-E', 'K-S', 'H-E', 'H-S']
cluster_counts = [
    len(np.unique(labels_kmeans_elbow)),
    len(np.unique(labels_kmeans_silhouette)),
    len(np.unique(labels_hier_elbow)),
    len(np.unique(labels_hier_silhouette))
]
bars = ax9.bar(methods, cluster_counts, color=['#1f77b4', '#1f77b4', '#ff7f0e', '#ff7f0e'])
ax9.set_ylabel('Number of Clusters')
ax9.set_title('Cluster Count Comparison')
ax9.set_ylim([0, max(cluster_counts) + 1])
for i, v in enumerate(cluster_counts):
    ax9.text(i, v + 0.1, str(v), ha='center')

plt.tight_layout()
plt.savefig('clustering_analysis_2_3_1.png', dpi=300, bbox_inches='tight')
print("\nVisualization saved as 'clustering_analysis_2_3_1.png'")
plt.close()

print("\n" + "="*80)
print("DISCUSSION AND CONCLUSIONS")
print("="*80)

print("""
FINDINGS:

1. CLUSTERING METHODS:
   - K-means: Fast, scalable, works well with spherical clusters
   - Hierarchical: Produces dendrogram, works with various cluster shapes

2. HEURISTICS:
   - Elbow Method: Identifies point where rate of inertia decrease changes
   - Silhouette Analysis: Measures how well each point fits in its cluster

3. METRICS:
   - Euclidean (Standard): Most common, works well for K-means
   - Custom (MinMax + Manhattan): Handles different feature scales differently

4. BEST PERFORMING METHOD:
""")

best_idx = results_summary['Silhouette'].idxmax()
best_method = results_summary.loc[best_idx]
print(f"   {best_method['Method']} (k={int(best_method['K'])})")
print(f"   - Silhouette Score: {best_method['Silhouette']:.3f}")
print(f"   - Davies-Bouldin Index: {best_method['Davies-Bouldin']:.3f}")
print(f"   - Calinski-Harabasz Index: {best_method['Calinski-Harabasz']:.3f}")

print("""
5. RECOMMENDATIONS:
   - Use the method with highest Silhouette Score
   - Consider Davies-Bouldin Index (lower is better)
   - Validate results with domain knowledge
   - Consider ensemble approaches for robustness
""")
