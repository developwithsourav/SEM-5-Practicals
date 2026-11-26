"""Aim: Perform agglomerative hierarchical clustering and draw a dendrogram.
Ward linkage merges clusters with the smallest increase in within-cluster SSE.
Run: python hierarchical_clustering.py
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Public UCI Iris data bundled with scikit-learn. Species labels are NOT used.
data = load_iris()
X = StandardScaler().fit_transform(data.data)
# Clustering is transductive exploration here: use all unlabeled data.
# PCA is only a 2-D visualization, not the space used to fit clusters.
projection = PCA(n_components=2).fit_transform(X)

from scipy.cluster.hierarchy import linkage, dendrogram, cut_tree

out = Path(__file__).resolve().parent / "results" / "hierarchical_clustering"
out.mkdir(parents=True, exist_ok=True)
tree = linkage(X, method="ward", metric="euclidean")
labels = cut_tree(tree, n_clusters=3).ravel()
print("Cluster sizes:", np.bincount(labels))
print("Silhouette score:", silhouette_score(X, labels))
print("Davies-Bouldin score:", davies_bouldin_score(X, labels))
print("First ten cluster assignments:", labels[:10])
print("Cluster IDs are arbitrary, not species labels.")
print("Hierarchical clustering assigns observed samples; it has no native predict for new data.")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
dendrogram(tree, truncate_mode="lastp", p=20, ax=axes[0])
axes[0].set(xlabel="Merged groups (parentheses show size)",
            ylabel="Ward linkage distance", title="Truncated dendrogram")
axes[1].scatter(projection[:, 0], projection[:, 1], c=labels, cmap="viridis")
axes[1].set(xlabel="PC1", ylabel="PC2", title="Hierarchical clusters (k=3)")
fig.tight_layout()
fig.savefig(out / "dendrogram_and_clusters.png", dpi=150)
plt.close(fig)
print("Plots saved to:", out)
