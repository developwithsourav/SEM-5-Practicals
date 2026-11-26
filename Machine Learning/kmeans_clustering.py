"""Aim: Cluster Iris observations using K-means with Euclidean distance.
K-means alternates nearest-centroid assignment and centroid updates.
Run: python kmeans_clustering.py
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

from sklearn.cluster import KMeans

out = Path(__file__).resolve().parent / "results" / "kmeans_clustering"
out.mkdir(parents=True, exist_ok=True)
ks = list(range(2, 9))
inertias, silhouettes = [], []
for k in ks:
    candidate = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    inertias.append(candidate.inertia_)
    silhouettes.append(silhouette_score(X, candidate.labels_))
    print(f"k={k}: inertia={inertias[-1]:.4f}, silhouette={silhouettes[-1]:.4f}")
print("Best scanned k by silhouette:", ks[int(np.argmax(silhouettes))])

# Fix k=3 for this practical's illustrative model; the scan is also reported.
model = KMeans(n_clusters=3, n_init=10, random_state=42)
labels = model.fit_predict(X)
print("Cluster sizes:", np.bincount(labels))
print("Inertia:", model.inertia_)
print("Silhouette score:", silhouette_score(X, labels))
print("Davies-Bouldin score:", davies_bouldin_score(X, labels))
print("First five cluster assignments:", model.predict(X[:5]))
print("Cluster IDs are arbitrary and are not species predictions.")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(ks, inertias, "o-")
axes[0].set(xlabel="k", ylabel="Inertia", title="Elbow inspection")
axes[1].plot(ks, silhouettes, "o-")
axes[1].set(xlabel="k", ylabel="Silhouette", title="Cluster separation")
axes[2].scatter(projection[:, 0], projection[:, 1], c=labels, cmap="viridis")
axes[2].set(xlabel="PC1", ylabel="PC2", title="K-means clusters (k=3)")
fig.tight_layout()
fig.savefig(out / "clusters_and_k_selection.png", dpi=150)
plt.close(fig)
print("Plots saved to:", out)
