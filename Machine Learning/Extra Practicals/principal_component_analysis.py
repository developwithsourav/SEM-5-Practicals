"""Reduce UCI Iris to two principal components and evaluate a KNN model."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.3, random_state=42, stratify=data.target)
model = make_pipeline(StandardScaler(), PCA(n_components=2), KNeighborsClassifier(5))
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print("Training CV accuracy:", cross_val_score(model, X_train, y_train, cv=folds).mean())
model.fit(X_train, y_train)
print("Held-out accuracy:", accuracy_score(y_test, model.predict(X_test)))
print("Explained variance ratios:", model[1].explained_variance_ratio_)
print("Total explained variance:", model[1].explained_variance_ratio_.sum())
# Visualize test samples using scaling and PCA learned from training only.
points = model[:-1].transform(X_test)
fig, ax = plt.subplots()
for label, name in enumerate(data.target_names):
    ax.scatter(points[y_test == label, 0], points[y_test == label, 1], label=name)
ax.set(xlabel="Principal component 1", ylabel="Principal component 2", title="PCA of Iris (test data)")
ax.legend()
fig.tight_layout()
out = Path(__file__).resolve().parent / "results"
out.mkdir(exist_ok=True)
fig.savefig(out / "principal_component_analysis.png", dpi=150)
plt.close(fig)
print("Plot saved to:", out / "principal_component_analysis.png")
