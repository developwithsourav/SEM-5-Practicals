"""Demonstrate five-fold stratified cross-validation on UCI Iris."""
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

X, y = load_iris(return_X_y=True)
model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# Scaling is fitted independently inside every fold to avoid leakage.
scores = cross_val_score(model, X, y, cv=folds, scoring="accuracy")
for i, score in enumerate(scores, start=1):
    print(f"Fold {i}: accuracy = {score:.4f}")
print(f"Mean accuracy: {scores.mean():.4f}")
print(f"Standard deviation: {scores.std():.4f}")
