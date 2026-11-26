"""Backward feature elimination using training-only cross-validation, followed by held-out evaluation.
UCI Iris has four features, so exhaustive subset search is small (15 subsets).
"""
from itertools import combinations
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.3, random_state=42, stratify=data.target)
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def make_model():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))

def score(columns):
    return cross_val_score(make_model(), X_train[:, columns], y_train,
                           cv=folds, scoring="accuracy").mean()


selected = list(range(X_train.shape[1]))
best_score = score(selected)
print("Start with all features, CV accuracy:", best_score)
while len(selected) > 1:
    candidates = [(score([j for j in selected if j != feature]), feature)
                  for feature in selected]
    value, feature = max(candidates, key=lambda item: item[0])
    if value < best_score - 1e-12:
        break  # Stop when every removal reduces CV accuracy.
    selected.remove(feature)
    best_score = value
    print("Removed:", data.feature_names[feature], "CV accuracy:", best_score)

print("Selected features:", [data.feature_names[i] for i in selected])
print("Selection CV accuracy (used to choose features):", best_score)
model = make_model().fit(X_train[:, selected], y_train)
print("Independent held-out accuracy:", accuracy_score(y_test, model.predict(X_test[:, selected])))
