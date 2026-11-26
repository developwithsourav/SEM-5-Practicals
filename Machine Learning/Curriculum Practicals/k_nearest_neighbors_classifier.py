"""K-nearest neighbors classification on UCI Iris: setosa, versicolor, virginica."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score,
    recall_score, f1_score, roc_auc_score, ConfusionMatrixDisplay, RocCurveDisplay)
from sklearn.neighbors import KNeighborsClassifier

data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.3, random_state=42, stratify=data.target)
model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_validate(model, X_train, y_train, cv=cv,
    scoring=["accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc_ovr"])
print("Training-only five-fold CV:")
for key in ["accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc_ovr"]:
    print(key, round(scores["test_" + key].mean(), 4))
model.fit(X_train, y_train)
pred = model.predict(X_test)
probability = model.predict_proba(X_test)
matrix = confusion_matrix(y_test, pred, labels=[0, 1, 2])
accuracy = accuracy_score(y_test, pred)
print("\nHeld-out test confusion matrix; order:", data.target_names)
print(matrix)
print("Accuracy:", accuracy, "Error:", 1 - accuracy)
for i, name in enumerate(data.target_names):
    tp = matrix[i, i]
    fn, fp = matrix[i].sum() - tp, matrix[:, i].sum() - tp
    tn = matrix.sum() - tp - fn - fp
    print(f"{name} vs rest: TP={tp}, TN={tn}, FP={fp}, FN={fn}, specificity={tn/(tn+fp):.4f}")
print("Macro precision:", precision_score(y_test, pred, average="macro", zero_division=0))
print("Macro recall:", recall_score(y_test, pred, average="macro"))
print("Macro F1:", f1_score(y_test, pred, average="macro"))
print("Macro one-vs-rest AUC:", roc_auc_score(y_test, probability, multi_class="ovr"))
for actual, predicted in zip(y_test[:5], pred[:5]):
    print("Actual:", data.target_names[actual], "Predicted:", data.target_names[predicted])
out = Path(__file__).resolve().parent / "results" / "k_nearest_neighbors_classifier"
out.mkdir(parents=True, exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=data.target_names,
                                       ax=axes[0], colorbar=False)
for i, name in enumerate(data.target_names):
    RocCurveDisplay.from_predictions((y_test == i).astype(int), probability[:, i],
                                    name=name, ax=axes[1])
fig.suptitle("K-nearest neighbors")
fig.tight_layout()
fig.savefig(out / "evaluation.png", dpi=150)
plt.close(fig)

print("Plots saved to:", out)
