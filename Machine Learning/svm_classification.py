"""Aim: Build and evaluate support vector machine classification.
SVM seeks a separating margin. C controls the penalty for errors; the RBF kernel and gamma control nonlinear boundaries.
Run: python svm_classification.py
"""

import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score,
    recall_score, f1_score, roc_auc_score, ConfusionMatrixDisplay, RocCurveDisplay)
from sklearn.svm import SVC

# Public Wisconsin Diagnostic Breast Cancer data bundled with scikit-learn.
# Encode malignant as positive (1), benign as negative (0).
data = load_breast_cancer()
X, y = data.data, (data.target == 0).astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Pipeline fits scaling on training data only, including inside each CV fold.
model = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale"))
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_validate(model, X_train, y_train, cv=cv,
    scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])
print("Five-fold cross-validation on TRAINING data:")
for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
    values = scores["test_" + metric]
    print(f"  {metric}: {values.mean():.4f} (+/- {values.std():.4f})")

model.fit(X_train, y_train)
pred = model.predict(X_test)
positive_score = model.decision_function(X_test)
tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
accuracy = accuracy_score(y_test, pred)
print("\nHeld-out test evaluation (positive = malignant):")
print("Confusion matrix: rows=actual, columns=predicted; order=[benign, malignant]")
print(confusion_matrix(y_test, pred, labels=[0, 1]))
print(f"TP={tp}, TN={tn}, FP={fp}, FN={fn}")
print(f"Accuracy: {accuracy:.4f}; Error: {1 - accuracy:.4f}")
print(f"Precision: {precision_score(y_test, pred, zero_division=0):.4f}")
print(f"Recall / sensitivity: {recall_score(y_test, pred):.4f}")
print(f"Specificity: {tn / (tn + fp):.4f}")
print(f"F1-score: {f1_score(y_test, pred):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test, positive_score):.4f}")
print("\nSample predictions from held-out data:")
for actual, predicted in zip(y_test[:5], pred[:5]):
    print("Actual:", "malignant" if actual else "benign",
          "| Predicted:", "malignant" if predicted else "benign")

out = Path(__file__).resolve().parent / "results" / "svm_classification"
out.mkdir(parents=True, exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
ConfusionMatrixDisplay.from_predictions(y_test, pred,
    display_labels=["benign", "malignant"], ax=axes[0], colorbar=False)
RocCurveDisplay.from_predictions(y_test, positive_score, ax=axes[1])
fig.suptitle("Support vector machine classification")
fig.tight_layout()
fig.savefig(out / "evaluation.png", dpi=150)
plt.close(fig)

print("Plots saved to:", out)
