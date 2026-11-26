"""Aim: Implement multiple linear regression.
Fits a linear combination of three predictors; multiple means three input features.
Run: python multiple_linear_regression.py
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# UCI Iris: predict petal width (column 3), never species IDs.
# Exclude petal width from the input features to avoid target leakage.
data = load_iris()
X = data.data[:, :3]  # Other three measurements
X_train, X_test, y_train, y_test = train_test_split(
    X, data.data[:, 3], test_size=0.2, random_state=42)
models = {"Multiple linear regression": make_pipeline(StandardScaler(), LinearRegression())}
cv = KFold(n_splits=5, shuffle=True, random_state=42)
out = Path(__file__).resolve().parent / "results" / "multiple_linear_regression"
out.mkdir(parents=True, exist_ok=True)

for name, model in models.items():
    scores = cross_validate(model, X_train, y_train, cv=cv,
        scoring=["neg_mean_squared_error", "r2"])
    print("\nModel:", name)
    print(f"5-fold training CV MSE: {-scores['test_neg_mean_squared_error'].mean():.4f}")
    print(f"5-fold training CV R2: {scores['test_r2'].mean():.4f}")
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"Held-out test MSE: {mean_squared_error(y_test, pred):.4f}")
    print(f"Held-out test R2: {r2_score(y_test, pred):.4f}")
    print("Intercept (transformed feature space):", model.steps[-1][1].intercept_)
    print("Coefficients (transformed feature space):", model.steps[-1][1].coef_)
    print("First five actual values:", y_test[:5])
    print("First five predictions:", np.round(pred[:5], 2))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_test, pred, alpha=0.7)
    bounds = [min(y_test.min(), pred.min()), max(y_test.max(), pred.max())]
    ax.plot(bounds, bounds, "r--", label="Perfect prediction")
    ax.set(xlabel="Actual petal width (cm)", ylabel="Predicted petal width (cm)")
    ax.legend()
    ax.set_title(name)
    fig.tight_layout()
    fig.savefig(out / (name.replace(" ", "_").lower() + ".png"), dpi=150)
    plt.close(fig)
print("Plots saved to:", out)
