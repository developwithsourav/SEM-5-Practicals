"""Aim: Implement polynomial regression.
Adds BMI squared to model curvature; still linear in coefficients. Degree is fixed in advance, not selected on the test set.
Run: python polynomial_regression.py
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Public diabetes dataset bundled with scikit-learn; target is progression
# one year after baseline. scaled=False preserves original feature units.
data = load_diabetes(scaled=False)
X = data.data[:, [2]]  # BMI only
X_train, X_test, y_train, y_test = train_test_split(
    X, data.target, test_size=0.2, random_state=42)
models = {"Polynomial regression degree 2": make_pipeline(PolynomialFeatures(degree=2, include_bias=False), StandardScaler(), LinearRegression())}
cv = KFold(n_splits=5, shuffle=True, random_state=42)
out = Path(__file__).resolve().parent / "results" / "polynomial_regression"
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
    ax.scatter(X_test[:, 0], y_test, label="Actual test data", alpha=0.7)
    grid = np.linspace(X[:, 0].min(), X[:, 0].max(), 200).reshape(-1, 1)
    ax.plot(grid[:, 0], model.predict(grid), color="red", label="Fitted model")
    ax.set(xlabel="BMI", ylabel="Disease progression")
    ax.legend()
    ax.set_title(name)
    fig.tight_layout()
    fig.savefig(out / (name.replace(" ", "_").lower() + ".png"), dpi=150)
    plt.close(fig)
print("Plots saved to:", out)
