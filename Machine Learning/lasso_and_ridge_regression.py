"""Aim: Implement lasso and ridge regression.
Lasso adds an L1 penalty and can make coefficients zero; Ridge adds L2 shrinkage. Compare both against an unpenalized baseline using identical splits. Alphas are illustrative, not tuned optima.
Run: python lasso_and_ridge_regression.py
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score

# Public diabetes dataset bundled with scikit-learn; target is progression
# one year after baseline. scaled=False preserves original feature units.
data = load_diabetes(scaled=False)
X = data.data
X_train, X_test, y_train, y_test = train_test_split(
    X, data.target, test_size=0.2, random_state=42)
models = {"Linear baseline": make_pipeline(StandardScaler(), LinearRegression()), "Lasso alpha 1": make_pipeline(StandardScaler(), Lasso(alpha=1.0, max_iter=10000)), "Ridge alpha 10": make_pipeline(StandardScaler(), Ridge(alpha=10.0))}
cv = KFold(n_splits=5, shuffle=True, random_state=42)
out = Path(__file__).resolve().parent / "results" / "lasso_and_ridge_regression"
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
    ax.set(xlabel="Actual progression", ylabel="Predicted progression")
    ax.legend()
    ax.set_title(name)
    fig.tight_layout()
    fig.savefig(out / (name.replace(" ", "_").lower() + ".png"), dpi=150)
    plt.close(fig)
print("Plots saved to:", out)
