"""Demonstrate a stratified holdout evaluation using KNN on UCI Iris."""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)
model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
model.fit(X_train, y_train)
predicted = model.predict(X_test)
print("Training samples:", len(y_train))
print("Testing samples:", len(y_test))
print("Holdout accuracy:", accuracy_score(y_test, predicted))
print(classification_report(y_test, predicted, target_names=load_iris().target_names))
print("First five predictions:", predicted[:5])
