"""Multiclass confusion matrix with one-vs-rest counts for each Iris class."""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score

data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.3, random_state=42, stratify=data.target)
model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
model.fit(X_train, y_train)
prediction = model.predict(X_test)
matrix = confusion_matrix(y_test, prediction, labels=[0, 1, 2])
print("Rows = actual, columns = predicted; class order:", data.target_names)
print(matrix)
def divide(a, b):
    return a / b if b else 0.0
for i, name in enumerate(data.target_names):
    tp = matrix[i, i]
    fn = matrix[i, :].sum() - tp
    fp = matrix[:, i].sum() - tp
    tn = matrix.sum() - tp - fn - fp
    accuracy = divide(tp + tn, matrix.sum())
    print(f"\n{name} versus all other classes: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print(f"Accuracy={accuracy:.4f}, Error={1-accuracy:.4f}")
    print(f"Precision={divide(tp, tp+fp):.4f}, Recall={divide(tp, tp+fn):.4f}")
    print(f"Specificity={divide(tn, tn+fp):.4f}, F1={divide(2*tp, 2*tp+fp+fn):.4f}")
print("\nOverall multiclass accuracy:", matrix.trace() / matrix.sum())
print("Macro one-vs-rest AUC:", roc_auc_score(
    y_test, model.predict_proba(X_test), multi_class="ovr", average="macro"))
