"""Single-layer perceptron from scratch with NumPy; pandas loads the offline Iris CSV.
Species: 0=setosa, 1=versicolor, 2=virginica. No sklearn functions are used.
Install: python -m pip install numpy pandas
Run: python single_layer_perceptron_from_scratch.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

class SingleLayerPerceptron:
    def fit(self, X, y, epochs=250, rate=0.01):
        self.W, self.b = np.zeros((X.shape[1], 3)), np.zeros(3)
        rng = np.random.default_rng(42)
        for _ in range(epochs):
            mistakes = 0
            for i in rng.permutation(len(y)):
                predicted = int(np.argmax(X[i] @ self.W + self.b))
                actual = y[i]
                if predicted != actual:
                    self.W[:, actual] += rate * X[i]
                    self.W[:, predicted] -= rate * X[i]
                    self.b[actual] += rate
                    self.b[predicted] -= rate
                    mistakes += 1
            if not mistakes:
                break
        return self

    def scores(self, X):
        return X @ self.W + self.b  # Three class scores, not probabilities.


MODEL = SingleLayerPerceptron


def evaluate(y, scores, verbose=False):
    predicted = scores.argmax(axis=1)
    matrix = np.zeros((3, 3), dtype=int)
    np.add.at(matrix, (y, predicted), 1)
    metrics = []
    for i in range(3):
        tp = matrix[i, i]
        fn, fp = matrix[i].sum()-tp, matrix[:, i].sum()-tp
        tn = matrix.sum()-tp-fn-fp
        positive, negative = scores[y == i, i], scores[y != i, i]
        pairs = positive[:, None] - negative[None, :]
        auc = ((pairs > 0) + 0.5*(pairs == 0)).mean()
        metrics.append([tp/max(tp+fp, 1), tp/max(tp+fn, 1), tn/max(tn+fp, 1),
                        2*tp/max(2*tp+fp+fn, 1), auc])
        if verbose:
            print(f"Class {i} vs rest: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    accuracy = (predicted == y).mean()
    if verbose:
        print("Confusion matrix (setosa, versicolor, virginica):\n", matrix)
    return np.r_[accuracy, 1-accuracy, np.mean(metrics, axis=0)]


def experiment(X, y, train, test, verbose=False):
    mean, std = X[train].mean(axis=0), X[train].std(axis=0)
    std[std == 0] = 1
    model = MODEL().fit((X[train]-mean)/std, y[train])
    scores = model.scores((X[test]-mean)/std)
    return evaluate(y[test], scores, verbose), scores.argmax(axis=1)


def main():
    data = pd.read_csv(Path(__file__).with_name("iris.csv"))
    X, y = data.iloc[:, :4].to_numpy(), data["species"].to_numpy(dtype=int)
    train, test, folds = [], [], [[] for _ in range(5)]
    rng = np.random.default_rng(42)
    for label in range(3):
        indices = rng.permutation(np.flatnonzero(y == label))
        test.extend(indices[:15])  # 30% stratified holdout per class.
        train.extend(indices[15:])
        for i, index in enumerate(indices[15:]):
            folds[i % 5].append(index)
    results = []
    for i in range(5):
        training = [idx for j, fold in enumerate(folds) if j != i for idx in fold]
        results.append(experiment(X, y, training, folds[i])[0])
    names = ["Accuracy", "Error", "Macro precision", "Macro recall", "Macro specificity", "Macro F1", "Macro OvR AUC"]
    print("Training-only five-fold CV:", dict(zip(names, np.mean(results, axis=0).round(4).tolist())))
    metrics, predicted = experiment(X, y, train, test, verbose=True)
    print("Held-out metrics:", dict(zip(names, metrics.round(4).tolist())))
    print("Species order: 0=setosa, 1=versicolor, 2=virginica")
    print("First five actual labels:", y[test][:5], "Predicted:", predicted[:5])


if __name__ == "__main__":
    main()
