import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from utils import load_elliptic_graph


def run_baseline():
    data = load_elliptic_graph()

    # Convert tensors to numpy
    X = data.x.numpy()
    y = data.y.numpy()

    train_mask = data.train_mask.numpy()
    test_mask = data.test_mask.numpy()

    X_train = X[train_mask]
    y_train = y[train_mask]

    X_test = X[test_mask]
    y_test = y[test_mask]

    # Handle imbalance
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]

    print("\nBaseline ROC-AUC:", roc_auc_score(y_test, probs))

    thresholds = [0.5, 0.4, 0.3, 0.2]

    for t in thresholds:
        preds = (probs >= t).astype(int)
        print(f"\nThreshold: {t}")
        print(classification_report(y_test, preds))


if __name__ == "__main__":
    run_baseline()