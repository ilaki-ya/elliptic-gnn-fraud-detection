import torch
import torch.nn as nn
from sklearn.metrics import classification_report, roc_auc_score
from utils import load_elliptic_graph
from model import GraphSAGE
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import numpy as np


def train():
    data = load_elliptic_graph()

    # model = GraphSAGE(data.num_features, 64)
    model = GraphSAGE(data.num_features, hidden_channels=128)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.0005,weight_decay=5e-4)
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Handle imbalance
    num_normal = (data.y == 0).sum().item()
    num_fraud = (data.y == 1).sum().item()
    weight_fraud = num_normal / num_fraud

    weights = torch.tensor([1.0, weight_fraud])
    criterion = nn.CrossEntropyLoss(weight=weights)

    for epoch in range(50):
        model.train()
        optimizer.zero_grad()

        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])

        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    return model, data



# def evaluate(model, data):
#     model.eval()

#     with torch.no_grad():
#         out = model(data.x, data.edge_index)

#         test_out = out[data.test_mask]
#         test_labels = data.y[data.test_mask]

#         probs = torch.softmax(test_out, dim=1)[:, 1].numpy()
#         y_true = test_labels.numpy()

#         print("\nROC-AUC:", roc_auc_score(y_true, probs))

#         thresholds = [0.5, 0.4, 0.3, 0.2]

#         for t in thresholds:
#             preds = (probs >= t).astype(int)

#             print(f"\nThreshold: {t}")
#             print(classification_report(y_true, preds))


def evaluate(model, data):
    model.eval()

    with torch.no_grad():
        out = model(data.x, data.edge_index)

        test_out = out[data.test_mask]
        test_labels = data.y[data.test_mask]

        probs = torch.softmax(test_out, dim=1)[:, 1].numpy()
        y_true = test_labels.numpy()

        print("\n==============================")
        print("ROC-AUC:", roc_auc_score(y_true, probs))
        print("==============================")

        thresholds = [0.5, 0.4, 0.3, 0.2]

        for t in thresholds:
            preds = (probs >= t).astype(int)

            print(f"\n🔎 Threshold: {t}")
            print(classification_report(y_true, preds))

            cm = confusion_matrix(y_true, preds)
            tn, fp, fn, tp = cm.ravel()

            print("Confusion Matrix:")
            print(cm)

            print(f"TN (Correct Normal): {tn}")
            print(f"FP (False Alarm): {fp}")
            print(f"FN (Missed Fraud): {fn}")
            print(f"TP (Detected Fraud): {tp}")

if __name__ == "__main__":
    model, data = train()
    torch.save(model.state_dict(), "graphsage_model.pth")
    print("Model saved successfully.")
    evaluate(model, data)