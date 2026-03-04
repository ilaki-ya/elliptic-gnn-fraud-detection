import torch
import torch.nn.functional as F
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

from model import GraphSAGE


# -----------------------------
# Configuration
# -----------------------------
THRESHOLD = 0.4
MODEL_PATH = "graphsage_model.pth"
INPUT_DIM = 165


# -----------------------------
# Define request schema
# -----------------------------
class Transaction(BaseModel):
    features: list[float]


# -----------------------------
# Initialize FastAPI app
# -----------------------------
app = FastAPI()

# Load model once at startup
model = GraphSAGE(in_channels=INPUT_DIM, hidden_channels=128)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()


@app.get("/")
def home():
    return {"message": "GraphSAGE Fraud Detection API is running."}


@app.post("/predict")
def predict(transaction: Transaction):

    if len(transaction.features) != INPUT_DIM:
        return {
            "error": f"Expected {INPUT_DIM} features, received {len(transaction.features)}"
        }

    # Convert to tensor
    x = torch.tensor([transaction.features], dtype=torch.float)

    # Dummy edge_index (single node, no neighbors)
    edge_index = torch.empty((2, 0), dtype=torch.long)

    with torch.no_grad():
        output = model(x, edge_index)
        probs = F.softmax(output, dim=1)[0][1].item()

    prediction = 1 if probs >= THRESHOLD else 0

    return {
        "fraud_probability": round(probs, 4),
        "threshold": THRESHOLD,
        "prediction": "Fraud" if prediction == 1 else "Not Fraud"
    }