# Graph Neural Network for Fraud Detection

This project implements a **Graph Neural Network (GraphSAGE)** for detecting fraudulent Bitcoin transactions using the **Elliptic dataset**.

## Problem

Financial fraud often occurs in **transaction networks** where suspicious accounts interact with other suspicious accounts. Traditional machine learning models struggle to capture these relational patterns.

Graph Neural Networks can leverage **graph structure and node features** to detect fraud more effectively.

---

## Dataset

Elliptic Bitcoin Transaction Dataset

- Nodes: Transactions
- Edges: Transaction flows
- Features: 165 transaction features
- Classes:
  - 0 → Legitimate
  - 1 → Illicit

---

## Model

Graph Neural Network:

- GraphSAGE
- PyTorch Geometric

Metrics achieved:

- ROC-AUC ≈ **0.96**
- Fraud Recall ≈ **95%**

---

## System Architecture
Elliptic Dataset
      │
      ▼
Data Preprocessing
(Node Features + Edge List)
      │
      ▼
Graph Construction
(PyTorch Geometric Graph)
      │
      ▼
GraphSAGE GNN Model
      │
      ▼
Fraud Prediction
      │
      ▼
FastAPI Inference API
      │
      ▼
Docker Container Deployment


---

## API Example

Endpoint:

POST /predict

Example response:

{
 "fraud_probability": 0.69,
 "threshold": 0.4,
 "prediction": "Fraud"
}

---

## Visualization

The transaction network can be visualized using NetworkX to observe clusters of connected transactions.

---

## Deployment

Run using Docker:

docker build -t elliptic-gnn-fraud .
docker run -p 8000:8000 elliptic-gnn-fraud

Swagger UI:

http://localhost:8000/docs

---

## Technologies Used

- Python
- PyTorch Geometric
- GraphSAGE
- FastAPI
- Docker
- NetworkX
- Matplotlib