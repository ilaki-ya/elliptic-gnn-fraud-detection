# Graph Neural Network for Fraud Detection

Graph Neural Network (GraphSAGE) model for detecting fraudulent Bitcoin transactions using the **Elliptic dataset**.

---

# Problem

Financial fraud often occurs in **transaction networks** where suspicious accounts interact with other suspicious accounts.

Traditional machine learning models treat transactions **independently**, missing relational patterns.

Graph Neural Networks use **graph connectivity + node features** to detect fraud more effectively.

---

# Dataset

Elliptic Bitcoin Transaction Dataset

| Property | Value                   |
| -------- | ----------------------- |
| Nodes    | ~203k transactions      |
| Edges    | ~234k transaction links |
| Features | 165 features            |
| Classes  | Legitimate / Illicit    |

Class mapping:

```
0 → Legitimate  
1 → Fraud
```

---

# Model

Graph Neural Network:

* GraphSAGE
* PyTorch Geometric

Results:

| Metric       | Value |
| ------------ | ----- |
| ROC-AUC      | 0.96  |
| Fraud Recall | ~95%  |
| Accuracy     | ~83%  |

---

# System Architecture

```
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
```

---

# API Example

Endpoint:

```
POST /predict
```

Example response:

```json
{
 "fraud_probability": 0.69,
 "threshold": 0.4,
 "prediction": "Fraud"
}
```

Swagger UI:

```
http://localhost:8000/docs
```

---

# Visualization

The transaction network can be visualized using **NetworkX** to observe clusters of connected transactions.

---

# Deployment

Build Docker image:

```
docker build -t elliptic-gnn-fraud .
```

Run container:

```
docker run -p 8000:8000 elliptic-gnn-fraud
```

API available at:

```
http://localhost:8000/docs
```

---

# Project Structure

```
elliptic_gnn/
│
├── api.py
├── train.py
├── model.py
├── utils.py
├── baseline.py
├── requirements.txt
├── Dockerfile
├── graphsage_model.pth
└── data/
```

---

# Technologies Used

* Python
* PyTorch
* PyTorch Geometric
* GraphSAGE
* FastAPI
* Docker
* NetworkX
* Matplotlib
