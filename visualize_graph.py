import torch
import networkx as nx
import matplotlib.pyplot as plt
from utils import load_elliptic_graph

# Load graph
data = load_elliptic_graph()

# Convert PyTorch Geometric graph to NetworkX
edge_index = data.edge_index.numpy()
G = nx.Graph()

# Add edges
for i in range(edge_index.shape[1]):
    G.add_edge(edge_index[0][i], edge_index[1][i])

# Sample small subgraph
sample_nodes = list(G.nodes())[:200]
subgraph = G.subgraph(sample_nodes)

# Node colors
colors = []
colors = []
for node in subgraph.nodes():
    if data.y[node].item() == 1:
        colors.append("red")      # Fraud
    else:
        colors.append("skyblue")  # Legit
# Draw graph

pos = nx.spring_layout(subgraph, k=0.15)

plt.figure(figsize=(10,10))
nx.draw(
    subgraph,
    pos,
    node_color=colors,
    node_size=40,
    edge_color="gray",
    alpha=0.8,
    with_labels=False
)

plt.title("Bitcoin Transaction Graph (Fraud vs Legit)")
plt.savefig("fraud_graph.png", dpi=300)
plt.show()