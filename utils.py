# import pandas as pd
# import torch
# from torch_geometric.data import Data


# def load_elliptic_graph():

#     features_df = pd.read_csv("data/elliptic_txs_features.csv", header=None)
#     edges_df = pd.read_csv("data/elliptic_txs_edgelist.csv")
#     classes_df = pd.read_csv("data/elliptic_txs_classes.csv")

#     features_df = features_df.rename(columns={0: "txId", 1: "time_step"})
#     classes_df.columns = ["txId", "class"]

#     df = features_df.merge(classes_df, on="txId")

#     # Remove unknown
#     df = df[df["class"] != "unknown"]

#     # Map labels
#     df["class"] = df["class"].map({"1": 1, "2": 0})

#     print("Final dataset shape:", df.shape)
#     print("Class distribution:")
#     print(df["class"].value_counts())

#     # Create mapping from txId to index
#     txid_to_index = {txid: idx for idx, txid in enumerate(df["txId"])}

#     # Keep only edges where both nodes are labeled
#     edges_df = edges_df[
#         edges_df["txId1"].isin(txid_to_index) &
#         edges_df["txId2"].isin(txid_to_index)
#     ]

#     # Convert edges to index-based
#     edge_index = torch.tensor([
#         [txid_to_index[row["txId1"]], txid_to_index[row["txId2"]]]
#         for _, row in edges_df.iterrows()
#     ], dtype=torch.long).t().contiguous()

#     # Node features
#     feature_cols = df.columns[2:-1]  # exclude txId, time_step, class
#     x = torch.tensor(df[feature_cols].values, dtype=torch.float)

#     # Labels
#     y = torch.tensor(df["class"].values, dtype=torch.long)

#     data = Data(x=x, edge_index=edge_index, y=y)

#     print(data)

#     return data


import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split


def load_elliptic_graph():

    # -------------------------
    # 1. Load CSV files
    # -------------------------
    features_df = pd.read_csv("data/elliptic_txs_features.csv", header=None)
    edges_df = pd.read_csv("data/elliptic_txs_edgelist.csv")
    classes_df = pd.read_csv("data/elliptic_txs_classes.csv")

    # Rename columns
    features_df = features_df.rename(columns={0: "txId", 1: "time_step"})
    classes_df.columns = ["txId", "class"]

    # -------------------------
    # 2. Merge features + labels
    # -------------------------
    df = features_df.merge(classes_df, on="txId")

    # Remove unknown labels
    df = df[df["class"] != "unknown"].copy()

    # Convert labels:
    # illicit (1) -> 1
    # licit (2) -> 0
    df["class"] = df["class"].map({"1": 1, "2": 0})

    print("Final dataset shape:", df.shape)
    print("Class distribution:")
    print(df["class"].value_counts())

    # -------------------------
    # 3. Map txId -> node index
    # -------------------------
    txid_to_index = {
        txid: idx for idx, txid in enumerate(df["txId"].values)
    }

    # -------------------------
    # 4. Filter edges
    # Keep only edges where BOTH nodes exist in labeled dataset
    # -------------------------
    edges_df = edges_df[
        edges_df["txId1"].isin(txid_to_index) &
        edges_df["txId2"].isin(txid_to_index)
    ]

    # Convert to index-based edges
    edge_index = torch.tensor(
        [
            [
                txid_to_index[row["txId1"]],
                txid_to_index[row["txId2"]],
            ]
            for _, row in edges_df.iterrows()
        ],
        dtype=torch.long
    ).t().contiguous()

    # -------------------------
    # 5. Node features
    # Remove txId, time_step, class
    # -------------------------
    feature_cols = df.columns[2:-1]
    x = torch.tensor(df[feature_cols].values, dtype=torch.float)

    # -------------------------
    # 6. Labels
    # -------------------------
    y = torch.tensor(df["class"].values, dtype=torch.long)

    # -------------------------
    # 7. Create PyG Data object
    # -------------------------
    data = Data(x=x, edge_index=edge_index, y=y)

    print(data)

    # -------------------------
    # 8. Train/Test Split (Stratified)
    # -------------------------
    indices = torch.arange(len(y))

    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    data.train_mask = torch.zeros(len(y), dtype=torch.bool)
    data.test_mask = torch.zeros(len(y), dtype=torch.bool)

    data.train_mask[train_idx] = True
    data.test_mask[test_idx] = True

    print("Train nodes:", data.train_mask.sum().item())
    print("Test nodes:", data.test_mask.sum().item())

    return data
