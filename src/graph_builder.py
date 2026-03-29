"""
Graph construction for GNN modeling.

Builds customer similarity graphs based on:
- Categorical feature similarity (job, education, marital)
- Numerical feature distance (age, balance)
- Contact timing similarity (day, month, duration buckets)

Uses k-NN to connect similar customers.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from src.config import repo_root
from src.features import prepare_data, ID_COL, TARGET


def _compute_categorical_similarity(
    data: pd.DataFrame,
    columns: list[str],
) -> np.ndarray:
    """Compute Jaccard similarity for categorical features."""
    # One-hot encode categorical features
    encoded = pd.get_dummies(data[columns], prefix_sep='__')
    
    # Jaccard similarity: intersection over union
    # For one-hot vectors, this simplifies to dot product / (2 - dot product)
    X = encoded.values.astype(np.float32)
    
    # Normalize: each row represents features present
    row_sums = X.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    
    return X


def _compute_numerical_features(
    data: pd.DataFrame,
    columns: list[str],
) -> np.ndarray:
    """Extract and normalize numerical features."""
    X = data[columns].fillna(0).values.astype(np.float32)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled


def build_customer_graph(
    data_dir: Path,
    feature_set: str = "blend_buckets",
    k_neighbors: int = 15,
    output_path: Path | None = None,
    include_test: bool = True,
) -> dict[str, Any]:
    """
    Build customer similarity graph using k-NN.
    
    Args:
        data_dir: Path to data directory containing train.csv and test.csv
        feature_set: Feature set to use (from src.features)
        k_neighbors: Number of neighbors to connect
        output_path: Optional path to save graph structure
        include_test: Whether to include test set in graph
        
    Returns:
        Dictionary containing:
            - edge_index: torch.LongTensor of shape [2, num_edges]
            - train_mask: boolean mask for training nodes
            - test_mask: boolean mask for test nodes
            - node_features: torch.FloatTensor of shape [num_nodes, num_features]
            - labels: torch.LongTensor of training labels
            - train_ids: original train IDs
            - test_ids: original test IDs
    """
    print(f"Loading data with feature_set={feature_set}...")
    prepared = prepare_data(
        data_dir,
        feature_set=feature_set,
        load_test=include_test,
    )
    
    # Combine train and test for unified graph
    train_df = prepared.x.copy()
    train_df[TARGET] = prepared.y
    train_df[ID_COL] = prepared.train_raw[ID_COL].values
    
    if include_test and prepared.x_test is not None:
        test_df = prepared.x_test.copy()
        test_df[TARGET] = -1  # Placeholder for test
        test_df[ID_COL] = prepared.test_raw[ID_COL].values
        combined_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
        num_train = len(train_df)
        num_test = len(test_df)
    else:
        combined_df = train_df
        num_train = len(train_df)
        num_test = 0
    
    print(f"Total nodes: {len(combined_df)} (train={num_train}, test={num_test})")
    
    # Define feature groups for similarity
    categorical_cols = ['job', 'education', 'marital', 'contact', 'month']
    numerical_cols = ['age', 'balance', 'duration', 'campaign', 'previous', 'pdays', 'day']
    
    # Filter to available columns
    categorical_cols = [c for c in categorical_cols if c in combined_df.columns]
    numerical_cols = [c for c in numerical_cols if c in combined_df.columns]
    
    print(f"Building similarity features from {len(categorical_cols)} categorical + {len(numerical_cols)} numerical columns...")
    
    # Compute categorical similarity features
    cat_features = _compute_categorical_similarity(combined_df, categorical_cols)
    
    # Compute numerical features
    num_features = _compute_numerical_features(combined_df, numerical_cols)
    
    # Combine features with weights (categorical gets higher weight for matching)
    # Weight categorical 2x higher since exact matches are more meaningful
    combined_features = np.hstack([
        cat_features * 2.0,
        num_features,
    ])
    
    print(f"Combined feature matrix shape: {combined_features.shape}")
    print(f"Building k-NN graph with k={k_neighbors}...")
    
    # Build k-NN graph
    nbrs = NearestNeighbors(
        n_neighbors=k_neighbors + 1,  # +1 because it includes self
        algorithm='auto',
        metric='euclidean',
        n_jobs=-1,
    )
    nbrs.fit(combined_features)
    distances, indices = nbrs.kneighbors(combined_features)
    
    # Build edge list (exclude self-loops)
    edge_list = []
    for i in range(len(combined_df)):
        for j in range(1, k_neighbors + 1):  # Skip index 0 (self)
            neighbor = indices[i, j]
            edge_list.append([i, neighbor])
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    print(f"Graph has {edge_index.shape[1]} directed edges")
    
    # Make graph undirected by adding reverse edges
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    # Remove duplicates
    edge_index = torch.unique(edge_index, dim=1)
    print(f"After making undirected: {edge_index.shape[1]} edges")
    
    # Prepare node features (all engineered features)
    feature_cols = [c for c in prepared.x.columns if c in combined_df.columns]
    
    # Convert categorical to numeric codes for GNN input
    node_features = combined_df[feature_cols].copy()
    for col in node_features.select_dtypes(include=['object', 'category']).columns:
        node_features[col] = pd.Categorical(node_features[col]).codes
    
    node_features = node_features.fillna(0).values.astype(np.float32)
    
    # Normalize numerical features
    scaler = StandardScaler()
    node_features = scaler.fit_transform(node_features)
    node_features_tensor = torch.tensor(node_features, dtype=torch.float32)
    
    print(f"Node features shape: {node_features_tensor.shape}")
    
    # Create masks
    train_mask = torch.zeros(len(combined_df), dtype=torch.bool)
    train_mask[:num_train] = True
    
    test_mask = torch.zeros(len(combined_df), dtype=torch.bool)
    if num_test > 0:
        test_mask[num_train:] = True
    
    # Labels (only for training nodes)
    labels = torch.tensor(combined_df[TARGET].values, dtype=torch.long)
    
    # Store original IDs
    train_ids = train_df[ID_COL].values
    test_ids = test_df[ID_COL].values if num_test > 0 else np.array([])
    
    graph_data = {
        'edge_index': edge_index,
        'node_features': node_features_tensor,
        'train_mask': train_mask,
        'test_mask': test_mask,
        'labels': labels,
        'train_ids': train_ids,
        'test_ids': test_ids,
        'num_train': num_train,
        'num_test': num_test,
        'feature_names': feature_cols,
        'scaler': scaler,
    }
    
    # Compute graph statistics
    avg_degree = edge_index.shape[1] / len(combined_df)
    print(f"\nGraph statistics:")
    print(f"  Nodes: {len(combined_df)}")
    print(f"  Edges: {edge_index.shape[1]}")
    print(f"  Avg degree: {avg_degree:.2f}")
    print(f"  Node features: {node_features_tensor.shape[1]}")
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(graph_data, output_path)
        print(f"\nGraph saved to: {output_path}")
    
    return graph_data


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build customer similarity graph")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to data directory (default: data/raw/)",
    )
    parser.add_argument(
        "--feature-set",
        type=str,
        default="blend_buckets",
        help="Feature set to use (default: blend_buckets)",
    )
    parser.add_argument(
        "--k-neighbors",
        type=int,
        default=15,
        help="Number of neighbors to connect (default: 15)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/graphs/customer_graph.pt",
        help="Output path for graph (default: outputs/graphs/customer_graph.pt)",
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir) if args.data_dir else repo_root() / "data" / "raw"
    output_path = repo_root() / args.output
    
    build_customer_graph(
        data_dir=data_dir,
        feature_set=args.feature_set,
        k_neighbors=args.k_neighbors,
        output_path=output_path,
        include_test=True,
    )
