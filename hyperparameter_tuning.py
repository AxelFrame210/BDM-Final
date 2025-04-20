import pandas as pd
import numpy as np
from itertools import product
import torch
from model.tgn_cpu import TGN_CPU
from utils.data import Data
from evaluation.evaluation import eval_recommendation
from utils.utils import get_neighbor_finder
import json
from sklearn.model_selection import train_test_split

# Define hyperparameter ranges
param_grid = {
    'n_degree': [5, 10, 15, 20],  # number of neighbors
    'n_epoch': [10, 20, 30, 40],  # number of epochs
    'memory_dim': [172],  # memory dimension (must match node feature dimension)
    'bs': [32, 64]  # batch size
}

# Create all combinations
param_combinations = [dict(zip(param_grid.keys(), v)) for v in product(*param_grid.values())]

# Results storage
results = []

# Load preprocessed data
processed_df = pd.read_csv('data/ml_retail.csv')

# Sort by timestamp
processed_df = processed_df.sort_values('ts')

# Split data into train and validation temporally
train_size = int(len(processed_df) * 0.8)
train_df = processed_df.iloc[:train_size]
val_df = processed_df.iloc[train_size:]

# Reset indices to ensure they are continuous
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

# Create Data objects
train_data = Data(
    sources=train_df['u'].values,
    destinations=train_df['i'].values,
    timestamps=train_df['ts'].values,
    edge_idxs=train_df.index.values,  # Use continuous indices
    labels=train_df['label'].values
)

val_data = Data(
    sources=val_df['u'].values,
    destinations=val_df['i'].values,
    timestamps=val_df['ts'].values,
    edge_idxs=val_df.index.values,  # Use continuous indices
    labels=val_df['label'].values
)

# Load node and edge features
node_features = np.load('data/ml_retail_node.npy')
edge_features = np.load('data/ml_retail.npy')

# Create neighbor finder
neighbor_finder = get_neighbor_finder(train_data, uniform=True)

print("Starting hyperparameter tuning...")
print(f"Total combinations to try: {len(param_combinations)}")

for idx, params in enumerate(param_combinations):
    print(f"\nTrying combination {idx + 1}/{len(param_combinations)}")
    print(f"Parameters: {params}")
    
    # Initialize model with current parameters
    tgn = TGN_CPU(
        neighbor_finder=neighbor_finder,
        node_features=node_features,
        edge_features=edge_features,
        n_layers=2,
        n_heads=2,
        dropout=0.1,
        use_memory=True,
        memory_update_at_start=True,
        message_dimension=params['memory_dim'],
        memory_dimension=params['memory_dim'],
        embedding_module_type="graph_attention",
        message_function="identity",
        mean_time_shift_src=0,
        std_time_shift_src=1,
        mean_time_shift_dst=0,
        std_time_shift_dst=1,
        n_neighbors=params['n_degree'],
        aggregator_type="last",
        memory_updater_type="gru",
        use_destination_embedding_in_message=False,
        use_source_embedding_in_message=False,
        dyrep=False
    )
    
    # Train model
    optimizer = torch.optim.Adam(tgn.parameters(), lr=0.0001)
    criterion = torch.nn.BCELoss()
    
    for epoch in range(params['n_epoch']):
        # Training loop
        tgn.train()
        optimizer.zero_grad()
        
        # Forward pass
        source_embedding, destination_embedding, _ = tgn.compute_temporal_embeddings(
            train_data.sources,
            train_data.destinations,
            train_data.destinations,  # Using same destinations as negatives for simplicity
            train_data.timestamps,
            train_data.edge_idxs,
            params['n_degree']
        )
        
        # Compute loss
        scores = torch.sum(source_embedding * destination_embedding, dim=1)
        loss = criterion(torch.sigmoid(scores), torch.ones_like(scores))
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    # Evaluate
    tgn.eval()
    eval_dict = eval_recommendation(
        tgn=tgn,
        data=val_data,
        batch_size=params['bs'],
        n_neighbors=params['n_degree'],
        NUM_NEG_EVAL=20,  # Reduced from 100 to match our dataset size
        is_test_run=False
    )
    
    # Calculate average metrics
    ndcg_10 = np.mean([x[2] for x in eval_dict['ndcgs']])  # NDCG@10
    recall_10 = np.mean([x[2] for x in eval_dict['recalls']])  # Recall@10
    
    results.append({
        'params': params,
        'ndcg@10': ndcg_10,
        'recall@10': recall_10
    })
    
    # Save intermediate results
    df = pd.DataFrame(results)
    df.to_csv('hyperparameter_results.csv', index=False)

# Find best parameters
best_ndcg = max(results, key=lambda x: x['ndcg@10'])
best_recall = max(results, key=lambda x: x['recall@10'])

print("\nHyperparameter Tuning Results:")
print("\nBest NDCG@10 Score:", best_ndcg['ndcg@10'])
print("Parameters:", best_ndcg['params'])
print("\nBest Recall@10 Score:", best_recall['recall@10'])
print("Parameters:", best_recall['params'])

# Create final results table
results_df = pd.DataFrame(results)
print("\nAll Results:")
print(results_df.to_string()) 