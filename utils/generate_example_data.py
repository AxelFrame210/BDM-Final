import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_example_data():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Parameters
    n_users = 100
    n_items = 50
    n_interactions = 1000
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    
    # Generate user-item interactions
    users = np.random.randint(0, n_users, n_interactions)
    items = np.random.randint(0, n_items, n_interactions)
    
    # Generate timestamps (convert to unix timestamp)
    time_delta = (end_date - start_date).total_seconds()
    timestamps = np.array([
        start_date.timestamp() + np.random.randint(0, int(time_delta))
        for _ in range(n_interactions)
    ])
    
    # Sort by time and create indices
    time_ix = np.argsort(timestamps)
    users = users[time_ix]
    items = items[time_ix]
    timestamps = timestamps[time_ix]
    
    # Create DataFrame with required columns
    df = pd.DataFrame({
        'u': users,  # source nodes (users)
        'i': items,  # destination nodes (items)
        'ts': timestamps,  # timestamps
        'label': np.random.randint(0, 2, n_interactions),  # binary interaction (like/dislike)
        'idx': range(n_interactions),  # index for temporal ordering
        'ext_roll': np.random.randint(0, 5, n_interactions)  # example of external feature
    })
    
    # Save to CSV in the expected format
    df.to_csv('data/ml_example.csv', index=False)
    
    # Create node features (combined for users and items)
    n_total_nodes = n_users + n_items
    node_features = np.random.randn(n_total_nodes, 172)  # 172 is default feature dimension
    np.save('data/ml_example_node.npy', node_features)
    
    # Create edge features
    edge_features = np.random.randn(n_interactions, 172)  # 172 is default feature dimension
    np.save('data/ml_example_edge.npy', edge_features)
    
    print(f"Generated example dataset with {n_users} users, {n_items} items, and {n_interactions} interactions")
    print("Files saved in data/ directory:")
    print("- ml_example.csv (interaction data)")
    print("- ml_example_node.npy (node features)")
    print("- ml_example_edge.npy (edge features)")

if __name__ == "__main__":
    generate_example_data() 