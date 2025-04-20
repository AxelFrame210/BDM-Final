import pandas as pd
import numpy as np
import os

# Read the events data and limit to first 60,000 lines
events_df = pd.read_csv('new_data/events.csv', nrows=60000)

# Sort by timestamp
events_df = events_df.sort_values('timestamp')

# Count interactions per user and item
user_counts = events_df['visitorid'].value_counts()
item_counts = events_df['itemid'].value_counts()

# Filter out users and items with too few interactions
valid_users = user_counts[user_counts >= 5].index
valid_items = item_counts[item_counts >= 5].index

# Filter the dataframe
events_df = events_df[
    (events_df['visitorid'].isin(valid_users)) & 
    (events_df['itemid'].isin(valid_items))
]

# Convert timestamps to Unix timestamps (they already are)
events_df['ts'] = events_df['timestamp']

# Create continuous IDs starting from 0
user_ids = {user: idx for idx, user in enumerate(sorted(events_df['visitorid'].unique()))}
n_users = len(user_ids)
item_ids = {item: idx + n_users for idx, item in enumerate(sorted(events_df['itemid'].unique()))}

# Map to new IDs
events_df['u'] = events_df['visitorid'].map(user_ids)
events_df['i'] = events_df['itemid'].map(item_ids)

# Create edge indices
events_df['idx'] = range(len(events_df))

# Convert event types to labels (view=0, addtocart=1, transaction=2)
event_map = {'view': 0, 'addtocart': 1, 'transaction': 2}
events_df['label'] = events_df['event'].map(event_map)

# Select and rename columns
processed_df = events_df[['u', 'i', 'ts', 'idx', 'label']]

# Save to CSV
processed_df.to_csv('data/ml_retail.csv', index=False)

# Create random features as required by the model
n_nodes = len(user_ids) + len(item_ids)
n_edges = len(events_df)
dim = 172  # Default dimension used in the model

# Create random node features
node_features = np.random.rand(n_nodes + 1, dim)  # +1 for index 0
np.save('data/ml_retail_node.npy', node_features)

# Create random edge features
edge_features = np.random.rand(n_edges, dim)
np.save('data/ml_retail.npy', edge_features)

print(f"Processed {len(events_df)} interactions")
print(f"Number of unique users: {len(user_ids)}")
print(f"Number of unique items: {len(item_ids)}")
print(f"Total number of nodes: {n_nodes}")
print(f"Time range: {events_df['timestamp'].min()} to {events_df['timestamp'].max()}")
print(f"Average interactions per user: {len(events_df)/len(user_ids):.2f}")
print(f"Average interactions per item: {len(events_df)/len(item_ids):.2f}")

# Verify node IDs are continuous
all_nodes = np.concatenate([events_df['u'].values, events_df['i'].values])
print(f"Min node ID: {all_nodes.min()}")
print(f"Max node ID: {all_nodes.max()}")
print(f"Expected max ID: {n_nodes - 1}") 