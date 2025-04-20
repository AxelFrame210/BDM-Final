import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Read the data
events_df = pd.read_csv('new_data/events.csv')
category_df = pd.read_csv('new_data/category_tree.csv')

# Convert timestamp to datetime
events_df['datetime'] = pd.to_datetime(events_df['timestamp'], unit='ms')

# 1. Interaction Types Distribution
plt.figure(figsize=(10, 6))
sns.countplot(data=events_df, x='event')
plt.title('Distribution of Interaction Types')
plt.xlabel('Interaction Type')
plt.ylabel('Count')
plt.savefig('visualizations/interaction_types.png')
plt.close()

# 2. Temporal Distribution of Interactions
plt.figure(figsize=(12, 6))
events_df.set_index('datetime').resample('D').size().plot()
plt.title('Daily Interaction Volume Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Interactions')
plt.savefig('visualizations/daily_interactions.png')
plt.close()

# 3. User Interaction Distribution
user_interactions = events_df.groupby('visitorid').size()
plt.figure(figsize=(10, 6))
sns.histplot(user_interactions, bins=50, log_scale=(False, True))
plt.title('Distribution of User Interactions')
plt.xlabel('Number of Interactions per User')
plt.ylabel('Count (log scale)')
plt.savefig('visualizations/user_interactions.png')
plt.close()

# 4. Item Popularity Distribution
item_interactions = events_df.groupby('itemid').size()
plt.figure(figsize=(10, 6))
sns.histplot(item_interactions, bins=50, log_scale=(False, True))
plt.title('Distribution of Item Interactions')
plt.xlabel('Number of Interactions per Item')
plt.ylabel('Count (log scale)')
plt.savefig('visualizations/item_interactions.png')
plt.close()

# 5. Category Tree Visualization
plt.figure(figsize=(12, 8))
category_counts = category_df['parentid'].value_counts().head(20)
sns.barplot(x=category_counts.values, y=category_counts.index)
plt.title('Top 20 Parent Categories by Number of Subcategories')
plt.xlabel('Number of Subcategories')
plt.ylabel('Parent Category ID')
plt.savefig('visualizations/category_tree.png')
plt.close()

# Print statistics
print("\nDataset Statistics:")
print(f"Total interactions: {len(events_df):,}")
print(f"Unique users: {events_df['visitorid'].nunique():,}")
print(f"Unique items: {events_df['itemid'].nunique():,}")
print("\nInteraction Types:")
print(events_df['event'].value_counts())
print("\nTime Range:")
print(f"Start: {events_df['datetime'].min()}")
print(f"End: {events_df['datetime'].max()}")
print(f"Duration: {events_df['datetime'].max() - events_df['datetime'].min()}") 