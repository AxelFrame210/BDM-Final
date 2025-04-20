import torch
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import sys
import os
from torch.serialization import add_safe_globals
import torch.nn.functional as F
import pandas as pd
from collections import defaultdict
from model.tgn_cpu import TGN_CPU
from utils.data import get_data, compute_time_statistics
from utils.utils import get_neighbor_finder

# Add numpy scalar to safe globals
add_safe_globals(['numpy._core.multiarray.scalar'])

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        # Get dimensions from the model
        self.n_node_features = model.n_node_features
        self.n_edge_features = model.n_edge_features
        self.n_time_features = model.n_node_features  # Time features use same dimension as node features
    
    def forward(self, x):
        # Split the input into user, item, and temporal embeddings
        batch_size = x.shape[0]
        user_emb = x[:, :self.n_node_features]  # First n_node_features dimensions are user embeddings
        item_emb = x[:, self.n_node_features:2*self.n_node_features]  # Next n_node_features dimensions are item embeddings
        temporal_emb = x[:, 2*self.n_node_features:]  # Remaining dimensions are temporal embeddings
        
        # Apply layer normalization to embeddings
        user_emb = F.layer_norm(user_emb, [user_emb.size(-1)])
        item_emb = F.layer_norm(item_emb, [item_emb.size(-1)])
        temporal_emb = F.layer_norm(temporal_emb, [temporal_emb.size(-1)])
        
        # Use the affinity score to predict
        predictions = self.model.affinity_score(user_emb, item_emb)
        
        # Scale predictions to make SHAP values more interpretable
        return predictions * 100  # Scale up to make differences more visible

def load_model_and_data(dataset_name, device):
    # Load data with correct dimensions
    node_features, edge_features, full_data, train_data, val_data, test_data = get_data(dataset_name, dim=64)
    
    # Calculate num_users and num_items from the data
    all_nodes = set(train_data.sources) | set(train_data.destinations)
    num_users = len(set(train_data.sources))
    num_items = len(set(train_data.destinations))
    
    print(f"The dataset has {len(full_data.sources)} interactions, involving {len(all_nodes)} different nodes")
    print(f"The training dataset has {len(train_data.sources)} interactions, involving {len(set(train_data.sources) | set(train_data.destinations))} different nodes")
    print(f"The validation dataset has {len(val_data.sources)} interactions, involving {len(set(val_data.sources) | set(val_data.destinations))} different nodes")
    print(f"The test dataset has {len(test_data.sources)} interactions, involving {len(set(test_data.sources) | set(test_data.destinations))} different nodes")
    
    # Initialize neighbor finder
    neighbor_finder = get_neighbor_finder(train_data, uniform=True)
    
    # Compute time statistics
    mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = \
        compute_time_statistics(full_data.sources, full_data.destinations, full_data.timestamps)
    
    # Initialize model with correct dimensions
    model = TGN_CPU(neighbor_finder=neighbor_finder,
                    node_features=node_features,
                    edge_features=edge_features,
                    n_layers=1,
                    n_heads=2,
                    dropout=0.1,
                    use_memory=True,
                    message_dimension=100,
                    memory_dimension=64,
                    memory_update_at_start=True,
                    embedding_module_type="graph_attention",
                    message_function="identity",
                    mean_time_shift_src=mean_time_shift_src,
                    std_time_shift_src=std_time_shift_src,
                    mean_time_shift_dst=mean_time_shift_dst,
                    std_time_shift_dst=std_time_shift_dst,
                    n_neighbors=10,
                    aggregator_type="last",
                    memory_updater_type="gru",
                    use_destination_embedding_in_message=False,
                    use_source_embedding_in_message=False,
                    dyrep=False)
    
    # Load checkpoint
    checkpoint = torch.load('saved/retail_tgn_best.pth', map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    
    return model, train_data, val_data, test_data

def prepare_shap_data(model, data, device, num_samples=1000):
    """Prepare data for SHAP analysis."""
    # Get a subset of the data
    indices = np.random.choice(len(data.sources), min(num_samples, len(data.sources)), replace=False)
    source_nodes = data.sources[indices]
    destination_nodes = data.destinations[indices]
    edge_times = data.timestamps[indices]
    edge_idxs = data.edge_idxs[indices]
    
    # Get model embeddings
    with torch.no_grad():
        # Initialize memory
        if model.use_memory:
            model.memory.__init_memory__()
        
        # Compute temporal embeddings
        source_emb, destination_emb, _ = model.compute_temporal_embeddings(
            source_nodes,
            destination_nodes,
            destination_nodes,  # Using destinations as negative samples (not used)
            edge_times,
            edge_idxs,
            n_neighbors=10
        )
        
        # Create temporal features
        edge_times_tensor = torch.from_numpy(edge_times).float().to(device)
        edge_times_tensor = edge_times_tensor.unsqueeze(1)  # Add batch dimension
        temporal_features = model.time_encoder(edge_times_tensor)
        
        # Combine embeddings for SHAP analysis
        combined_emb = torch.cat([
            source_emb,
            destination_emb,
            temporal_features.squeeze(1)  # Remove batch dimension
        ], dim=1)
    
    return combined_emb.cpu().numpy()

def compute_shap_values(model, background_data, evaluation_data, device):
    """Compute SHAP values for the model predictions using GradientExplainer."""
    # Convert data to tensors
    background_tensor = torch.tensor(background_data, dtype=torch.float32).to(device)
    evaluation_tensor = torch.tensor(evaluation_data, dtype=torch.float32).to(device)
    
    try:
        # Initialize GradientExplainer
        print("Initializing SHAP GradientExplainer...")
        explainer = shap.GradientExplainer(model, background_tensor)
        
        # Compute SHAP values
        print("Computing SHAP values... This may take a while.")
        shap_tensor = explainer.shap_values(evaluation_tensor)
        
        # Convert to numpy array and process
        if isinstance(shap_tensor, list):
            shap_values = np.array(shap_tensor[0])  # Take first output for binary classification
        else:
            shap_values = shap_tensor.cpu().numpy()
        
        # Handle any numerical issues
        shap_values = np.nan_to_num(shap_values, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Scale values to make them more interpretable
        abs_max = np.max(np.abs(shap_values))
        if abs_max > 0:
            shap_values = shap_values / abs_max
        
        return shap_values, explainer
        
    except Exception as e:
        print(f"Error computing SHAP values with GradientExplainer: {e}")
        print("Falling back to simpler computation method...")
        
        # Create a simple explainer object for visualization
        class SimpleExplainer:
            def __init__(self, expected_value):
                self.expected_value = expected_value
        
        # Compute simple feature importance
        with torch.no_grad():
            base_pred = model(background_tensor.mean(dim=0, keepdim=True))
            base_pred = base_pred.cpu().numpy().mean()
            
            shap_values = np.zeros((len(evaluation_data), evaluation_data.shape[1]))
            for i in range(len(evaluation_data)):
                actual_pred = model(evaluation_tensor[i:i+1])
                diff = (actual_pred - base_pred).cpu().numpy()
                shap_values[i] = diff * (evaluation_data[i] - background_data.mean(axis=0))
            
            # Normalize values
            abs_max = np.max(np.abs(shap_values))
            if abs_max > 0:
                shap_values = shap_values / abs_max
            
            explainer = SimpleExplainer(base_pred)
            
            return shap_values, explainer

def visualize_shap_values(shap_values, output_dir):
    """Visualize SHAP values with enhanced density and information."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Handle any remaining infinite or NaN values
    shap_values = np.nan_to_num(shap_values, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Calculate feature importance (mean absolute SHAP values)
    feature_importance = np.mean(np.abs(shap_values), axis=0)
    
    # Create feature names based on the model's dimensions
    n_node_features = 64  # From get_data function
    n_time_features = 64  # Same as node features
    
    # Create feature names
    feature_names = []
    feature_types = []
    
    # User features
    for i in range(n_node_features):
        feature_names.append(f'user_emb_{i}')
        feature_types.append('User')
    
    # Item features
    for i in range(n_node_features):
        feature_names.append(f'item_emb_{i}')
        feature_types.append('Item')
    
    # Temporal features
    for i in range(n_time_features):
        feature_names.append(f'temporal_emb_{i}')
        feature_types.append('Temporal')
    
    # Create DataFrame for analysis
    analysis_df = pd.DataFrame({
        'Feature': feature_names,
        'Type': feature_types,
        'Importance': feature_importance
    })
    
    # Aggregate features by type
    type_importance = analysis_df.groupby('Type')['Importance'].agg(['mean', 'max', 'min', 'count'])
    
    # Print summary statistics
    print("\nFeature Importance Summary:")
    for feature_type in ['User', 'Item', 'Temporal']:
        stats = type_importance.loc[feature_type]
        print(f"\n{feature_type} Features:")
        print(f"Average Importance: {stats['mean']:.6f}")
        print(f"Max Importance: {stats['max']:.6f}")
        print(f"Min Importance: {stats['min']:.6f}")
        print(f"Count: {stats['count']}")
    
    # Save analysis to CSV
    analysis_df.to_csv(os.path.join(output_dir, 'feature_importance_analysis.csv'), index=False)
    
    # Create bar plot for top features
    plt.figure(figsize=(20, 12))
    
    # Select top 30 features
    top_features = analysis_df.nlargest(30, 'Importance')
    
    # Create positions for bars
    positions = np.arange(len(top_features))
    
    # Plot bars with color coding
    colors = {'User': 'blue', 'Item': 'green', 'Temporal': 'red'}
    for feature_type in ['User', 'Item', 'Temporal']:
        mask = top_features['Type'] == feature_type
        plt.bar(positions[mask], top_features['Importance'][mask], 
                color=colors[feature_type], label=feature_type)
    
    plt.xlabel('Feature')
    plt.ylabel('Feature Importance (mean |SHAP value|)')
    plt.title('Top 30 Most Important Features')
    plt.xticks(positions, top_features['Feature'], rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_30_features.png'))
    plt.close()
    
    # Create summary plot
    plt.figure(figsize=(20, 12))
    shap.summary_plot(shap_values, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shap_summary.png'))
    plt.close()

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = 'results/shap_analysis'
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading model and data...")
    model, train_data, val_data, test_data = load_model_and_data('retail', device)
    
    # Initialize memory
    if model.use_memory:
        model.memory.__init_memory__()
    
    # Wrap model for SHAP analysis
    wrapped_model = ModelWrapper(model).to(device)
    wrapped_model.eval()
    
    print("Preparing data for SHAP analysis...")
    background_data = prepare_shap_data(model, train_data, device, num_samples=100)
    evaluation_data = prepare_shap_data(model, test_data, device, num_samples=100)
    
    print("Computing SHAP values...")
    shap_values, explainer = compute_shap_values(wrapped_model, background_data, evaluation_data, device)
    
    print("Visualizing results...")
    visualize_shap_values(shap_values, output_dir)
    
    print(f"\nResults have been saved to {output_dir}/")

if __name__ == "__main__":
    main() 