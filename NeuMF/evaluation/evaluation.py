import torch
import numpy as np
from sklearn.metrics import ndcg_score
import torch.nn.functional as F

def hit_ratio_at_k(y_true, y_pred, k=10):
    """Compute Hit Ratio@k."""
    # Convert inputs to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Handle scalar values and single-item arrays
    if y_true.ndim == 0:
        y_true = np.array([y_true])
    if y_pred.ndim == 0:
        y_pred = np.array([y_pred])
    
    # Ensure 2D arrays
    if y_true.ndim == 1:
        y_true = y_true.reshape(1, -1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(1, -1)
    
    # Handle empty arrays
    if y_true.size == 0 or y_pred.size == 0:
        return 0.0
    
    # Get top-k predictions
    top_k = np.argsort(y_pred, axis=1)[:, -k:]
    
    # Check if any true items are in top-k
    hits = np.array([np.any(np.isin(top_k[i], np.where(y_true[i] > 0)[0])) 
                    for i in range(len(y_true))])
    
    return np.mean(hits)

def safe_ndcg_score(y_true, y_pred, k=10):
    """Compute NDCG score safely handling edge cases."""
    # Convert inputs to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Handle scalar values and single-item arrays
    if y_true.ndim == 0:
        y_true = np.array([y_true])
    if y_pred.ndim == 0:
        y_pred = np.array([y_pred])
    
    # Ensure 2D arrays
    if y_true.ndim == 1:
        y_true = y_true.reshape(1, -1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(1, -1)
    
    # Handle empty arrays
    if y_true.size == 0 or y_pred.size == 0:
        return 0.0
    
    # Calculate DCG
    order = np.argsort(y_pred, axis=1)[:, ::-1]
    y_true_sorted = np.take_along_axis(y_true, order, axis=1)
    gains = 2 ** y_true_sorted - 1
    discounts = np.log2(np.arange(2, y_true.shape[1] + 2))
    dcg = np.sum(gains[:, :k] / discounts[:k], axis=1)
    
    # Calculate IDCG
    ideal_order = np.argsort(y_true, axis=1)[:, ::-1]
    ideal_gains = 2 ** np.take_along_axis(y_true, ideal_order, axis=1) - 1
    idcg = np.sum(ideal_gains[:, :k] / discounts[:k], axis=1)
    
    # Avoid division by zero
    idcg[idcg == 0] = 1
    
    # Calculate NDCG
    ndcg = dcg / idcg
    return np.mean(ndcg)

def evaluate_model(model, data_loader, device, k=10):
    """Evaluate model performance."""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Get predictions
            predictions = model(batch)
            labels = batch['label']
            
            # Collect predictions and labels
            all_predictions.append(predictions.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    # Concatenate all predictions and labels
    predictions = np.concatenate(all_predictions, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    # Calculate metrics
    ndcg = safe_ndcg_score(labels, predictions, k)
    hit_ratio = hit_ratio_at_k(labels, predictions, k)
    
    return {
        f'ndcg@{k}': ndcg,
        f'hit_ratio@{k}': hit_ratio
    } 