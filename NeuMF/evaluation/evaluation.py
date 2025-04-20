import torch
import numpy as np
from sklearn.metrics import ndcg_score
import torch.nn.functional as F

def recall_at_k(recommendations, test_items, k):
    hits = len(set(recommendations[:k]) & set(test_items))
    return hits / min(k, len(test_items))

def precision_at_k(recommendations, test_items, k):
    hits = len(set(recommendations[:k]) & set(test_items))
    return hits / k

def eval_recommendation(model, data_loader, device, k=10, num_neg=100):
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            user = batch['user'].to(device)
            item = batch['item'].to(device)
            label = batch['label'].to(device)
            temporal_embedding = batch['temporal_embedding'].to(device)
            
            # Generate negative samples
            batch_size = user.size(0)
            neg_items = torch.randint(0, model.num_items, (batch_size, num_neg), device=device)
            neg_users = user.unsqueeze(1).repeat(1, num_neg)
            
            # Repeat temporal embeddings for negative samples
            temporal_embedding_neg = temporal_embedding.unsqueeze(1).repeat(1, num_neg, 1)
            temporal_embedding_neg = temporal_embedding_neg.view(-1, temporal_embedding.size(-1))
            
            # Get predictions for positive and negative samples
            pos_pred = model(user, item, temporal_embedding)
            neg_pred = model(
                neg_users.view(-1),
                neg_items.view(-1),
                temporal_embedding_neg
            ).view(batch_size, num_neg)
            
            # Apply temperature scaling
            temperature = 0.1
            pos_pred = pos_pred / temperature
            neg_pred = neg_pred / temperature
            
            # Combine predictions and apply sigmoid
            all_pred = torch.cat([pos_pred.unsqueeze(1), neg_pred], dim=1)
            all_pred = torch.sigmoid(all_pred)
            
            all_label = torch.zeros_like(all_pred)
            all_label[:, 0] = 1  # First item is positive
            
            all_predictions.extend(all_pred.cpu().numpy())
            all_labels.extend(all_label.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    ndcg_scores = []
    recall_scores = []
    precision_scores = []
    
    for i in range(len(all_predictions)):
        predictions = all_predictions[i]
        labels = all_labels[i]
        
        # Get top-k recommendations
        top_k_indices = np.argsort(predictions)[::-1][:k]
        recommendations = top_k_indices
        test_items = np.where(labels == 1)[0]
        
        # Calculate metrics
        ndcg_scores.append(ndcg_score(labels.reshape(1, -1), predictions.reshape(1, -1), k=k))
        recall_scores.append(recall_at_k(recommendations, test_items, k))
        precision_scores.append(precision_at_k(recommendations, test_items, k))
    
    # Average metrics
    ndcg = np.mean(ndcg_scores)
    recall = np.mean(recall_scores)
    precision = np.mean(precision_scores)
    
    # Add standard deviation for uncertainty estimation
    metrics = {
        'ndcg@10': ndcg,
        'ndcg@10_std': np.std(ndcg_scores),
        'recall@10': recall,
        'recall@10_std': np.std(recall_scores),
        'precision@10': precision,
        'precision@10_std': np.std(precision_scores)
    }
    
    return metrics 