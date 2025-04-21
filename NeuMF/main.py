import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np
import argparse
import pickle
from pathlib import Path
import time

from model.neumf import NeuMF
from data.data_loader import create_data_loaders
from evaluation.evaluation import evaluate_model

class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight=2.0, reduction='mean'):
        super().__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction
        
    def forward(self, pred, target):
        # Add label smoothing
        target = target * 0.9 + 0.05
        
        # Add small noise to break ties
        noise = torch.randn_like(pred) * 0.01
        pred = pred + noise
        
        # Calculate weighted BCE loss
        loss = -(self.pos_weight * target * torch.log(torch.sigmoid(pred) + 1e-10) + 
                (1 - target) * torch.log(1 - torch.sigmoid(pred) + 1e-10))
        
        if self.reduction == 'mean':
            return loss.mean()
        return loss.sum()

class TimeAwareNegativeSampler:
    def __init__(self, num_items, temporal_window=1000):
        self.num_items = num_items
        self.temporal_window = temporal_window
        self.item_history = {}  # item_id -> list of timestamps
        self.item_freq = torch.ones(num_items)  # Initialize with ones to avoid zero probabilities
    
    def update(self, items, timestamps):
        for item, ts in zip(items.cpu().numpy(), timestamps.cpu().numpy()):
            if item not in self.item_history:
                self.item_history[item] = []
            self.item_history[item].append(ts)
            self.item_freq[item] += 1
    
    def sample(self, pos_items, timestamps, num_samples, device):
        batch_size = pos_items.size(0)
        neg_items = []
        
        # Convert to numpy for easier processing
        pos_items = pos_items.cpu().numpy()
        timestamps = timestamps.cpu().numpy()
        
        for i in range(batch_size):
            current_ts = timestamps[i]
            pos_item = pos_items[i]
            
            # Get items that haven't been interacted with in the temporal window
            valid_items = []
            weights = []
            
            for item in range(self.num_items):
                if item == pos_item:
                    continue
                
                history = self.item_history.get(item, [])
                if not history or abs(history[-1] - current_ts) > self.temporal_window:
                    valid_items.append(item)
                    weights.append(max(1.0, self.item_freq[item].item()))  # Use at least 1.0 as weight
            
            if not valid_items:
                valid_items = list(range(self.num_items))
                valid_items.remove(pos_item)
                weights = [max(1.0, self.item_freq[item].item()) for item in valid_items]
            
            # Convert to numpy arrays
            valid_items = np.array(valid_items)
            weights = np.array(weights, dtype=np.float64)  # Use float64 for better numerical stability
            
            # Ensure weights are positive and sum to 1
            weights = np.maximum(weights, 1e-10)  # Add small epsilon to avoid zeros
            weights = weights / weights.sum()
            
            # Sample negative items
            item_samples = np.random.choice(valid_items, size=num_samples, p=weights)
            neg_items.append(item_samples)
        
        return torch.tensor(neg_items, device=device)

def save_results_to_file(results, filename="results.txt"):
    with open(filename, "w") as f:
        # Define column headers and widths
        columns = [
            ("Epoch", 8),
            ("Loss", 10),
            ("Val NDCG", 10),
            ("Test NDCG", 10),
            ("Val Recall", 12),
            ("Test Recall", 12),
            ("Val Prec", 10),
            ("Test Prec", 10),
            ("LR", 10),
            ("Neg Samp", 10)
        ]
        
        # Write header
        header = "".join(f"{name:<{width}}" for name, width in columns)
        f.write(header + "\n")
        
        # Write subheader for @10 indicators
        subheader = " " * 8 + " " * 10  # Epoch and Loss columns
        subheader += "".join(f"{'@10':<{width-3}}" for name, width in columns[2:8])  # NDCG, Recall, Prec columns
        subheader += " " * 20  # LR and Neg Samp columns
        f.write(subheader + "\n")
        
        f.write("-" * sum(width for _, width in columns) + "\n")
        
        # Write data rows
        for epoch, metrics in results.items():
            row = (
                f"{epoch:<8}"
                f"{metrics['loss']:<10.4f}"
                f"{metrics['val_ndcg']:<10.4f}"
                f"{metrics['test_ndcg']:<10.4f}"
                f"{metrics['val_recall']:<12.4f}"
                f"{metrics['test_recall']:<12.4f}"
                f"{metrics['val_precision']:<10.4f}"
                f"{metrics['test_precision']:<10.4f}"
                f"{metrics['lr']:<10.6f}"
                f"{metrics['neg_samples']:<10.1f}"
            )
            f.write(row + "\n")
        
        # Write best results
        f.write("\nBest Results:\n")
        f.write("-" * sum(width for _, width in columns) + "\n")
        
        best_epoch = max(results.items(), key=lambda x: x[1]['val_ndcg'])[0]
        best_metrics = results[best_epoch]
        
        f.write(f"Best Epoch: {best_epoch}\n")
        f.write(f"Best Val NDCG@10: {best_metrics['val_ndcg']:.4f}\n")
        f.write(f"Test NDCG@10 at Best Epoch: {best_metrics['test_ndcg']:.4f}\n")
        f.write(f"Val Recall@10 at Best Epoch: {best_metrics['val_recall']:.4f}\n")
        f.write(f"Test Recall@10 at Best Epoch: {best_metrics['test_recall']:.4f}\n")
        f.write(f"Val Precision@10 at Best Epoch: {best_metrics['val_precision']:.4f}\n")
        f.write(f"Test Precision@10 at Best Epoch: {best_metrics['test_precision']:.4f}\n")
        f.write("Parameters at Best Epoch:\n")
        f.write(f"- Learning Rate: {best_metrics['lr']:.6f}\n")
        f.write(f"- Negative Samples: {best_metrics['neg_samples']:.1f}\n")
        f.write(f"- Loss: {best_metrics['loss']:.4f}\n")

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr=1e-6):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr, 0.5 * (1.0 + np.cos(np.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train_neumf(model, train_loader, val_loader, test_loader, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Initialize loss and optimizer
    criterion = WeightedBCELoss(pos_weight=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
    
    # Learning rate scheduler with longer warmup
    num_warmup_steps = len(train_loader) * 5  # 5 epochs of warmup
    num_training_steps = len(train_loader) * config.num_epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, 
                                              num_warmup_steps=num_warmup_steps,
                                              num_training_steps=num_training_steps)
    
    # Early stopping
    best_val_metric = 0
    patience = 20
    patience_counter = 0
    best_epoch = 0
    
    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.num_epochs}')
        
        for batch in progress_bar:
            user_ids = batch['user'].to(device)
            item_ids = batch['item'].to(device)
            labels = batch['label'].to(device)
            temporal_embeddings = batch['temporal_embedding'].to(device)
            
            # Forward pass
            pred = model(user_ids, item_ids, temporal_embeddings)
            loss = criterion(pred, labels)
            
            # L2 regularization
            l2_reg = 0
            for param in model.parameters():
                l2_reg += torch.norm(param)
            loss += 0.001 * l2_reg
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.6f}'
            })
        
        # Evaluate
        avg_loss = total_loss / len(train_loader)
        val_metrics = evaluate_model(model, val_loader, device)
        test_metrics = evaluate_model(model, test_loader, device)
        
        print(f'Epoch {epoch+1}: Loss: {avg_loss:.4f}, '
              f'Val NDCG@10: {val_metrics["ndcg"]:.4f}, Test NDCG@10: {test_metrics["ndcg"]:.4f}, '
              f'Val Hit Ratio@10: {val_metrics["hit_ratio"]:.4f}, Test Hit Ratio@10: {test_metrics["hit_ratio"]:.4f}, '
              f'LR: {scheduler.get_last_lr()[0]:.6f}')
        
        # Early stopping check
        if val_metrics['ndcg'] > best_val_metric:
            best_val_metric = val_metrics['ndcg']
            best_epoch = epoch + 1
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping after {patience} epochs without improvement')
                print(f'Best epoch: {best_epoch}, Best Val NDCG@10: {best_val_metric:.4f}')
                break

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='retail', help='Dataset name')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--num_neg_samples', type=int, default=10, help='Number of negative samples')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    parser.add_argument('--embedding_dim', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--num_heads', type=int, default=16, help='Number of attention heads')
    args = parser.parse_args()
    
    # Create save directories if they don't exist
    Path("saved/").mkdir(parents=True, exist_ok=True)
    
    # Load TGN embeddings and data
    with open(f'results/{args.data}_tgn_embeddings.pkl', 'rb') as f:
        tgn_data = pickle.load(f)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        tgn_data['train'],
        tgn_data['val'],
        tgn_data['test'],
        batch_size=args.batch_size
    )
    
    # Initialize NeuMF model with improved architecture
    num_users = max(tgn_data['train']['user_indices']) + 1
    num_items = max(tgn_data['train']['item_indices']) + 1
    
    model = NeuMF(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=args.embedding_dim,
        num_heads=args.num_heads,
        dropout=args.dropout
    ).to(args.device)
    
    # Train model
    train_neumf(
        model,
        train_loader,
        val_loader,
        test_loader,
        args
    )

if __name__ == '__main__':
    main() 