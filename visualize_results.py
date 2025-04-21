import matplotlib.pyplot as plt
import numpy as np
import pickle
import glob
import os
import seaborn as sns

def load_epoch_results(prefix):
    """Load all epoch results for a given experiment prefix."""
    results = []
    for filepath in sorted(glob.glob(f"results/{prefix}_epoch_*_eval.pkl")):
        with open(filepath, 'rb') as f:
            results.append(pickle.load(f))
    return results

def plot_training_curves(results, prefix, save_dir='plots'):
    """Plot training and evaluation metrics."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract metrics
    epochs = [r['epoch'] for r in results]
    train_losses = [r['loss'] for r in results]
    
    # Calculate mean metrics across all validation and test instances
    val_ndcgs = [np.mean([x[2] for x in r['val']['ndcgs']], axis=0) for r in results]  # NDCG@10
    test_ndcgs = [np.mean([x[2] for x in r['test']['ndcgs']], axis=0) for r in results]  # NDCG@10
    
    val_hit_ratios = [np.mean([x[2] for x in r['val']['hit_ratios']], axis=0) for r in results]  # Hit Ratio@10
    test_hit_ratios = [np.mean([x[2] for x in r['test']['hit_ratios']], axis=0) for r in results]  # Hit Ratio@10
    
    # Set style
    plt.style.use('bmh')
    
    # Plot 1: Training Loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, marker='o', linewidth=2)
    plt.title('Training Loss per Epoch', fontsize=14, pad=15)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{prefix}_training_loss.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: NDCG@10
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, val_ndcgs, marker='o', linewidth=2, label='Validation')
    plt.plot(epochs, test_ndcgs, marker='o', linewidth=2, label='Test')
    plt.title('NDCG@10 per Epoch', fontsize=14, pad=15)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('NDCG@10', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{prefix}_ndcg.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Hit Ratio@10
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, val_hit_ratios, marker='o', linewidth=2, label='Validation')
    plt.plot(epochs, test_hit_ratios, marker='o', linewidth=2, label='Test')
    plt.title('Hit Ratio@10 per Epoch', fontsize=14, pad=15)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Hit Ratio@10', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{prefix}_hit_ratio.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 4: Combined Metrics
    plt.figure(figsize=(15, 10))
    metrics = {
        'NDCG@10': (val_ndcgs, test_ndcgs),
        'Hit Ratio@10': (val_hit_ratios, test_hit_ratios)
    }
    
    for i, (metric_name, (val_metric, test_metric)) in enumerate(metrics.items(), 1):
        plt.subplot(2, 1, i)
        plt.plot(epochs, val_metric, marker='o', linewidth=2, label='Validation')
        plt.plot(epochs, test_metric, marker='o', linewidth=2, label='Test')
        plt.title(f'{metric_name} per Epoch', fontsize=12, pad=10)
        plt.xlabel('Epoch', fontsize=10)
        plt.ylabel(metric_name, fontsize=10)
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{prefix}_combined_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Print best results
    best_epoch = np.argmax(val_ndcgs)
    print(f"\nBest Results (Epoch {best_epoch}):")
    print(f"Training Loss: {train_losses[best_epoch]:.4f}")
    print(f"Validation NDCG@10: {val_ndcgs[best_epoch]:.4f}")
    print(f"Test NDCG@10: {test_ndcgs[best_epoch]:.4f}")
    print(f"Validation Hit Ratio@10: {val_hit_ratios[best_epoch]:.4f}")
    print(f"Test Hit Ratio@10: {test_hit_ratios[best_epoch]:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser('Visualize TGN results')
    parser.add_argument('--prefix', type=str, required=True, help='Prefix of the saved results')
    args = parser.parse_args()
    
    results = load_epoch_results(args.prefix)
    plot_training_curves(results, args.prefix) 