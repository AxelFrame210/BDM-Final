import matplotlib.pyplot as plt
import numpy as np
import os

def plot_metric(epochs, val_values, test_values, metric_name, output_path):
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, val_values, 'b-', label=f'Validation {metric_name}')
    plt.plot(epochs, test_values, 'r--', label=f'Test {metric_name}')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} over Epochs')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_path}/{metric_name.lower().replace("@", "_at_")}.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_loss_and_lr(epochs, loss_values, lr_values, output_path):
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss_values, 'b-', label='Loss')
    plt.plot(epochs, lr_values, 'r--', label='Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Loss and Learning Rate over Epochs')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_path}/loss_and_lr.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Data
    epochs = np.arange(1, 12)
    loss = [10.0278, 9.5078, 9.1313, 8.9168, 8.8273, 8.6288, 8.2671, 7.9467, 7.6942, 7.5194, 7.4371]
    val_ndcg = [0.1896, 0.1748, 0.1630, 0.1583, 0.1391, 0.1243, 0.1386, 0.1237, 0.1169, 0.1222, 0.1228]
    test_ndcg = [0.1745, 0.1581, 0.1521, 0.1732, 0.1372, 0.1315, 0.1463, 0.1232, 0.1326, 0.1186, 0.1027]
    val_recall = [0.1577, 0.1422, 0.1279, 0.1246, 0.1036, 0.0882, 0.1025, 0.0871, 0.0794, 0.0849, 0.0849]
    test_recall = [0.1411, 0.1246, 0.1180, 0.1389, 0.1014, 0.0959, 0.1103, 0.0871, 0.0959, 0.0805, 0.0650]
    val_precision = [0.0158, 0.0142, 0.0128, 0.0125, 0.0104, 0.0088, 0.0103, 0.0087, 0.0079, 0.0085, 0.0085]
    test_precision = [0.0141, 0.0125, 0.0118, 0.0139, 0.0101, 0.0096, 0.0110, 0.0087, 0.0096, 0.0080, 0.0065]
    learning_rates = [0.000027, 0.000020, 0.000011, 0.000003, 0.000030, 0.000029, 0.000027, 0.000024, 0.000020, 0.000015, 0.000011]

    # Create output directory if it doesn't exist
    os.makedirs('results/plots', exist_ok=True)

    # Plot each metric separately
    plot_loss_and_lr(epochs, loss, learning_rates, 'results/plots')
    plot_metric(epochs, val_ndcg, test_ndcg, 'NDCG@10', 'results/plots')
    plot_metric(epochs, val_recall, test_recall, 'Recall@10', 'results/plots')
    plot_metric(epochs, val_precision, test_precision, 'Precision@10', 'results/plots')

if __name__ == '__main__':
    main() 