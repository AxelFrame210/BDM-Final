import matplotlib.pyplot as plt
import numpy as np
import os

# Create output directory
os.makedirs('results/plots', exist_ok=True)

# Data from detailed_results_summary.txt
epochs = np.arange(1, 12)
val_ndcg = [0.1896, 0.1748, 0.1630, 0.1583, 0.1391, 0.1243, 0.1386, 0.1237, 0.1169, 0.1222, 0.1228]
test_ndcg = [0.1745, 0.1581, 0.1521, 0.1732, 0.1372, 0.1315, 0.1463, 0.1232, 0.1326, 0.1186, 0.1027]
val_hr = [0.1577, 0.1422, 0.1279, 0.1246, 0.1036, 0.0882, 0.1025, 0.0871, 0.0794, 0.0849, 0.0849]
test_hr = [0.1411, 0.1246, 0.1180, 0.1389, 0.1014, 0.0959, 0.1103, 0.0871, 0.0959, 0.0805, 0.0650]

# Set style
plt.style.use('bmh')  # Using built-in style
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.size'] = 12

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

# Plot NDCG@10
ax1.plot(epochs, val_ndcg, 'b-', marker='o', linewidth=2, label='Validation')
ax1.plot(epochs, test_ndcg, 'r--', marker='s', linewidth=2, label='Test')
ax1.set_title('NDCG@10 over Epochs', pad=15)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('NDCG@10')
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.set_ylim(0.05, 0.25)  # Set y-axis limits to better show the range

# Plot Hit Ratio@10
ax2.plot(epochs, val_hr, 'b-', marker='o', linewidth=2, label='Validation')
ax2.plot(epochs, test_hr, 'r--', marker='s', linewidth=2, label='Test')
ax2.set_title('Hit Ratio@10 over Epochs', pad=15)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Hit Ratio@10')
ax2.grid(True, alpha=0.3)
ax2.legend()
ax2.set_ylim(0.05, 0.20)  # Set y-axis limits to better show the range

# Add annotations for best performance
ax1.annotate(f'Best: {max(val_ndcg):.4f}', 
            xy=(epochs[val_ndcg.index(max(val_ndcg))], max(val_ndcg)),
            xytext=(10, 10), textcoords='offset points')
ax2.annotate(f'Best: {max(val_hr):.4f}', 
            xy=(epochs[val_hr.index(max(val_hr))], max(val_hr)),
            xytext=(10, 10), textcoords='offset points')

# Adjust layout and save
plt.tight_layout()
plt.savefig('results/plots/metrics_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# Create individual plots for each metric
# NDCG@10
plt.figure(figsize=(10, 6))
plt.plot(epochs, val_ndcg, 'b-', marker='o', linewidth=2, label='Validation')
plt.plot(epochs, test_ndcg, 'r--', marker='s', linewidth=2, label='Test')
plt.title('NDCG@10 over Epochs', pad=15)
plt.xlabel('Epoch')
plt.ylabel('NDCG@10')
plt.grid(True, alpha=0.3)
plt.legend()
plt.ylim(0.05, 0.25)
plt.tight_layout()
plt.savefig('results/plots/ndcg_plot.png', dpi=300, bbox_inches='tight')
plt.close()

# Hit Ratio@10
plt.figure(figsize=(10, 6))
plt.plot(epochs, val_hr, 'b-', marker='o', linewidth=2, label='Validation')
plt.plot(epochs, test_hr, 'r--', marker='s', linewidth=2, label='Test')
plt.title('Hit Ratio@10 over Epochs', pad=15)
plt.xlabel('Epoch')
plt.ylabel('Hit Ratio@10')
plt.grid(True, alpha=0.3)
plt.legend()
plt.ylim(0.05, 0.20)
plt.tight_layout()
plt.savefig('results/plots/hit_ratio_plot.png', dpi=300, bbox_inches='tight')
plt.close()

print("Graphs have been generated and saved to results/plots/") 