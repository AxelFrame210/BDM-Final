import math
import logging
import time
import sys
import argparse
import torch
import numpy as np
import pickle
from pathlib import Path
import os
import gc
from tqdm import tqdm
import json
from model.tgn_cpu import TGN_CPU
from evaluation.evaluation import eval_recommendation
from utils.data import get_data, compute_time_statistics
from utils.utils import EarlyStopMonitor, RandEdgeSampler, get_neighbor_finder
torch.manual_seed(0)
np.random.seed(0)
torch.autograd.set_detect_anomaly(True)  # Enable anomaly detection

"""
argument
"""
# setting
parser = argparse.ArgumentParser('TGN recommender training')
parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. wikipedia or reddit)', default='transaction')
parser.add_argument('--prefix', type=str, default='', help='Prefix to name the checkpoints')
# model 
parser.add_argument('--memory_dim', type=int, default=64, help='Dimensions of the memory for each user')
parser.add_argument('--n_degree', type=int, default=10, help='Number of neighbors to sample')
parser.add_argument('--embedding_module', type=str, default="graph_attention", choices=["graph_attention", "graph_ngcf", "graph_sum", "identity", "time"], help='Type of embedding module')
parser.add_argument('--memory_updater', type=str, default="gru", choices=["gru", "rnn"], help='Type of memory updater')
parser.add_argument('--dyrep', action='store_true', help='Whether to run the dyrep model')
parser.add_argument('--use_destination_embedding_in_message', action='store_true', help='Whether to use the embedding of the destination node as part of the message')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
# training
parser.add_argument('--n_epoch', type=int, default=2, help='Number of epochs')
parser.add_argument('--bs', type=int, default=1000, help='Batch_size')
parser.add_argument('--num_candidates', type=int, default=3, help='*part of batch items')
parser.add_argument('--num_neg_train', type=int, default=5, help='*p_pos and p_neg items')
parser.add_argument('--test_run', action='store_true', help='*run only first two batches')
parser.add_argument('--use_memory', action='store_true', help='Whether to augment the model with a node memory')
# evaluation
parser.add_argument('--in_sample', action='store_true', help='*Whether to use in-sample setting for evaluation')
parser.add_argument('--num_neg_eval', type=int, default=100, help='*neg items for evaluation')
parser.add_argument('--num_rec', type=int, default=3, help='*top k items for evaluation')
args = parser.parse_args()

"""
global variables
"""
BATCH_SIZE = 128
NUM_NEIGHBORS = args.n_degree
NUM_EPOCH = args.n_epoch
NUM_HEADS = 8
DROP_OUT = args.dropout
DATA = args.data
NUM_LAYER = 1
LEARNING_RATE = 0.001
USE_MEMORY = True
MEMORY_DIM = 128
MESSAGE_DIM = 100
NUM_CANDIDATES = args.num_candidates
NUM_NEG_TRAIN = args.num_neg_train
NUM_NEG_EVAL = args.num_neg_eval
NUM_REC = args.num_rec
AGGREGATOR = 'last'
MESSAGE_FUNCTION = 'identity'
MEMORY_UPDATE_AT_END = False
MEMORY_UPDATE_AT_START = True
PATIENCE = 5
BACKPROP_EVERY = 1
UNIFORM = False
USE_SOURCE_EMBEDDING_IN_MESSAGE = False

print(args.prefix)

"""
save paths
"""
Path("results/").mkdir(parents=True, exist_ok=True)
Path("saved/").mkdir(parents=True, exist_ok=True)
get_checkpoint_path = lambda epoch: f'./saved/{args.prefix}_{epoch}.pth'

"""
data
""" 
node_features, edge_features, full_data, train_data, val_data, test_data = get_data(DATA, MEMORY_DIM)

"""
init
"""
# Initialize neighbor finder to retrieve temporal graph
train_ngh_finder = get_neighbor_finder(train_data, UNIFORM)
full_ngh_finder = get_neighbor_finder(full_data, UNIFORM)

# Compute time statistics
mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = \
  compute_time_statistics(full_data.sources, full_data.destinations, full_data.timestamps)

# Initialize Model
tgn = TGN_CPU(neighbor_finder=train_ngh_finder, node_features=node_features,
          edge_features=edge_features,
          n_layers=NUM_LAYER,
          n_heads=NUM_HEADS, dropout=DROP_OUT, use_memory=USE_MEMORY,
          message_dimension=MESSAGE_DIM, memory_dimension=MEMORY_DIM,
          memory_update_at_start=MEMORY_UPDATE_AT_START,
          embedding_module_type=args.embedding_module,
          message_function=MESSAGE_FUNCTION,
          aggregator_type=AGGREGATOR,
          memory_updater_type=args.memory_updater,
          n_neighbors=NUM_NEIGHBORS,
          mean_time_shift_src=mean_time_shift_src, std_time_shift_src=std_time_shift_src,
          mean_time_shift_dst=mean_time_shift_dst, std_time_shift_dst=std_time_shift_dst,
          use_destination_embedding_in_message=args.use_destination_embedding_in_message,
          use_source_embedding_in_message=USE_SOURCE_EMBEDDING_IN_MESSAGE,
          dyrep=args.dyrep)

# Update loss function
class WeightedBCELoss(torch.nn.Module):
    def __init__(self, pos_weight=2.0):
        super().__init__()
        self.pos_weight = pos_weight
        
    def forward(self, pred, target):
        # Add label smoothing
        target = target * 0.9 + 0.05
        
        # Calculate weighted BCE loss
        loss = -(self.pos_weight * target * torch.log(torch.sigmoid(pred) + 1e-10) + 
                (1 - target) * torch.log(1 - torch.sigmoid(pred) + 1e-10))
        return loss.mean()

# Initialize loss and optimizer
criterion = WeightedBCELoss(pos_weight=2.0)
optimizer = torch.optim.AdamW(tgn.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

num_instance = len(train_data.sources)
num_batch = math.ceil(num_instance / BATCH_SIZE)

"""
epoch loop
"""
early_stopper = EarlyStopMonitor(max_round=PATIENCE)
best_val_score = 0

for epoch in tqdm(range(NUM_EPOCH), desc="Progress: Epoch Loop"):  
    start_epoch = time.time()
  
    """
    Train
    """
    # Reinitialize memory of the model at the start of each epoch
    if USE_MEMORY:
        tgn.memory.__init_memory__()

    # Train using only training graph
    tgn.set_neighbor_finder(train_ngh_finder)

    """
    batch loop
    """
    losses_batch = []

    for batch in tqdm(range(0, num_batch, BACKPROP_EVERY), total=num_batch//BACKPROP_EVERY, desc="Progress: Train Batch Loop"):
        # test run
        if args.test_run and batch == 2:
            break

        loss = 0
        optimizer.zero_grad()

        # Custom loop to allow to perform backpropagation only every certain number of batches
        for j in range(BACKPROP_EVERY):
            batch_idx = batch + j

            if batch_idx >= num_batch:
                continue

            s_idx = batch_idx * BATCH_SIZE
            e_idx = min(num_instance, s_idx + BATCH_SIZE)
      
            sources_batch = train_data.sources[s_idx:e_idx]
            destinations_batch = train_data.destinations[s_idx:e_idx]
            edge_idxs_batch = train_data.edge_idxs[s_idx: e_idx]
            timestamps_batch = train_data.timestamps[s_idx:e_idx]
      
            # candidate sampling
            train_rand_sampler = RandEdgeSampler(sources_batch, destinations_batch)
            negative_batch = train_rand_sampler.sample(size=max(NUM_CANDIDATES, len(sources_batch)))
            
            """
            compute embeddings
            """
            tgn.train()
            source_embedding, destination_embedding, neg_embedding = tgn.compute_temporal_embeddings(
                sources_batch,
                destinations_batch,
                negative_batch,
                timestamps_batch,
                edge_idxs_batch,
                NUM_NEIGHBORS
            )

            """
            compute loss
            """
            pos_score = tgn.affinity_score(source_embedding, destination_embedding).squeeze(dim=0)
            n_neg = len(negative_batch) // len(sources_batch)
            neg_score = tgn.affinity_score(source_embedding.repeat(n_neg, 1), neg_embedding).squeeze(dim=0)
            
            pos_score = pos_score.sigmoid()
            neg_score = neg_score.sigmoid()
            
            pos_label = torch.ones_like(pos_score)
            neg_label = torch.zeros_like(neg_score)
            
            loss += criterion(pos_score, pos_label)
            loss += criterion(neg_score, neg_label)

        loss /= BACKPROP_EVERY
        loss.backward(retain_graph=True)
        optimizer.step()
        losses_batch.append(loss.item())

    """
    Validation
    """
    val_metrics = eval_recommendation(tgn, val_data, BATCH_SIZE, NUM_NEIGHBORS, NUM_NEG_EVAL, args.test_run)
    test_metrics = eval_recommendation(tgn, test_data, BATCH_SIZE, NUM_NEIGHBORS, NUM_NEG_EVAL, args.test_run)

    """
    Save results
    """
    # Save metrics
    pickle.dump({
        "val": val_metrics,
        "test": test_metrics,
        "epoch": epoch,
        "loss": np.mean(losses_batch),
    }, open(f"results/{args.prefix}_epoch_{epoch}_eval.pkl", "wb"))

    # Save temporal embeddings
    tgn.eval()
    with torch.no_grad():
        source_embedding, destination_embedding, _ = tgn.compute_temporal_embeddings(
            train_data.sources,
            train_data.destinations,
            [],  # No negative samples
            train_data.timestamps,
            train_data.edge_idxs,
            NUM_NEIGHBORS
        )
        train_embeddings = {
            'user_indices': train_data.sources,
            'item_indices': train_data.destinations,
            'labels': np.ones(len(train_data.sources)),  # Positive interactions
            'temporal_embeddings': torch.cat([source_embedding, destination_embedding], dim=1).cpu().numpy()
        }
        
        source_embedding, destination_embedding, _ = tgn.compute_temporal_embeddings(
            val_data.sources,
            val_data.destinations,
            [],  # No negative samples
            val_data.timestamps,
            val_data.edge_idxs,
            NUM_NEIGHBORS
        )
        val_embeddings = {
            'user_indices': val_data.sources,
            'item_indices': val_data.destinations,
            'labels': np.ones(len(val_data.sources)),  # Positive interactions
            'temporal_embeddings': torch.cat([source_embedding, destination_embedding], dim=1).cpu().numpy()
        }
        
        source_embedding, destination_embedding, _ = tgn.compute_temporal_embeddings(
            test_data.sources,
            test_data.destinations,
            [],  # No negative samples
            test_data.timestamps,
            test_data.edge_idxs,
            NUM_NEIGHBORS
        )
        test_embeddings = {
            'user_indices': test_data.sources,
            'item_indices': test_data.destinations,
            'labels': np.ones(len(test_data.sources)),  # Positive interactions
            'temporal_embeddings': torch.cat([source_embedding, destination_embedding], dim=1).cpu().numpy()
        }
        
        pickle.dump({
            'train': train_embeddings,
            'val': val_embeddings,
            'test': test_embeddings
        }, open(f"results/{args.data}_tgn_embeddings.pkl", "wb"))

    # Save model
    torch.save(tgn.state_dict(), get_checkpoint_path(epoch))

    # Early stopping
    val_ndcgs = np.mean(val_metrics['ndcgs'], axis=0)
    test_ndcgs = np.mean(test_metrics['ndcgs'], axis=0)
    val_score = val_ndcgs[2]  # NDCG@10 is at index 2 ([1, 5, 10, 20])
    
    if val_score > best_val_score:
        best_val_score = val_score
        best_epoch = epoch
        torch.save(tgn.state_dict(), get_checkpoint_path("best"))

    if early_stopper.early_stop_check(val_score):
        print("Early stopping triggered")
        break

    print(f'Epoch {epoch+1}: train loss: {np.mean(losses_batch):.4f}, '
          f'val ndcg@10: {val_metrics["ndcgs"][0][2]:.4f}, test ndcg@10: {test_metrics["ndcgs"][0][2]:.4f}, '
          f'val hr@10: {val_metrics["hits"][0][2]:.4f}, test hr@10: {test_metrics["hits"][0][2]:.4f}') 