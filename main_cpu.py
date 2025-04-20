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
BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_EPOCH = args.n_epoch
NUM_HEADS = 2
DROP_OUT = 0.1
DATA = args.data
NUM_LAYER = 1
LEARNING_RATE = 0.0001
USE_MEMORY = True
MEMORY_DIM = args.memory_dim
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

optimizer = torch.optim.Adam(tgn.parameters(), lr=LEARNING_RATE)

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
            negative_batch = train_rand_sampler.sample(size=NUM_CANDIDATES)
      
            # flatten negative_batch
            negative_batch = np.array([x for y in negative_batch for x in y])

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
            neg_score = tgn.affinity_score(source_embedding.repeat(NUM_CANDIDATES, 1), neg_embedding).squeeze(dim=0)
            
            pos_score = pos_score.sigmoid()
            neg_score = neg_score.sigmoid()
            
            pos_label = torch.ones_like(pos_score)
            neg_label = torch.zeros_like(neg_score)
            
            loss_fn = torch.nn.BCELoss()
            loss += loss_fn(pos_score, pos_label)
            loss += loss_fn(neg_score, neg_label)

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

    print(f"Epoch {epoch}: train loss: {np.mean(losses_batch):.4f}, val ndcg@10: {val_score:.4f}, test ndcg@10: {test_ndcgs[2]:.4f}") 