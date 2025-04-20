import math
import random
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import average_precision_score, roc_auc_score
from utils.utils import EarlyStopMonitor, RandEdgeSampler, get_neighbor_finder


def recall_at_k(recommendations, test_items, k):
    hits = len(set(recommendations[:k]) & set(test_items))
    return hits / min(k, len(test_items))

def ndcg_at_k(recommendations, test_items, k):
    dcg = 0
    idcg = sum([1 / np.log2(i + 2) for i in range(min(k, len(test_items)))])
    for i, item in enumerate(recommendations[:k]):
        if item in test_items:
            dcg += 1 / np.log2(i + 2)
    return dcg / idcg

def MRR_at_k(recommendations, test_items, k):
    for i, item in enumerate(recommendations[:k]):
        if item in test_items:
            return 1 / (i + 1)
    return 0

def Hit_at_k(recommendations, test_items, k):
    for i, item in enumerate(recommendations[:k]):
        if item in test_items:
            return 1
    return 0
  
def Precision_at_k(recommendations, test_items, k):
    hits = len(set(recommendations[:k]) & set(test_items))
    return hits / k

def eval_recommendation(tgn, data, batch_size, n_neighbors, num_neg_eval, is_test_run):
    with torch.no_grad():
        tgn.eval()
        
        # Initialize metrics
        recalls = []
        ndcgs = []
        mrrs = []
        hits = []
        precisions = []
        
        num_instance = len(data.sources)
        num_batch = math.ceil(num_instance / batch_size)
        
        for batch in tqdm(range(num_batch), desc="Progress: Eval Batch"):
            if is_test_run and batch == 2:
                break

            s_idx = batch * batch_size
            e_idx = min(num_instance, s_idx + batch_size)

            sources_batch = data.sources[s_idx:e_idx]
            destinations_batch = data.destinations[s_idx:e_idx]
            timestamps_batch = data.timestamps[s_idx:e_idx]
            edge_idxs_batch = data.edge_idxs[s_idx:e_idx]

            test_rand_sampler = RandEdgeSampler(sources_batch, destinations_batch, seed=2023)
            negatives_batch = test_rand_sampler.sample(size=num_neg_eval * len(sources_batch))

            source_embedding, destination_embedding, negative_embedding = tgn.compute_temporal_embeddings_eval(
                sources_batch,
                destinations_batch,
                negatives_batch,
                timestamps_batch,
                edge_idxs_batch,
                n_neighbors,
                num_neg_eval
            )

            bsbs = len(sources_batch)

            source_embedding = source_embedding.view(bsbs, 1, -1)
            destination_embedding = destination_embedding.view(bsbs, 1, -1)
            negative_embedding = negative_embedding.view(bsbs, num_neg_eval, -1)

            pos_scores = torch.sum(source_embedding * destination_embedding, dim=2).cpu().numpy()
            neg_scores = torch.sum(source_embedding * negative_embedding, dim=2).cpu().numpy()

            for i in range(bsbs):
                pos_score = pos_scores[i]
                neg_score = neg_scores[i]

                scores = np.concatenate((pos_score, neg_score))
                ranking = np.argsort(scores)[::-1]

                pos_ranking = [0]
                topk = [1, 5, 10, 20]
                recall = [recall_at_k(ranking, pos_ranking, top) for top in topk]
                ndcg = [ndcg_at_k(ranking, pos_ranking, top) for top in topk]
                mrr = [MRR_at_k(ranking, pos_ranking, top) for top in topk]
                hit = [Hit_at_k(ranking, pos_ranking, top) for top in topk]
                precision = [Precision_at_k(ranking, pos_ranking, top) for top in topk]

                recalls.append(recall)
                ndcgs.append(ndcg)
                mrrs.append(mrr)
                hits.append(hit)
                precisions.append(precision)

        metrics = {
            'recalls': recalls,
            'ndcgs': ndcgs,
            'mrrs': mrrs,
            'hits': hits,
            'precisions': precisions
        }

        return metrics

    
    
    
    
    
    # source_nodes = data.sources
    # destination_nodes = data.destinations
    # size = len(source_nodes)
    # _, negative_nodes = negative_edge_sampler.sample(size)
    # edge_times = data.timestamps
    # edge_idxs = data.edge_idxs
    
    # """
    # node embedding 생성
    # """
    # source_embedding, destination_embedding, _ = tgn.compute_temporal_embeddings(source_nodes,
    #                                                                               destination_nodes,
    #                                                                               negative_nodes,
    #                                                                               edge_times,
    #                                                                               edge_idxs,
    #                                                                               n_neighbors)
    
    # """
    # 유저마다 user_purchase_history 생성
    # """
    
    # # create a dict 'user_buy_dict' where keys are unique source_nodes and values are lists of destination_nodes that the source_nodes have purchased
    # source_nodes_set = np.unique(source_nodes)
    # destination_nodes_set = np.unique(destination_nodes)
    # user_buy_dict = {source_node: destination_nodes[source_nodes == source_node] for source_node in source_nodes_set}
    
    # """
    # 유저 loop 돌면서 평가
    # """
    
    # # print('Evaluation Start, num of users: ', len(user_buy_dict), len(source_nodes_set))
    # # print('Evaluation Start, num of items: ', len(destination_nodes_set))
    # sum_recall = 0.0
    # sum_ndcg = 0.0
    # sum_mrr = 0.0
    # sum_hit = 0.0
    # sum_precision = 0.0
    # total_user = 0
    
    # for user, pos_items in user_buy_dict.items():
      
    #   """
    #   예시
    #   user:  1                                                      # numpy.int64
    #   pos_items:  [274 274 274 274 274 274 274 274 274 274 274 274] # numpy.ndarray
    #   neg_items = [517]                                             # numpy.ndarray
    #   """
      
    #   # pos_items 없는 유저는 평가에서 제외
    #   if len(pos_items) == 0:
    #     continue
      
    #   neg_items = np.setdiff1d(destination_nodes_set, pos_items)
    #   # neg_items 100개 미만인 유저는 평가에서 제외
    #   if len(neg_items) < 100:
    #     continue
    #   neg_items = random.sample(list(neg_items), 100)
      
    #   user_tensor = torch.LongTensor([user]).to(tgn.device)
    #   pos_tensor = torch.LongTensor(pos_items).to(tgn.device)
    #   neg_tensor = torch.LongTensor(neg_items).to(tgn.device)
      
    #   user_emb = source_embedding[user_tensor]
    #   pos_emb = destination_embedding[pos_tensor]
    #   neg_emb = destination_embedding[neg_tensor]
      
    #   pos_scores = torch.sum(user_emb * pos_emb, dim=1)
    #   neg_scores = torch.sum(user_emb * neg_emb, dim=1)
      
    #   k = 10
    #   ranking = torch.argsort(torch.cat([pos_scores.flatten(), neg_scores.flatten()]), descending=True).cpu().numpy().tolist()
    #   pos_ranking = [i for i in range(len(pos_scores))]
      
    #   recall = recall_at_k(ranking, pos_ranking, k)
    #   ndcg = ndcg_at_k(ranking, pos_ranking, k)
    #   mrr = MRR_at_k(ranking, pos_ranking, k)
    #   hit = Hit_at_k(ranking, pos_ranking, k)
    #   precision = Precision_at_k(ranking, pos_ranking, k)

    #   sum_recall += recall   
    #   sum_ndcg += ndcg
    #   sum_mrr += mrr
    #   sum_hit += hit
    #   sum_precision += precision
    #   total_user += 1
      
    # return sum_recall/total_user, sum_ndcg/total_user, sum_mrr/total_user, sum_hit/total_user, sum_precision/total_user