import torch
import numpy as np
import random
import os

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_metric(pred_list, topk=10):
    """
    pred_list: list of tensors, 每个tensor包含预测的TopK item index
    target_list: list of scalars, 每个scalar是真实的item index
    """
    NDCG = []
    HIT = []
    
    for pred, target in pred_list:
        # pred: (K), target: scalar
        hit_mask = (pred == target)
        
        if hit_mask.sum() > 0:
            HIT.append(1.0)
            rank = hit_mask.nonzero(as_tuple=True)[0].item()
            NDCG.append(1.0 / np.log2(rank + 2.0))
        else:
            HIT.append(0.0)
            NDCG.append(0.0)
            
    return np.mean(HIT), np.mean(NDCG)
