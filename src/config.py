from dataclasses import dataclass, field
from typing import List
import torch

@dataclass
class UniGCRConfig:
    # --- 任务开关 ---
    enable_ctr: bool = True
    patience: int = 3
    
    # --- 数据特征配置 (新加) ---
    # 类别特征的词表大小列表。例如: [User_ID_Vocab, Gender_Vocab, Region_Vocab]
    # 这里假设有3个类别特征，词表大小分别为 1000, 20, 50
    cat_feature_vocab_sizes: List[int] = field(default_factory=lambda: [1000, 20, 50])
    
    # 数值特征的数量。例如: [Age, ReviewCount, AvgRating] -> 3个
    num_feature_size: int = 3
    
    # --- 路径与基础配置 ---
    data_path: str = "data/reviews_Beauty_5.json.gz"
    min_seq_len: int = 5
    max_seq_len: int = 50
    
    # --- 模型参数 ---
    embed_dim: int = 64
    hstu_layers: int = 2
    hstu_heads: int = 2
    dropout: float = 0.1
    
    # --- 训练参数 ---
    batch_size: int = 64
    lr: float = 1e-3
    epochs: int = 50
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    
    # --- Uni-GCR 参数 ---
    memory_bank_size: int = 20
    num_hard_negatives: int = 5
    temp: float = 0.07  
    loss_alpha: float = 1.0 
    loss_beta: float = 0.5 
    
    # --- 动态更新 ---
    num_items: int = 0 # Item ID 的词表大小
