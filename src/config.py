from dataclasses import dataclass
import torch

@dataclass
class UniGCRConfig:
    # --- 任务控制 ---
    enable_ctr: bool = True  # 新增：控制是否训练 CTR 任务
    
    # --- 数据路径 ---
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
    epochs: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    
    # --- Uni-GCR 特有参数 (仅在 enable_ctr=True 时生效) ---
    memory_bank_size: int = 20
    num_hard_negatives: int = 5
    temp: float = 0.07  
    loss_alpha: float = 1.0 # InfoNCE 权重
    loss_beta: float = 0.5  # BCE 权重
    
    # --- 动态参数 ---
    num_users: int = 0
    num_items: int = 0
