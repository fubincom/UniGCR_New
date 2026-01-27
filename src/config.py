from dataclasses import dataclass, field
from typing import List
import torch

# 引用官方 Config (如果安装成功)
try:
    from generative_recommenders.modeling.sequential.hstu import HSTUConfig
except ImportError:
    HSTUConfig = None
    print("Warning: generative_recommenders package not found.")

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

    # --- 模型核心参数 (需与 HSTU 对齐) ---
    embed_dim: int = 64
    hstu_layers: int = 2
    hstu_heads: int = 2
    max_seq_len: int = 50
    dropout: float = 0.1
    
    # 官方 HSTU 特有参数 (通常保持默认即可)
    attn_alpha: float = 1.0     # 线性 Attention 的放缩系数
    max_attn_len: int = 4096    # 算子支持的最大长度
    
    def to_hstu_config(self):
        """将 UniGCR 配置转换为官方 HSTUConfig"""
        if HSTUConfig is None:
            raise ImportError("Please install generative_recommenders first.")
            
        return HSTUConfig(
            embedding_dim=self.embed_dim,
            num_heads=self.hstu_heads,
            num_blocks=self.hstu_layers,
            dropout_rate=self.dropout,
            # 下面是 HSTU 论文中的默认推荐配置
            linear_dropout_rate=0.0,
            attn_dropout_rate=0.0,
            forward_dropout_rate=self.dropout,
            normalization="layer_norm", # 或 rms_norm
            activation="silu",          # HSTU 标配 SiLU
            max_seq_len=self.max_seq_len,
            # 这里的 alpha 对应论文中的 Linear Attention 缩放
            attn_alpha=self.attn_alpha 
        )
