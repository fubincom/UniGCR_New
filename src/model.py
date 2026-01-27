import torch
import torch.nn as nn

# 尝试导入官方实现
try:
    from generative_recommenders.modeling.sequential.hstu import HSTU as OfficialHSTU
except ImportError:
    OfficialHSTU = None

class FeatureTokenizer(nn.Module):
    """
    将非序列特征（Categorical + Numerical）转换为 Transformer 可理解的 Token Embeddings。
    """
    def __init__(self, cat_vocab_sizes, num_feature_size, embed_dim, dropout=0.1):
        super().__init__()
        self.num_cat = len(cat_vocab_sizes)
        self.num_num = num_feature_size
        
        # 1. 类别特征 Embeddings
        # ModuleList 对应每一个 categorical feature 的 embedding table
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, embed_dim) for vocab_size in cat_vocab_sizes
        ])
        
        # 2. 数值特征 Projection
        # 将每个数值标量投影到 embed_dim 维度
        if num_feature_size > 0:
            # 方式A: 共享投影层 (所有数值特征用同一个 Linear)
            # 方式B: 独立投影层 (每个数值特征有自己的 Linear) -> 这里选 B，更灵活
            self.num_projections = nn.ModuleList([
                nn.Linear(1, embed_dim) for _ in range(num_feature_size)
            ])
        
        # 3. 特征变换层 (MLP + GELU)
        # 按照需求，将 Embedding 后的结果再过一层 MLP，使其分布适应 HSTU
        self.feature_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, cat_inputs, num_inputs):
        """
        cat_inputs: (B, Num_Cats)
        num_inputs: (B, Num_Nums)
        Return: (B, Num_Cats + Num_Nums, Embed_Dim)
        """
        tokens = []
        
        # 处理类别特征
        for i, emb_layer in enumerate(self.cat_embeddings):
            # cat_inputs[:, i] shape: (B,) -> emb -> (B, D) -> unsqueeze -> (B, 1, D)
            val = cat_inputs[:, i]
            emb = emb_layer(val).unsqueeze(1)
            tokens.append(emb)
            
        # 处理数值特征
        if self.num_num > 0:
            for i, proj_layer in enumerate(self.num_projections):
                # num_inputs[:, i] shape: (B,) -> unsqueeze -> (B, 1) -> proj -> (B, D) -> unsqueeze -> (B, 1, D)
                val = num_inputs[:, i].unsqueeze(-1)
                emb = proj_layer(val).unsqueeze(1)
                tokens.append(emb)
        
        if len(tokens) == 0:
            return None
            
        # 拼接所有特征 Token: (B, Total_Feats, D)
        concat_tokens = torch.cat(tokens, dim=1)
        
        # MLP 变换
        out = self.feature_mlp(concat_tokens)
        
        return out

class UniGCRModel(nn.Module):
    def __init__(self, config: UniGCRConfig):
        super().__init__()
        self.config = config
        
        if OfficialHSTU is None:
            raise RuntimeError("generative-recommenders not installed. Cannot use official HSTU.")

        # 1. Embeddings & Feature Tokenizer (保持不变)
        self.item_emb = nn.Embedding(config.num_items + 1, config.embed_dim, padding_idx=0)
        self.feature_tokenizer = FeatureTokenizer(
            cat_vocab_sizes=config.cat_feature_vocab_sizes,
            num_feature_size=config.num_feature_size,
            embed_dim=config.embed_dim,
            dropout=config.dropout
        )
        
        # 2. Backbone: 使用官方 HSTU
        # 转换配置
        hstu_config = config.to_hstu_config()
        self.backbone = OfficialHSTU(
            config=hstu_config,
            # 官方库通常需要传入 embedding_module 来做 tie_weights，
            # 但我们这里只用它做 Encoder，所以可以不传或者传 None
            embedding_module=None 
        )
        
        # 3. Heads (GR & CTR) (保持不变)
        self.gr_head = nn.Linear(config.embed_dim, config.num_items + 1)
        
        self.cand_proj = nn.Sequential(
            nn.LayerNorm(config.embed_dim),
            nn.Linear(config.embed_dim, config.embed_dim)
        )
        self.cross_attn = nn.MultiheadAttention(config.embed_dim, num_heads=2, batch_first=True)
        self.scorer = nn.Sequential(
            nn.Linear(config.embed_dim * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def get_user_state(self, history, cat_feats, num_feats):
        # 1. Prepare Embeddings (Prefix Sharing)
        seq_emb = self.item_emb(history) # (B, T, D)
        prefix_emb = self.feature_tokenizer(cat_feats, num_feats) # (B, P, D)
        
        if prefix_emb is not None:
            x = torch.cat([prefix_emb, seq_emb], dim=1)
        else:
            x = seq_emb
            
        # 计算有效长度 (非 Padding 部分)
        # 假设 Item ID = 0 是 Padding
        # Prefix 部分视为全有效
        prefix_len = prefix_emb.shape[1]
        hist_valid_mask = (history != 0) # (B, T_hist)
        hist_lens = hist_valid_mask.sum(dim=1) # (B,)
        total_lens = prefix_len + hist_lens # (B,)
        
        # --- Packing (转为 Jagged) ---
        # 这种方式能最大化官方 HSTU 的性能
        valid_tokens_list = []
        for i in range(x.size(0)):
            # 截取有效部分
            valid_len = total_lens[i]
            # 注意 x 已经是 concat 过的，前面是 prefix，后面是 history
            # 但 history 可能中间夹杂 padding (如果是 left padding)，或者右边 padding
            # 这里假设是 Right Padding
            valid_tokens_list.append(x[i, :valid_len, :])
            
        packed_x = torch.cat(valid_tokens_list, dim=0) # (Sum_Lens, D)
        lengths = total_lens.to(torch.int64)
        
        # 调用官方
        # 官方 HSTU 通常接受 (all_tokens, lengths)
        out_packed = self.backbone(packed_x, lengths)
        
        # --- Unpacking (还原回 Batch 取最后一个) ---
        # 我们只需要每条样本的最后一个 Token
        # 计算每个样本在 packed 序列中的结束位置索引
        offsets = torch.cumsum(lengths, dim=0)
        end_indices = offsets - 1
        
        u = out_packed[end_indices] # (B, D)
        
        return u

    def predict_ctr(self, u, targets, gr_logits, device):
        # (代码同上一版，无需修改，因为 u 已经包含了 side info 的信息)
        B = u.size(0)
        N = self.config.memory_bank_size
        K = self.config.num_hard_negatives
        gt_emb = self.item_emb(targets).unsqueeze(1)
        _, top_k = torch.topk(gr_logits.detach(), K, dim=1)
        hard_emb = self.item_emb(top_k)
        num_easy = N - 1 - K
        rand_ids = torch.randint(1, self.config.num_items, (B, num_easy)).to(device)
        easy_emb = self.item_emb(rand_ids)
        bank = torch.cat([gt_emb, hard_emb, easy_emb], dim=1)
        labels = torch.zeros((B, N), device=device); labels[:, 0] = 1.0
        idx = torch.randperm(N).to(device)
        bank = bank[:, idx, :]; labels = labels[:, idx]
        bank_proj = self.cand_proj(bank)
        u_q = u.unsqueeze(1)
        ctx, _ = self.cross_attn(u_q, bank_proj, bank_proj)
        u_exp = u.unsqueeze(1).repeat(1, N, 1); ctx_exp = ctx.repeat(1, N, 1)
        feat = torch.cat([bank_proj, u_exp, ctx_exp], dim=-1)
        return self.scorer(feat).squeeze(-1), labels
    
    def forward(self, history, cat_feats, num_feats):
        u = self.get_user_state(history, cat_feats, num_feats)
        gr_logits = self.gr_head(u)
        return u, gr_logits
