import torch
import torch.nn as nn

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

class HSTU(nn.Module):
    def __init__(self, dim, heads, layers, dropout):
        super().__init__()
        # 使用 PyTorch 原生 TransformerEncoder
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=dim*4, dropout=dropout, batch_first=True, norm_first=True)
            for _ in range(layers)
        ])
        self.ln = nn.LayerNorm(dim)

    def forward(self, x, mask=None):
        for block in self.blocks:
            x = block(x, src_mask=mask)
        return self.ln(x)

class UniGCRModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Item Embedding
        self.item_emb = nn.Embedding(config.num_items + 1, config.embed_dim, padding_idx=0)
        
        # --- 新增: 特征 Tokenization 层 ---
        self.feature_tokenizer = FeatureTokenizer(
            cat_vocab_sizes=config.cat_feature_vocab_sizes,
            num_feature_size=config.num_feature_size,
            embed_dim=config.embed_dim,
            dropout=config.dropout
        )
        
        # Backbone
        self.encoder = HSTU(config.embed_dim, config.hstu_heads, config.hstu_layers, config.dropout)
        
        # Heads
        self.gr_head = nn.Linear(config.embed_dim, config.num_items + 1)
        
        # CTR Components
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
        """
        Prefix Sharing: [Feature_Tokens, Sequence_Tokens] -> HSTU
        """
        # 1. Item Sequence Embeddings
        seq_emb = self.item_emb(history) # (B, Seq_Len, D)
        
        # 2. Side Feature Embeddings (Prefix)
        # (B, Num_Feats, D)
        prefix_emb = self.feature_tokenizer(cat_feats, num_feats)
        
        # 3. Prefix Sharing Concatenation
        if prefix_emb is not None:
            x = torch.cat([prefix_emb, seq_emb], dim=1) # (B, N_Feat + Seq_Len, D)
        else:
            x = seq_emb
            
        # 4. Causal Mask Construction
        # 我们的 Prefix 特征之间通常是全连接可见的（双向），或者也遵循因果律（单向）
        # 通常推荐系统中，Profile 特征对所有历史行为可见，历史行为对 Profile 可见。
        # 最简单做法：全 Causual Mask。即 Feature1 看不到 Feature2，Feature2 看不到 Item1...
        # 更好做法：Prefix 部分互相可见，Sequence 部分 Causal。
        # 这里为了兼容 HSTU/GPT 的标准实现，我们采用标准的 Causal Mask。
        sz = x.size(1)
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1).to(x.device)
        
        # 5. Encoder
        out = self.encoder(x, mask)
        
        # 6. 取最后一个 Token 作为 User State u
        return out[:, -1, :] 

    # ... (predict_ctr 等方法保持不变，因为它们只依赖 u) ...
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
