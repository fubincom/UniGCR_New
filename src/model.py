import torch
import torch.nn as nn

class HSTU(nn.Module):
    def __init__(self, dim, heads, layers, dropout):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=dim*4, dropout=dropout, batch_first=True)
            for _ in range(layers)
        ])
        self.ln = nn.LayerNorm(dim)

    def forward(self, x, mask=None):
        # PyTorch Transformer 需要 mask
        for block in self.blocks:
            x = block(x, src_mask=mask)
        return self.ln(x)

class UniGCRModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.item_emb = nn.Embedding(config.num_items + 1, config.embed_dim, padding_idx=0)
        self.user_emb = nn.Embedding(config.num_users + 1, config.embed_dim)
        
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

    def get_user_state(self, history, user_id):
        # 1. Embedding
        seq_emb = self.item_emb(history) 
        u_emb = self.user_emb(user_id).unsqueeze(1)
        
        # 2. Prefix Sharing
        x = torch.cat([u_emb, seq_emb], dim=1)
        
        # 3. Causal Mask
        sz = x.size(1)
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1).to(x.device)
        
        # 4. Encoder
        out = self.encoder(x, mask)
        return out[:, -1, :] # User embedding u

    def forward(self, history, user_id):
        """仅返回 User State 和 GR Logits，用于 Training 和 Inference"""
        u = self.get_user_state(history, user_id)
        gr_logits = self.gr_head(u)
        return u, gr_logits

    def predict_ctr(self, u, targets, gr_logits, device):
        """专门用于训练时的 CTR 分支 forward"""
        # 构建 Memory Bank
        B = u.size(0)
        N = self.config.memory_bank_size
        K = self.config.num_hard_negatives
        
        # GT & Hard Neg & Easy Neg logic (同前)
        gt_emb = self.item_emb(targets).unsqueeze(1)
        _, top_k = torch.topk(gr_logits.detach(), K, dim=1) # Detach to stop gradient from CTR to GR via indices
        hard_emb = self.item_emb(top_k)
        
        num_easy = N - 1 - K
        rand_ids = torch.randint(1, self.config.num_items, (B, num_easy)).to(device)
        easy_emb = self.item_emb(rand_ids)
        
        bank = torch.cat([gt_emb, hard_emb, easy_emb], dim=1)
        labels = torch.zeros((B, N), device=device)
        labels[:, 0] = 1.0 
        
        # Shuffle
        idx = torch.randperm(N).to(device)
        bank = bank[:, idx, :]
        labels = labels[:, idx]
        
        # Interaction
        bank_proj = self.cand_proj(bank)
        u_q = u.unsqueeze(1)
        ctx, _ = self.cross_attn(u_q, bank_proj, bank_proj)
        
        # Scoring
        u_exp = u.unsqueeze(1).repeat(1, N, 1)
        ctx_exp = ctx.repeat(1, N, 1)
        feat = torch.cat([bank_proj, u_exp, ctx_exp], dim=-1)
        
        ctr_logits = self.scorer(feat).squeeze(-1)
        return ctr_logits, labels
