import torch
import torch.nn as nn
from .config import UniGCRConfig

try:
    from generative_recommenders.modeling.sequential.hstu import HSTU as OfficialHSTU
except ImportError:
    OfficialHSTU = None

class UnifiedInputLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        dim = config.embed_dim
        
        # 1. Semantic Embedding (核心)
        if config.use_semantic_seq:
            self.sem_emb = nn.Embedding(config.sem_total_vocab, dim, padding_idx=0)
            
        # 2. Atomic Embedding (仅特征)
        if config.use_atomic_seq:
            self.atom_emb = nn.Embedding(config.num_atomic_items + 1, dim, padding_idx=0)
            
        # 3. Profile Embeddings
        if config.use_cat_profile:
            self.cat_embs = nn.ModuleList([nn.Embedding(v, dim) for v in config.cat_feature_vocab_sizes])
        if config.use_num_profile:
            self.num_projs = nn.ModuleList([nn.Linear(1, dim) for _ in range(config.num_feature_size)])
            
        if config.use_cat_profile or config.use_num_profile:
            self.feat_mlp = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.LayerNorm(dim))

    def forward(self, input_dict):
        tokens = []
        
        # A. Profile (Prefix)
        prefix = []
        if self.config.use_cat_profile:
            for i, emb in enumerate(self.cat_embs):
                prefix.append(emb(input_dict['cat_feats'][:, i]).unsqueeze(1))
        if self.config.use_num_profile:
            for i, proj in enumerate(self.num_projs):
                prefix.append(proj(input_dict['num_feats'][:, i].unsqueeze(-1)).unsqueeze(1))
        if prefix:
            tokens.append(self.feat_mlp(torch.cat(prefix, dim=1)))
            
        # B. Atomic Sequence (Context)
        if self.config.use_atomic_seq and 'atom_history' in input_dict:
            tokens.append(self.atom_emb(input_dict['atom_history']))
            
        # C. Semantic Sequence (Main Context)
        if self.config.use_semantic_seq and 'sem_history' in input_dict:
            tokens.append(self.sem_emb(input_dict['sem_history']))
            
        if not tokens: raise ValueError("Empty inputs")
        return torch.cat(tokens, dim=1)

class UniGCRModel(nn.Module):
    def __init__(self, config: UniGCRConfig):
        super().__init__()
        self.config = config
        
        # 1. 输入层 & Backbone
        self.input_layer = UnifiedInputLayer(config)
        if OfficialHSTU is None: raise RuntimeError("HSTU lib missing")
        self.backbone = OfficialHSTU(config=config.to_hstu_config(), embedding_module=None)
        
        # 2. GR Head (Target: Semantic Code)
        # 只有这一个 Head，因为我们不预测 Atomic
        self.gr_head = nn.Linear(config.embed_dim, config.sem_total_vocab)
        
        # 3. CTR Components
        self.cand_proj = nn.Sequential(nn.LayerNorm(config.embed_dim), nn.Linear(config.embed_dim, config.embed_dim))
        
        # Attention Modules
        scorer_input_dim = config.embed_dim * 2
        if config.ctr_use_self_attn:
            self.cand_self_attn = nn.MultiheadAttention(config.embed_dim, 2, batch_first=True)
            scorer_input_dim += config.embed_dim
        if config.ctr_use_cross_attn:
            self.user_cross_attn = nn.MultiheadAttention(config.embed_dim, 2, batch_first=True)
            scorer_input_dim += config.embed_dim
            
        self.scorer = nn.Sequential(nn.Linear(scorer_input_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def _get_item_embedding(self, codes):
        """
        将 Semantic Codes 转换为 Item Embedding。
        codes: (B, Layers)
        Strategy: Sum Embedding of all layers
        """
        # codes: (B, Layers) -> embedding -> (B, Layers, D)
        embs = self.input_layer.sem_emb(codes)
        # Sum Pooling: (B, D)
        return torch.sum(embs, dim=1)

    def forward(self, batch_dict):
        # 1. Encode User
        x = self.input_layer(batch_dict)
        B, L, _ = x.shape
        lengths = torch.full((B,), L, dtype=torch.long, device=x.device)
        u_seq = self.backbone(x, lengths=lengths)
        u = u_seq[:, -1, :] # User State
        
        # 2. GR Prediction (Next Code)
        gr_logits = self.gr_head(u)
        
        return u, gr_logits

    def predict_ctr(self, u, pos_codes, device):
        """
        u: User State (B, D)
        pos_codes: Positive Item Semantic Codes (B, Layers)
        """
        B = u.size(0)
        N = self.config.memory_bank_size
        
        # 1. Positive Item Embedding (GT)
        # (B, D) -> (B, 1, D)
        pos_emb = self._get_item_embedding(pos_codes).unsqueeze(1)
        
        # 2. Negative Sampling (Random Negatives)
        # 这里简化：只做随机负采样。
        # 如果要做 Hard Negatives，需要从 GR Logits 做 Beam Search 得到 Codes，计算量较大
        num_negs = N - 1
        
        # 随机生成负样本 Codes (B, Negs, Layers)
        # 注意范围：需要由 GridMapper 控制范围，这里简化为全词表随机
        # 实际上应该从 GridMapper.total_vocab_size 里取，但为了严格，这里假设随机取
        rand_codes = torch.randint(1, self.config.sem_total_vocab, (B, num_negs, self.config.sem_id_layers)).to(device)
        
        # (B, Negs, Layers, D) -> Sum -> (B, Negs, D)
        neg_embs = self.input_layer.sem_emb(rand_codes).sum(dim=2)
        
        # 3. Construct Memory Bank
        bank = torch.cat([pos_emb, neg_embs], dim=1) # (B, N, D)
        
        # 4. Labels
        labels = torch.zeros((B, N), device=device)
        labels[:, 0] = 1.0
        
        # 5. Shuffle
        perm = torch.randperm(N).to(device)
        bank = bank[:, perm, :]
        labels = labels[:, perm]
        
        # 6. Scoring (User-Centric & Candidate-Aware)
        bank_proj = self.cand_proj(bank)
        u_exp = u.unsqueeze(1).repeat(1, N, 1)
        feats = [bank_proj, u_exp]
        
        if self.config.ctr_use_self_attn:
            self_out, _ = self.cand_self_attn(bank_proj, bank_proj, bank_proj)
            feats.append(self_out)
            
        if self.config.ctr_use_cross_attn:
            u_q = u.unsqueeze(1)
            cross_out, _ = self.user_cross_attn(u_q, bank_proj, bank_proj)
            feats.append(cross_out.repeat(1, N, 1))
            
        final_feats = torch.cat(feats, dim=-1)
        ctr_logits = self.scorer(final_feats).squeeze(-1)
        
        return ctr_logits, labels
