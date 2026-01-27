import torch
import torch.nn as nn
from tqdm import tqdm
from .utils import get_metric
import os

class UniGCRTrainer:
    def __init__(self, config, model, train_loader, val_loader=None):
        self.config = config
        self.device = torch.device(config.device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # 优化器
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
        
        # Loss Functions
        self.gr_criterion = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def calculate_loss(self, gr_logits, ctr_logits, ctr_labels, targets):
        """封装具体的 Loss 计算逻辑"""
        # 1. GR Loss
        loss_gr = self.gr_criterion(gr_logits, targets)
        
        # 2. CTR Loss (InfoNCE + BCE)
        # InfoNCE: 找出 label 为 1 的 index
        target_idx = torch.argmax(ctr_labels, dim=1)
        loss_info = self.ce_loss(ctr_logits / self.config.temp, target_idx)
        loss_bce = self.bce_loss(ctr_logits, ctr_labels)
        
        loss_ctr = (self.config.loss_alpha * loss_info) + \
                   (self.config.loss_beta * loss_bce)
                   
        total_loss = loss_gr + loss_ctr
        return total_loss, loss_gr, loss_ctr

    def train_epoch(self, epoch_idx):
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.train_loader, desc=f"Train Epoch {epoch_idx}")
        
        for batch in pbar:
            # 1. Move Data
            hist = batch['history'].to(self.device)
            uid = batch['user_id'].to(self.device)
            tgt = batch['target'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # 2. Forward (Phase 1: Common & GR)
            u, gr_logits = self.model(hist, uid)
            
            # 3. Forward (Phase 2: CTR)
            # 在 Trainer 中调用模型的辅助方法来获取 CTR 结果
            ctr_logits, ctr_labels = self.model.predict_ctr(u, tgt, gr_logits, self.device)
            
            # 4. Loss
            loss, l_gr, l_ctr = self.calculate_loss(gr_logits, ctr_logits, ctr_labels, tgt)
            
            # 5. Backward
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'gr': f"{l_gr.item():.2f}"})
            
        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def evaluate(self, topk=10):
        if not self.val_loader: return 0, 0
        
        self.model.eval()
        preds = []
        
        for batch in tqdm(self.val_loader, desc="Evaluating"):
            hist = batch['history'].to(self.device)
            uid = batch['user_id'].to(self.device)
            tgt = batch['target'].to(self.device)
            
            # Inference: 只用 GR head 进行召回评估
            # 完整系统应该 GR TopK -> CTR Re-rank -> Top1，这里评估 GR 能力
            u, gr_logits = self.model(hist, uid)
            
            _, top_indices = torch.topk(gr_logits, topk, dim=1)
            
            for i in range(len(tgt)):
                preds.append((top_indices[i], tgt[i]))
                
        hit, ndcg = get_metric(preds, topk=topk)
        return hit, ndcg

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def train(self):
        best_hit = 0
        for epoch in range(1, self.config.epochs + 1):
            train_loss = self.train_epoch(epoch)
            hit, ndcg = self.evaluate()
            
            print(f"Epoch {epoch} | Loss: {train_loss:.4f} | Hit@10: {hit:.4f} | NDCG@10: {ndcg:.4f}")
            
            if hit > best_hit:
                best_hit = hit
                self.save(f"best_model_epoch_{epoch}.pth")
