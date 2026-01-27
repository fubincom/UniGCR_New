import torch
import torch.nn as nn
from tqdm import tqdm
import deepspeed
from .utils import compute_gr_metrics, compute_ctr_metrics, is_main_process, gather_tensors

class UniGCRTrainer:
    def __init__(self, config, args, model, train_loader, val_loader=None):
        self.config = config
        self.args = args
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Loss Functions
        self.gr_criterion = nn.CrossEntropyLoss()
        
        if config.enable_ctr:
            self.bce_loss = nn.BCEWithLogitsLoss()
            self.ce_loss = nn.CrossEntropyLoss()

        # DeepSpeed Initialize
        self.model_engine, self.optimizer, _, _ = deepspeed.initialize(
            args=args,
            model=model,
            model_parameters=model.parameters(),
            dist_init_required=True
        )

    def calculate_ctr_loss(self, ctr_logits, ctr_labels):
        """仅在 enable_ctr=True 时调用"""
        target_idx = torch.argmax(ctr_labels, dim=1)
        loss_info = self.ce_loss(ctr_logits / self.config.temp, target_idx)
        loss_bce = self.bce_loss(ctr_logits, ctr_labels)
        
        loss_ctr = (self.config.loss_alpha * loss_info) + \
                   (self.config.loss_beta * loss_bce)
        return loss_ctr, loss_info, loss_bce

    def train_epoch(self, epoch_idx):
        self.model_engine.train()
        
        # 用于记录平均 Loss
        total_loss = 0
        total_gr_loss = 0
        total_ctr_loss = 0
        
        if is_main_process():
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch_idx}")
        else:
            pbar = self.train_loader

        for batch in pbar:
            device = self.model_engine.device
            hist = batch['history'].to(device)
            uid = batch['user_id'].to(device)
            tgt = batch['target'].to(device)
            
            self.model_engine.zero_grad()
            
            # --- 1. GR Task (Always Run) ---
            # 获取 user state 和 GR logits
            u, gr_logits = self.model_engine(hist, uid)
            loss_gr = self.gr_criterion(gr_logits, tgt)
            loss = loss_gr
            
            # --- 2. CTR Task (Conditional) ---
            loss_ctr_val = 0.0
            if self.config.enable_ctr:
                if hasattr(self.model_engine, 'module'):
                    real_model = self.model_engine.module
                else:
                    real_model = self.model_engine
                
                # 只有开启 CTR 才进行 Memory Bank 构建和重排
                ctr_logits, ctr_labels = real_model.predict_ctr(u, tgt, gr_logits, device)
                loss_ctr, _, _ = self.calculate_ctr_loss(ctr_logits, ctr_labels)
                
                loss += loss_ctr
                loss_ctr_val = loss_ctr.item()

            # Backward
            self.model_engine.backward(loss)
            self.model_engine.step()
            
            # Logging Accumulation
            total_loss += loss.item()
            total_gr_loss += loss_gr.item()
            total_ctr_loss += loss_ctr_val
            
            if is_main_process():
                logs = {'loss': f"{loss.item():.4f}", 'gr': f"{loss_gr.item():.4f}"}
                if self.config.enable_ctr:
                    logs['ctr'] = f"{loss_ctr_val:.4f}"
                pbar.set_postfix(logs)
            
        # 返回平均值
        n_batches = len(self.train_loader)
        return {
            'loss': total_loss / n_batches,
            'gr_loss': total_gr_loss / n_batches,
            'ctr_loss': total_ctr_loss / n_batches
        }

    @torch.no_grad()
    def evaluate(self, topk=10):
        if not self.val_loader: return {}
        
        self.model_engine.eval()
        device = self.model_engine.device
        
        # 存储所有 Batch 的结果
        # GR
        all_gr_preds = []
        all_gr_targets = []
        
        # CTR (Flattened)
        all_ctr_logits = []
        all_ctr_labels = [] # 这里存 0/1 flat labels
        
        iterator = tqdm(self.val_loader, desc="Eval") if is_main_process() else self.val_loader
        
        for batch in iterator:
            hist = batch['history'].to(device)
            uid = batch['user_id'].to(device)
            tgt = batch['target'].to(device)
            
            u, gr_logits = self.model_engine(hist, uid)
            
            # --- GR Eval ---
            _, top_indices = torch.topk(gr_logits, topk, dim=1)
            all_gr_preds.append(top_indices)
            all_gr_targets.append(tgt)
            
            # --- CTR Eval (如果开启) ---
            if self.config.enable_ctr:
                if hasattr(self.model_engine, 'module'):
                    real_model = self.model_engine.module
                else:
                    real_model = self.model_engine
                
                # 在 Eval 阶段，predict_ctr 会动态构建 validation 用的 memory bank (1 GT + N-1 Negs)
                ctr_logits, ctr_labels = real_model.predict_ctr(u, tgt, gr_logits, device)
                
                # Flatten for AUC calculation
                all_ctr_logits.append(ctr_logits.view(-1))
                all_ctr_labels.append(ctr_labels.view(-1))

        # 1. 拼接当前 GPU 上的结果
        local_gr_preds = torch.cat(all_gr_preds, dim=0)
        local_gr_targets = torch.cat(all_gr_targets, dim=0)
        
        # 2. Gather 所有 GPU 的结果到 Tensor (用于精确计算 Metrics)
        global_gr_preds = gather_tensors(local_gr_preds)
        global_gr_targets = gather_tensors(local_gr_targets)
        
        results = {}
        
        # 只有主进程负责计算最终指标并打印
        if is_main_process():
            # GR Metrics
            hit, ndcg = compute_gr_metrics(global_gr_preds, global_gr_targets, k=topk)
            results['Hit@10'] = hit
            results['NDCG@10'] = ndcg
            
            # CTR Metrics
            if self.config.enable_ctr and len(all_ctr_logits) > 0:
                local_ctr_logits = torch.cat(all_ctr_logits, dim=0)
                local_ctr_labels = torch.cat(all_ctr_labels, dim=0)
                
                # 为了计算 AUC，我们需要所有样本
                global_ctr_logits = gather_tensors(local_ctr_logits)
                global_ctr_labels = gather_tensors(local_ctr_labels)
                
                auc, logloss = compute_ctr_metrics(global_ctr_logits, global_ctr_labels)
                results['AUC'] = auc
                results['LogLoss'] = logloss
        
        return results

    def save(self, tag):
        self.model_engine.save_checkpoint(save_dir="checkpoints", tag=tag)

    def train(self):
        best_metric = 0 # 可以是 Hit 或者 AUC，取决于你关注哪个
        
        for epoch in range(1, self.config.epochs + 1):
            if hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)
            
            # Training
            train_metrics = self.train_epoch(epoch)
            
            # Evaluation
            eval_metrics = self.evaluate()
            
            if is_main_process():
                # 格式化输出
                log_str = f"Epoch {epoch} | "
                log_str += f"Train Loss: {train_metrics['loss']:.4f} (GR: {train_metrics['gr_loss']:.4f}"
                if self.config.enable_ctr:
                    log_str += f", CTR: {train_metrics['ctr_loss']:.4f})"
                else:
                    log_str += ")"
                
                log_str += " | Eval: "
                for k, v in eval_metrics.items():
                    log_str += f"{k}: {v:.4f}  "
                
                print(log_str)
                
                # 模型保存策略：优先看 HitRate，如果只训 CTR 可以改成看 AUC
                current_metric = eval_metrics.get('Hit@10', 0)
                if current_metric > best_metric:
                    best_metric = current_metric
                    self.save("best_model")
