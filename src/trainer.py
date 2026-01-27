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
        target_idx = torch.argmax(ctr_labels, dim=1)
        loss_info = self.ce_loss(ctr_logits / self.config.temp, target_idx)
        loss_bce = self.bce_loss(ctr_logits, ctr_labels)
        
        loss_ctr = (self.config.loss_alpha * loss_info) + \
                   (self.config.loss_beta * loss_bce)
        return loss_ctr, loss_info, loss_bce

    def train_epoch(self, epoch_idx):
        self.model_engine.train()
        
        total_loss = 0
        total_gr_loss = 0
        total_ctr_loss = 0
        
        if is_main_process():
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch_idx}")
        else:
            pbar = self.train_loader

        for batch in pbar:
            device = self.model_engine.device
            
            # --- Unpack Batch (Hybrid Input) ---
            hist = batch['history'].to(device)
            cat_feats = batch['cat_feats'].to(device)
            num_feats = batch['num_feats'].to(device)
            tgt = batch['target'].to(device)
            
            self.model_engine.zero_grad()
            
            # 1. GR Task: 传入所有特征
            u, gr_logits = self.model_engine(hist, cat_feats, num_feats)
            loss_gr = self.gr_criterion(gr_logits, tgt)
            loss = loss_gr
            
            # 2. CTR Task
            loss_ctr_val = 0.0
            if self.config.enable_ctr:
                if hasattr(self.model_engine, 'module'): real_model = self.model_engine.module
                else: real_model = self.model_engine
                
                # predict_ctr 内部逻辑不变
                ctr_logits, ctr_labels = real_model.predict_ctr(u, tgt, gr_logits, device)
                loss_ctr, _, _ = self.calculate_ctr_loss(ctr_logits, ctr_labels)
                loss += loss_ctr
                loss_ctr_val = loss_ctr.item()

            self.model_engine.backward(loss)
            self.model_engine.step()

            
            total_loss += loss.item()
            total_gr_loss += loss_gr.item()
            total_ctr_loss += loss_ctr_val
            
            # 实时进度条显示
            if is_main_process():
                logs = {
                    'loss': f"{loss.item():.4f}", 
                    'gr_loss': f"{loss_gr.item():.4f}"
                }
                if self.config.enable_ctr:
                    logs['ctr_loss'] = f"{loss_ctr_val:.4f}"
                pbar.set_postfix(logs)
            
        # 返回平均 Loss
        n = len(self.train_loader)
        return {
            'loss': total_loss / n,
            'gr_loss': total_gr_loss / n,
            'ctr_loss': total_ctr_loss / n
        }

    @torch.no_grad()
    def evaluate(self, topk=10):
        if not self.val_loader: return {}
        
        self.model_engine.eval()
        device = self.model_engine.device
        
        all_gr_preds = []
        all_gr_targets = []
        all_ctr_logits = []
        all_ctr_labels = []
        
        iterator = tqdm(self.val_loader, desc="Eval") if is_main_process() else self.val_loader
        
        for batch in iterator:
            hist = batch['history'].to(device)
            cat_feats = batch['cat_feats'].to(device)
            num_feats = batch['num_feats'].to(device)
            tgt = batch['target'].to(device)
            
            # Inference 也要传入特征
            u, gr_logits = self.model_engine(hist, cat_feats, num_feats)
            
            # --- GR Eval (Always) ---
            _, top_indices = torch.topk(gr_logits, topk, dim=1)
            all_gr_preds.append(top_indices)
            all_gr_targets.append(tgt)
            
            # --- CTR Eval (Conditional) ---
            if self.config.enable_ctr:
                if hasattr(self.model_engine, 'module'):
                    real_model = self.model_engine.module
                else:
                    real_model = self.model_engine
                
                ctr_logits, ctr_labels = real_model.predict_ctr(u, tgt, gr_logits, device)
                all_ctr_logits.append(ctr_logits.view(-1))
                all_ctr_labels.append(ctr_labels.view(-1))

        # Gather results from all GPUs
        global_gr_preds = gather_tensors(torch.cat(all_gr_preds, dim=0))
        global_gr_targets = gather_tensors(torch.cat(all_gr_targets, dim=0))
        
        results = {}
        
        # 1. GR Metrics (Always Calculate)
        hit, ndcg = compute_gr_metrics(global_gr_preds, global_gr_targets, k=topk)
        results['Hit@10'] = hit
        results['NDCG@10'] = ndcg
        
        # 2. CTR Metrics (Conditional)
        if self.config.enable_ctr and len(all_ctr_logits) > 0:
            global_ctr_logits = gather_tensors(torch.cat(all_ctr_logits, dim=0))
            global_ctr_labels = gather_tensors(torch.cat(all_ctr_labels, dim=0))
            
            auc, logloss = compute_ctr_metrics(global_ctr_logits, global_ctr_labels)
            results['AUC'] = auc
            results['LogLoss'] = logloss
            
        return results

    def save(self, tag):
        self.model_engine.save_checkpoint(save_dir="checkpoints", tag=tag)

    def train(self):
        # --- 策略配置 ---
        if self.config.enable_ctr:
            monitor_metric = 'LogLoss'
            mode = 'min'  # LogLoss 越小越好
            best_metric = float('inf')
        else:
            monitor_metric = 'Hit@10'
            mode = 'max'  # HitRate 越大越好
            best_metric = 0.0
            
        patience_counter = 0
        
        if is_main_process():
            print(f"Start Training... Monitor: {monitor_metric} (Best is {mode})")

        for epoch in range(1, self.config.epochs + 1):
            if hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)
            
            # Training
            train_metrics = self.train_epoch(epoch)
            
            # Evaluation
            eval_metrics = self.evaluate()
            
            # --- 日志输出 ---
            if is_main_process():
                # 基础日志 (GR部分)
                log_str = f"Epoch {epoch} | "
                log_str += f"Train Loss: {train_metrics['loss']:.4f} (GR: {train_metrics['gr_loss']:.4f}"
                
                # CTR Loss日志
                if self.config.enable_ctr:
                    log_str += f", CTR: {train_metrics['ctr_loss']:.4f})"
                else:
                    log_str += ")"
                
                log_str += " | Eval: "
                
                # GR Metrics (必显)
                log_str += f"Hit@10: {eval_metrics.get('Hit@10', 0):.4f} NDCG@10: {eval_metrics.get('NDCG@10', 0):.4f} "
                
                # CTR Metrics (选显)
                if self.config.enable_ctr:
                    log_str += f"AUC: {eval_metrics.get('AUC', 0):.4f} LogLoss: {eval_metrics.get('LogLoss', 0):.4f}"
                
                print(log_str)
            
            # --- Early Stopping & Best Model Logic ---
            current_val = eval_metrics.get(monitor_metric)
            if current_val is None:
                continue # Should not happen

            improved = False
            if mode == 'min':
                if current_val < best_metric:
                    improved = True
            else: # mode == 'max'
                if current_val > best_metric:
                    improved = True
            
            if improved:
                best_metric = current_val
                patience_counter = 0
                if is_main_process():
                    print(f" >> New Best {monitor_metric}: {best_metric:.4f}. Saving Model...")
                self.save("best_model")
            else:
                patience_counter += 1
                if is_main_process():
                    print(f" >> No improvement. Patience: {patience_counter}/{self.config.patience}")
            
            if patience_counter >= self.config.patience:
                if is_main_process():
                    print(" >> Early Stopping Triggered.")
                break
