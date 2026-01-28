import torch
import torch.nn as nn
from tqdm import tqdm
import deepspeed
import json
import numpy as np
from .utils import is_main_process

class UniGCRTrainer:
    def __init__(self, config, args, model, train_loader, val_loader=None):
        self.config = config
        self.args = args
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # --- Loss Definitions ---
        # GR 任务: 预测下一个 Semantic Code (Cross Entropy)
        self.gr_criterion = nn.CrossEntropyLoss()
        
        # CTR 任务: 联合 Loss
        if config.enable_ctr:
            self.bce_loss = nn.BCEWithLogitsLoss()
            self.ce_loss = nn.CrossEntropyLoss()
            
        # --- DeepSpeed Initialization ---
        ds_config = None
        if hasattr(args, 'deepspeed_config') and args.deepspeed_config:
            try:
                with open(args.deepspeed_config, 'r') as f:
                    ds_config = json.load(f)
                if is_main_process():
                    print(f"[Trainer] Loaded DeepSpeed config from {args.deepspeed_config}")
            except Exception as e:
                print(f"[Trainer] Error loading DS config: {e}")
                
        # 初始化 DeepSpeed Engine
        self.model_engine, self.optimizer, _, _ = deepspeed.initialize(
            args=args, 
            model=model, 
            model_parameters=model.parameters(), 
            config=ds_config, 
            dist_init_required=True
        )

    def calculate_ctr_loss(self, ctr_logits, ctr_labels):
        """
        计算 CTR 任务的混合 Loss (InfoNCE + BCE)
        """
        # 1. InfoNCE (Ranking): 找出正样本所在的 Index
        target_idx = torch.argmax(ctr_labels, dim=1)
        loss_info = self.ce_loss(ctr_logits / self.config.temp, target_idx)
        
        # 2. BCE (Calibration): 逐个判断是点击(1)还是未点击(0)
        loss_bce = self.bce_loss(ctr_logits, ctr_labels)
        
        # 加权求和
        return (self.config.loss_alpha * loss_info) + (self.config.loss_beta * loss_bce)

    def train_epoch(self, epoch_idx):
        """
        完整的训练 Epoch 逻辑
        """
        self.model_engine.train()
        
        # 仅主进程显示进度条
        if is_main_process():
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch_idx}")
        else:
            pbar = self.train_loader
            
        # 累计 Loss 用于日志
        total_loss = 0.0
        gr_loss_sum = 0.0
        ctr_loss_sum = 0.0
        
        # 获取 GridMapper (用于 Beam Search 时的 Masking)
        # 注意: DataLoader 可能经过 DistributedSampler 封装，需要通过 dataset 访问
        grid_mapper = self.train_loader.dataset.grid_mapper
        if grid_mapper is None and self.config.use_semantic_seq:
            raise ValueError("GridMapper is required for Semantic ID training but not found.")
        
        for batch in pbar:
            # 1. 将 Batch 数据移动到 GPU
            batch = {
                k: v.to(self.model_engine.device) 
                for k, v in batch.items() 
                if isinstance(v, torch.Tensor)
            }
            
            self.model_engine.zero_grad()
            
            # 2. Forward Pass (Shared Backbone & GR Head)
            # u: User State (B, D)
            # gr_logits: (B, Vocab) - Next Token Prediction Logits
            u, gr_logits = self.model_engine(batch)
            
            loss = 0.0
            
            # --- Task A: Generative Retrieval (GR) ---
            if self.config.use_semantic_seq:
                # Target: Flattened Semantic Sequence
                # shape: (B * Layers)
                target_flat = batch['sem_target'].view(-1)
                loss_gr = self.gr_criterion(gr_logits, target_flat)
                
                loss += loss_gr
                gr_loss_sum += loss_gr.item()
            
            # --- Task B: CTR Prediction (Optional) ---
            loss_ctr_val = 0.0
            if self.config.enable_ctr:
                # 获取原始模型 (DeepSpeed wrap 了 module)
                real_model = self.model_engine.module if hasattr(self.model_engine, 'module') else self.model_engine
                
                # 正样本: 用户实际点击的 Item 的 Semantic Codes
                # shape: (B, Layers)
                pos_codes = batch['ctr_pos_codes']
                
                # 调用 predict_ctr
                # 注意: 内部包含 Beam Search 逻辑，需要传入 batch (获取 History) 和 mapper
                ctr_logits, ctr_labels = real_model.predict_ctr(
                    u, batch, pos_codes, self.model_engine.device, grid_mapper
                )
                
                # 计算 Loss
                loss_ctr = self.calculate_ctr_loss(ctr_logits, ctr_labels)
                
                loss += loss_ctr
                loss_ctr_val = loss_ctr.item()
                ctr_loss_sum += loss_ctr_val

            # 3. Backward & Step (DeepSpeed)
            self.model_engine.backward(loss)
            self.model_engine.step()
            
            total_loss += loss.item()
            
            # 更新进度条
            if is_main_process():
                logs = {
                    'Loss': f"{loss.item():.4f}", 
                    'GR': f"{loss_gr.item() if self.config.use_semantic_seq else 0:.4f}"
                }
                if self.config.enable_ctr:
                    logs['CTR'] = f"{loss_ctr_val:.4f}"
                pbar.set_postfix(logs)
            
        # 计算平均 Loss
        num_batches = len(self.train_loader)
        return {
            'loss': total_loss / num_batches,
            'gr_loss': gr_loss_sum / num_batches,
            'ctr_loss': ctr_loss_sum / num_batches
        }

    @torch.no_grad()
    def evaluate(self, topk=10):
        """
        完整的评估逻辑：Beam Search 生成 -> Item 还原 -> 指标计算
        """
        if not self.val_loader:
            return {}
        
        self.model_engine.eval()
        device = self.model_engine.device
        
        all_hits = []
        all_ndcgs = []
        
        # 必须获取 Mapper 进行 ID 还原
        dataset = self.val_loader.dataset
        grid_mapper = dataset.grid_mapper
        if not grid_mapper:
            print("[Eval] Warning: GridMapper missing, skipping evaluation.")
            return {}

        if is_main_process():
            iterator = tqdm(self.val_loader, desc="Evaluating")
        else:
            iterator = self.val_loader
        
        # 获取原始模型 (用于调用 generate 方法)
        real_model = self.model_engine.module if hasattr(self.model_engine, 'module') else self.model_engine

        for batch in iterator:
            # 移到 GPU
            batch = {
                k: v.to(device) 
                for k, v in batch.items() 
                if isinstance(v, torch.Tensor)
            }
            
            # GT Item Codes: (B, Layers) - 评估用的 Target
            tgt_codes = batch['sem_target_eval']
            
            # --- 1. 生成候选 (Beam Search) ---
            # 返回: (B, K, Layers)
            # 注意: 这里使用 Beam Search 确保生成的 Semantic ID 是合法的
            candidates = real_model.generate_gr_candidates(
                batch, k=topk, grid_mapper=grid_mapper
            )
            
            # --- 2. 还原 Item ID 并计算指标 ---
            # 转 CPU 进行查表操作
            cand_cpu = candidates.cpu().numpy()
            tgt_cpu = tgt_codes.cpu().numpy()
            
            B = cand_cpu.shape[0]
            
            for i in range(B):
                # 还原真实 Item ID
                true_item_id = grid_mapper.codes_to_item(tgt_cpu[i])
                
                # 如果 Target 对应的 Code 组合在映射表中不存在 (数据问题)，跳过
                if true_item_id is None:
                    continue 
                
                # 还原预测的 Top-K Item IDs
                pred_item_ids = []
                for k_idx in range(topk):
                    code_tuple = cand_cpu[i, k_idx]
                    pid = grid_mapper.codes_to_item(code_tuple)
                    # 如果生成的 Code 组合无效 (pid is None)，视为未命中
                    pred_item_ids.append(pid)
                
                # 计算 Hit & NDCG
                hit = 0.0
                ndcg = 0.0
                
                if true_item_id in pred_item_ids:
                    hit = 1.0
                    rank = pred_item_ids.index(true_item_id)
                    ndcg = 1.0 / np.log2(rank + 2.0)
                
                all_hits.append(hit)
                all_ndcgs.append(ndcg)
        
        # --- 3. 分布式指标聚合 (All Reduce) ---
        # 计算当前 GPU 的平均值
        local_hit_sum = np.sum(all_hits)
        local_ndcg_sum = np.sum(all_ndcgs)
        local_count = len(all_hits)
        
        # 转 Tensor 方便通信
        stats_tensor = torch.tensor([local_hit_sum, local_ndcg_sum, local_count], dtype=torch.float32, device=device)
        
        # 全局求和
        torch.distributed.all_reduce(stats_tensor, op=torch.distributed.ReduceOp.SUM)
        
        global_hit_sum = stats_tensor[0].item()
        global_ndcg_sum = stats_tensor[1].item()
        global_count = stats_tensor[2].item()
        
        final_hit = global_hit_sum / global_count if global_count > 0 else 0.0
        final_ndcg = global_ndcg_sum / global_count if global_count > 0 else 0.0
        
        return {'Hit@10': final_hit, 'NDCG@10': final_ndcg}

    def save(self, tag):
        """保存 Checkpoint"""
        self.model_engine.save_checkpoint(save_dir="checkpoints", tag=tag)

    def train(self):
        """
        主训练流程控制: Early Stopping + Logging
        """
        # 确定监控指标
        if self.config.enable_ctr:
            # 如果开启 CTR，我们通常更关注 GR 的召回能力是否因为联合训练提升了，
            # 或者关注 CTR 的 LogLoss。这里以 HitRate 为准，因为最终目标是推荐。
            monitor_metric = 'Hit@10' 
            mode = 'max'
        else:
            monitor_metric = 'Hit@10'
            mode = 'max'
            
        best_metric = 0.0 if mode == 'max' else float('inf')
        patience_counter = 0
        
        if is_main_process():
            print(f"Start Training... Monitor: {monitor_metric} (Best is {mode})")
            print(f"DeepSpeed Config Enabled: {self.args.deepspeed_config is not None}")

        for epoch in range(1, self.config.epochs + 1):
            # 必须设置 DistributedSampler 的 epoch 以保证 shuffle
            if hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)
            
            # 1. Train
            train_metrics = self.train_epoch(epoch)
            
            # 2. Eval
            eval_metrics = self.evaluate(topk=10)
            
            # 3. Logging
            if is_main_process():
                log_str = f"Epoch {epoch} | "
                log_str += f"Loss: {train_metrics['loss']:.4f} "
                log_str += f"(GR: {train_metrics['gr_loss']:.4f}, CTR: {train_metrics['ctr_loss']:.4f}) | "
                log_str += f"Eval: Hit@10: {eval_metrics.get('Hit@10', 0):.4f} NDCG@10: {eval_metrics.get('NDCG@10', 0):.4f}"
                print(log_str)
            
            # 4. Early Stopping Logic
            current_val = eval_metrics.get(monitor_metric, 0.0)
            
            improved = False
            if mode == 'max':
                if current_val > best_metric: improved = True
            else:
                if current_val < best_metric: improved = True
            
            if improved:
                best_metric = current_val
                patience_counter = 0
                if is_main_process():
                    print(f" >> New Best {monitor_metric}! Saving Model...")
                self.save("best_model")
            else:
                patience_counter += 1
                if is_main_process():
                    print(f" >> No improvement. Patience: {patience_counter}/{self.config.patience}")
            
            if patience_counter >= self.config.patience:
                if is_main_process():
                    print(" >> Early Stopping Triggered.")
                break
