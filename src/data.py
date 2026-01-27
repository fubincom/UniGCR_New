import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import json
import gzip
import os
import numpy as np
from collections import defaultdict

class AmazonSeqDataset(Dataset):
    def __init__(self, config, mode='train'):
        """
        mode: 'train', 'val', 'test'
        """
        self.config = config
        self.mode = mode
        self.data = []
        
        # 1. 原始数据读取与清洗 (只在内存中构建，不落盘)
        print(f"Loading data for {mode}...")
        self.user_map = {}
        self.item_map = {}
        self._load_raw_data(config.data_path)
        
    def _load_raw_data(self, path):
        # 实际项目中，这里可以替换为读取已处理好的 index map 以保证一致性
        # 为演示方便，这里做简单的内存处理
        
        # 读取所有交互
        user_interactions = defaultdict(list)
        u_cnt, i_cnt = 1, 1 # 0 是 padding
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data not found at {path}")

        # 第一遍扫描：构建词表 (如果是Train模式)
        # 真实场景通常读取外部vocab文件
        with gzip.open(path, 'r') as f:
            for line in f:
                d = json.loads(line)
                uid, iid = d['reviewerID'], d['asin']
                if uid not in self.user_map: 
                    self.user_map[uid] = u_cnt; u_cnt += 1
                if iid not in self.item_map: 
                    self.item_map[iid] = i_cnt; i_cnt += 1
                
                u_idx = self.user_map[uid]
                i_idx = self.item_map[iid]
                user_interactions[u_idx].append((d['unixReviewTime'], i_idx))
        
        # 更新 Config 中的词表大小
        self.config.num_users = u_cnt
        self.config.num_items = i_cnt
        
        # 构建序列
        for uid, interactions in user_interactions.items():
            # 按时间排序
            interactions.sort(key=lambda x: x[0])
            item_seq = [x[1] for x in interactions]
            
            if len(item_seq) < 3: continue
            
            # 数据切分策略 (Leave-one-out)
            # Train: ... -> T-2
            # Val: ... -> T-1
            # Test: ... -> T
            
            if self.mode == 'train':
                # 截取到倒数第二个
                self.data.append({
                    'user': uid, 
                    'seq': item_seq[:-2], 
                    'target': item_seq[-2]
                })
            elif self.mode == 'val':
                self.data.append({
                    'user': uid, 
                    'seq': item_seq[:-1], 
                    'target': item_seq[-1]
                })
            # 可以添加 test 逻辑

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        seq = item['seq']
        
        # 动态 Padding/截断
        max_len = self.config.max_seq_len
        if len(seq) > max_len:
            seq = seq[-max_len:]
        else:
            pad_len = max_len - len(seq)
            seq = [0] * pad_len + seq # 左Padding
            
        return {
            'user_id': torch.tensor(item['user'], dtype=torch.long),
            'history': torch.tensor(seq, dtype=torch.long),
            'target': torch.tensor(item['target'], dtype=torch.long)
        }

# def get_dataloaders(config):
#     # 实例化 Dataset
#     train_ds = AmazonSeqDataset(config, mode='train')
#     val_ds = AmazonSeqDataset(config, mode='val')
#     # 共享词表大小
#     val_ds.config.num_users = train_ds.config.num_users
#     val_ds.config.num_items = train_ds.config.num_items
    
#     train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=4)
#     val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=4)
    
#     return train_loader, val_loader

class HybridDataset(Dataset):
    def __init__(self, config, mode='train'):
        self.config = config
        self.mode = mode
        self.data = []
        
        # 模拟加载数据
        # 真实场景中，你会从 feature store 或预处理好的 pkl 读取这些列
        self._load_raw_data(config.data_path)

    def _load_raw_data(self, path):
        # 简单逻辑：如果文件不存在，生成随机数据方便调试
        if not os.path.exists(path):
            print(f"Data file {path} not found. Generating dummy hybrid data...")
            num_samples = 1000
            for _ in range(num_samples):
                # 随机生成类别特征索引 (例如: UID, Region, Device)
                cats = [np.random.randint(0, v) for v in self.config.cat_feature_vocab_sizes]
                # 随机生成数值特征 (例如: Age, Rating, ClickCount) - 归一化后的数据
                nums = np.random.rand(self.config.num_feature_size).tolist()
                
                self.data.append({
                    'seq': np.random.randint(1, 100, 20).tolist(),
                    'cat_feats': cats,
                    'num_feats': nums,
                    'target': np.random.randint(1, 100)
                })
            self.config.num_items = 1000
            return

        # 真实读取逻辑 (以Amazon为例，我们把 reviewerID 当作其中一个 cat feature)
        # 这里仅作演示，不展开具体的 Pandas/ETL 代码
        # 核心是构建 list of dicts: [{'seq':..., 'cat_feats': [u_id, ...], 'num_feats': [...]}, ...]
        pass 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        seq = item['seq']
        
        # 序列截断/Padding
        max_len = self.config.max_seq_len
        if len(seq) > max_len:
            seq = seq[-max_len:]
        else:
            seq = [0] * (max_len - len(seq)) + seq

        return {
            'history': torch.tensor(seq, dtype=torch.long),
            # 类别特征: LongTensor (Batch, Num_Cats)
            'cat_feats': torch.tensor(item['cat_feats'], dtype=torch.long),
            # 数值特征: FloatTensor (Batch, Num_Nums)
            'num_feats': torch.tensor(item['num_feats'], dtype=torch.float),
            'target': torch.tensor(item['target'], dtype=torch.long)
        }

def get_dataloaders(config):
    # 使用新的 HybridDataset
    train_ds = HybridDataset(config, mode='train')
    val_ds = HybridDataset(config, mode='val')
    val_ds.config.num_items = train_ds.config.num_items
    
    train_sampler = DistributedSampler(train_ds, shuffle=True)
    val_sampler = DistributedSampler(val_ds, shuffle=False)
    
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, sampler=train_sampler, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, sampler=val_sampler, num_workers=2)
    
    return train_loader, val_loader
