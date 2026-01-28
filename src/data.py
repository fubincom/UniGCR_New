import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from .grid_utils import GridMapper

class UniversalDataset(Dataset):
    def __init__(self, config, mode='train'):
        self.config = config
        self.mode = mode
        self.data = []
        
        # Semantics
        self.grid_mapper = None
        if config.use_semantic_seq:
            self.grid_mapper = GridMapper(
                config.grid_mapping_path,
                config.sem_id_layers,
                config.sem_id_codebook_size
            )
            self.config.sem_total_vocab = self.grid_mapper.total_vocab_size
            
        self._load_data(config.data_path)

    def _load_data(self, path):
        # TODO: 替换为真实的 Pickle/JSON 读取
        print(f"Loading data from {path} (Simulation Mode)")
        for _ in range(1000):
            sample = {}
            if self.config.use_semantic_seq:
                sample['sem_seq'] = np.random.randint(1, 1000, 20).tolist()
                sample['target'] = np.random.randint(1, 1000)
            if self.config.use_cat_profile:
                sample['cat_feats'] = [np.random.randint(0, v) for v in self.config.cat_feature_vocab_sizes]
            if self.config.use_num_profile:
                sample['num_feats'] = np.random.rand(self.config.num_feature_size).tolist()
            self.data.append(sample)

    def __getitem__(self, idx):
        raw_item = self.data[idx]
        output = {}
        
        if self.config.use_semantic_seq:
            # Flatten History
            seq_codes = self.grid_mapper.flatten_sequence(raw_item['sem_seq'])
            tgt_codes = self.grid_mapper.get_codes(raw_item['target'])
            
            # Training Sequence: History + Target
            full_seq = seq_codes + tgt_codes
            max_len = self.config.max_seq_len
            if len(full_seq) > max_len:
                full_seq = full_seq[-max_len:]
            else:
                full_seq = [0] * (max_len - len(full_seq)) + full_seq
            
            output['sem_history'] = torch.tensor(full_seq[:-1], dtype=torch.long)
            output['sem_target'] = torch.tensor(full_seq[1:], dtype=torch.long)
            output['sem_target_eval'] = torch.tensor(tgt_codes, dtype=torch.long) # for Eval

        if self.config.use_cat_profile:
            output['cat_feats'] = torch.tensor(raw_item['cat_feats'], dtype=torch.long)
        if self.config.use_num_profile:
            output['num_feats'] = torch.tensor(raw_item['num_feats'], dtype=torch.float)
            
        return output

    def __len__(self):
        return len(self.data)

def get_dataloaders(config):
    train_ds = UniversalDataset(config, mode='train')
    val_ds = UniversalDataset(config, mode='val')
    # Sync dynamic config
    if config.use_semantic_seq:
        val_ds.config.sem_total_vocab = train_ds.config.sem_total_vocab
        
    train_sampler = DistributedSampler(train_ds, shuffle=True)
    val_sampler = DistributedSampler(val_ds, shuffle=False)
    
    train_dl = DataLoader(train_ds, batch_size=config.batch_size, sampler=train_sampler, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=config.batch_size, sampler=val_sampler, num_workers=2)
    return train_dl, val_dl
