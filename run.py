import sys
import os
sys.path.append(os.getcwd()) # 确保能找到 src

from src.config import UniGCRConfig
from src.data import get_dataloaders
from src.model import UniGCRModel
from src.trainer import UniGCRTrainer
from src.utils import set_seed

def main():
    # 1. 初始化配置
    conf = UniGCRConfig()
    set_seed(conf.seed)
    
    # 2. 准备数据 (内部处理了 build vocab)
    print("Preparing Data...")
    train_dl, val_dl = get_dataloaders(conf)
    print(f"Vocab Size -> Users: {conf.num_users}, Items: {conf.num_items}")
    
    # 3. 初始化模型
    model = UniGCRModel(conf)
    
    # 4. 初始化 Trainer
    trainer = UniGCRTrainer(
        config=conf,
        model=model,
        train_loader=train_dl,
        val_loader=val_dl
    )
    
    # 5. 开始训练
    print("Start Training...")
    trainer.train()

if __name__ == "__main__":
    main()
