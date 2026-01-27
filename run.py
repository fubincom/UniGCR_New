import sys
import os
sys.path.append(os.getcwd()) # 确保能找到 src

from src.config import UniGCRConfig
from src.data import get_dataloaders
from src.model import UniGCRModel
from src.trainer import UniGCRTrainer
from src.utils import set_seed

def parse_args():
    parser = argparse.ArgumentParser(description="Uni-GCR DeepSpeed Training")
    
    # DeepSpeed 分布式必须参数
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='local rank passed from distributed launcher')
    
    # 显式指定 DeepSpeed 配置文件路径
    parser.add_argument('--deepspeed_config', type=str, default='ds_config.json',
                        help='Path to DeepSpeed config file')
    
    # 添加 DeepSpeed 自身的参数 (如 --deepspeed, --deepspeed_mpi 等)
    parser = deepspeed.add_config_arguments(parser)
    
    args = parser.parse_args()
    return args


def main():
    # 1. 解析参数
    args = parse_args()
    
    # 2. 初始化分布式环境 (torchrun 会自动设置环境变量)
    setup_distributed()
    
    # 3. 初始化配置 (重点修改区域)
    conf = UniGCRConfig()
    
    # --- [关键修改] 配置混合特征参数 ---
    # 这些参数必须和你真实数据的列对应
    # 例如：你有3个ID类特征(UserID, City, Gender)，词表大小分别是 10000, 500, 2
    conf.cat_feature_vocab_sizes = [10000, 500, 2] 
    
    # 例如：你有2个数值特征(Age, ReviewCount)
    conf.num_feature_size = 2
    
    # 开启 CTR 联合训练
    conf.enable_ctr = True 
    # -------------------------------
    
    set_seed(conf.seed)
    
    # 4. 准备数据
    # 注意：get_dataloaders 内部现在使用的是 HybridDataset
    # 它会根据 conf.cat_feature_vocab_sizes 自动生成/读取对应的数据列
    if is_main_process():
        print("Preparing Hybrid Data (Sequence + User Profile)...")
        
    train_dl, val_dl = get_dataloaders(conf)
    
    if is_main_process():
        print(f"Vocab Stats -> Items: {conf.num_items}, "
              f"Cat Feats: {len(conf.cat_feature_vocab_sizes)}, "
              f"Num Feats: {conf.num_feature_size}")
    
    # 5. 初始化模型
    # 模型会根据 conf 自动创建 FeatureTokenizer 和 HSTU Backbone
    model = UniGCRModel(conf)
    
    # 6. 初始化 Trainer
    # Trainer 内部会读取 args.deepspeed_config 并显式加载 JSON
    trainer = UniGCRTrainer(
        config=conf,
        args=args, # 务必传入 args
        model=model,
        train_loader=train_dl,
        val_loader=val_dl
    )
    
    if is_main_process():
        mode_str = "Joint Training (GR + CTR)" if conf.enable_ctr else "GR Only"
        print(f"Start Training... Mode: [{mode_str}]")
        print(f"DeepSpeed Config: {args.deepspeed_config}")
    
    # 7. 开始训练
    trainer.train()

if __name__ == "__main__":
    main()
