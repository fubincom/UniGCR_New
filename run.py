import sys
import os
sys.path.append(os.getcwd()) # 确保能找到 src

from src.config import UniGCRConfig
from src.data import get_dataloaders
from src.model import UniGCRModel
from src.trainer import UniGCRTrainer
from src.utils import set_seed

def main():
    args = parse_args()
    setup_distributed()
    
    # 你可以在这里手动修改 config，或者通过 argparse 传参覆盖
    conf = UniGCRConfig()
    conf.enable_ctr = True # 或者 False 来进行消融实验 (Ablation Study)
    
    set_seed(conf.seed)
    train_dl, val_dl = get_dataloaders(conf)
    
    model = UniGCRModel(conf)
    
    trainer = UniGCRTrainer(conf, args, model, train_dl, val_dl)
    
    if is_main_process():
        mode_str = "Joint Training (GR + CTR)" if conf.enable_ctr else "GR Only"
        print(f"Start Training... Mode: [{mode_str}]")
    
    trainer.train()

if __name__ == "__main__":
    main()

