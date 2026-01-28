统一生成式检索 (Generative Retrieval, GR) 与点击率预测 (CTR) 的多任务学习框架。其核心思想是利用共享的序列表达能力，同时驱动两种不同性质的推荐任务。


```Text
UniGCR_Repo/
├── ds_config.json          # DeepSpeed 配置文件
├── requirements.txt        # 依赖列表 (含安装顺序说明)
├── run.py                  # 启动入口
└── src/
    ├── __init__.py
    ├── config.py           # 全局配置 (Dataclass)
    ├── data.py             # UniversalDataset & DataLoader
    ├── grid_utils.py       # Semantic ID 映射与反查工具
    ├── model.py            # 模型核心 (InputLayer, HSTU, Heads)
    ├── trainer.py          # 训练循环, EarlyStop, Eval
    └── utils.py            # Metrics, Distributed Utils


配置
PyTorch 带CUDA
# 示例：安装 PyTorch 2.1 + CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
安装Flash Attention 2
pip install packaging ninja
pip install flash-attn --no-build-isolation

安装其他依赖包
# 安装 DeepSpeed, Scikit-learn 等
pip install deepspeed numpy pandas scikit-learn tqdm wget triton

# 安装 Meta 的 generative-recommenders (HSTU)
pip install git+https://github.com/facebookresearch/generative-recommenders.git@main

如何使用这套代码
准备数据：将下载好的 reviews_Beauty_5.json.gz 放入 data/ 目录。
# 指定可见设备
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 run.py \
    --deepspeed \
    --deepspeed_config ds_config.json

评价指标完善：
GR: 保持了 HitRate 和 NDCG。
CTR: 新增了 AUC 和 LogLoss。通过 gather_tensors 确保了在多 GPU 环境下，AUC 是基于全局数据计算的，而不是局部 AUC 的平均值（那是不准确的）。

配置灵活性 (enable_ctr)：
enable_ctr=False: 纯序列推荐模型（类似 SASRec/BERT4Rec） ； enable_ctr=True: Uni-GCR 完整模式，联合优化排序能力。





代码重构的亮点总结
高内聚低耦合：

UniGCRModel 只负责网络的前向传播，不再包含 Loss 计算逻辑。

AmazonSeqDataset 封装了所有脏数据处理逻辑（JSON 解析、Sequence 截断），外部拿到的是干净的 Tensor。

UniGCRTrainer 接管了 Loop 循环、Device 迁移、Loss 组合。这使得如果你想换个模型（比如把 HSTU 换成 BERT），只需要改 model.py，而不需要动训练逻辑。

符合 PyTorch 原生范式：

使用了 torch.utils.data.Dataset 进行懒加载（在 __init__ 做索引，__getitem__ 做 Tensor 转换），这比预先存成 .pkl 文件更节省磁盘空间，也更灵活。

易于扩展 (Avazu 支持)：

如果要支持 Avazu，只需要在 src/data.py 中新写一个 AvazuDataset 类，然后在 run.py 中替换即可。Trainer 和 Model 的大部分代码无需修改（只需要在 Model 的 get_user_state 中适配 Avazu 的 Feature 处理方式）。
