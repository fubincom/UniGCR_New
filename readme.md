```Text
UniGCR_Project/
├── data/                   # 存放原始数据
├── src/
│   ├── __init__.py
│   ├── config.py           # 配置类
│   ├── data.py             # Dataset 和 DataLoader 逻辑
│   ├── model.py            # Uni-GCR 模型架构
│   ├── trainer.py          # 封装训练和评估流程
│   └── utils.py            # 评价指标 NDCG/Hit 等
└── run.py                  # 启动脚本
```Text

如何使用这套代码
准备数据：将下载好的 reviews_Beauty_5.json.gz 放入 data/ 目录。
运行：python run.py


代码重构的亮点总结
高内聚低耦合：

UniGCRModel 只负责网络的前向传播，不再包含 Loss 计算逻辑。

AmazonSeqDataset 封装了所有脏数据处理逻辑（JSON 解析、Sequence 截断），外部拿到的是干净的 Tensor。

UniGCRTrainer 接管了 Loop 循环、Device 迁移、Loss 组合。这使得如果你想换个模型（比如把 HSTU 换成 BERT），只需要改 model.py，而不需要动训练逻辑。

符合 PyTorch 原生范式：

使用了 torch.utils.data.Dataset 进行懒加载（在 __init__ 做索引，__getitem__ 做 Tensor 转换），这比预先存成 .pkl 文件更节省磁盘空间，也更灵活。

易于扩展 (Avazu 支持)：

如果要支持 Avazu，只需要在 src/data.py 中新写一个 AvazuDataset 类，然后在 run.py 中替换即可。Trainer 和 Model 的大部分代码无需修改（只需要在 Model 的 get_user_state 中适配 Avazu 的 Feature 处理方式）。
