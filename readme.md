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
