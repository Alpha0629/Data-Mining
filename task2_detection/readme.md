# Task 2: 图像异常检测任务

基于深度学习的二分类项目，实现对坚果和拉链异常的检测。使用从头训练的 ResNet18 模型，包含训练、评估和 t-SNE 特征可视化功能。通过固定随机种子，保证结果可复现。

## 使用方法

### 1. 数据集

数据需要按照下面的结构组织：
```
task2_detection/
├── dataset/
│   ├── hazelnut/          # 坚果数据集（可选）
│   │   ├── train/
│   │   │   ├── good/      # 正常样本
│   │   │   └── bad/       # 异常样本
│   │   └── test/
│   │       ├── good/
│   │       └── bad/
│   └── zipper/            # 拉链数据集
│       ├── train/
│       │   ├── good/
│       │   └── bad/
│       └── test/
│           ├── good/
│           └── bad/
├── nuts_only.py          # 坚果检测脚本
├── zipper_only.py        # 拉链检测脚本
└── ...
```

### 2. 运行程序

先切换到项目目录：
```bash
cd task2_detection
```

然后根据要检测的数据集选择对应的脚本：

**检测坚果异常：**
```bash
python nuts_only.py
```

**检测拉链异常：**
```bash
python zipper_only.py
```

### 3. 参数配置

如果想调整训练参数，可以在对应脚本的 `main()` 函数里修改：

#### 数据集相关
- `"dataset"`: 数据集目录路径（默认：`"dataset"`）
- `img_size`: 图像尺寸（默认：`64`）

#### 模型相关
在 `main()` 函数中可以选择不同的模型：
```python
model = ResNet18(num_classes=2, pretrained=False, freeze_backbone=False)
# model = CNN(num_classes=2, dropout=0.1)
# model = MLP(num_classes=2, dropout=0.1)
```

#### 训练相关
- `batch_size`: 批次大小（默认：`32`）
- `learning_rate`: 学习率（默认：`0.001`）
- `num_epochs`: 训练轮数（默认：`60`）

#### t-SNE 可视化相关
在 `visualize_tsne()` 方法中：
- `max_samples`: 用于可视化的最大样本数（默认：`1000`）

## 结果输出

### 1. 控制台输出

训练过程中会在控制台打印这些信息：
- 训练集和测试集大小
- 类别分布和类别权重
- 每个 epoch 的训练损失
- **测试集评估结果**：
  - 准确率（百分比）
  - 分类说明（good=0, bad=1）

示例输出：
```
============================================================
测试集评估结果
============================================================
准确率: 85.67% (123/144)
分类: good=0, bad=1
预测结果已保存到: ckpt/20251218_232252_zipper_pred.csv
============================================================
```

### 2. 输出文件

训练完成后，所有结果会保存在 `ckpt/` 目录下。文件名格式是：`{时间戳}_{数据集类型}_{类型}.{扩展名}`

#### 2.1 日志文件 (`{timestamp}_{dataset}_logging.log`)
- 记录了整个训练过程
- 每个 epoch 的损失值都有
- 测试集评估结果
- 还有文件保存路径

#### 2.2 预测结果文件 (`{timestamp}_{dataset}_pred.csv`)
- CSV 格式，两列：`true` 和 `pred`
- `true`: 真实标签（0=good, 1=bad）
- `pred`: 预测标签（0=good, 1=bad）
- 可以拿来进一步分析

#### 2.3 t-SNE 可视化图 (`{timestamp}_tsne_visualization.png`)
- 把模型提取的特征降维到 2D 空间画出来
- 不同颜色代表不同的类别（good/bad）
- 可以直观看看模型分类效果怎么样

### 3. 结果文件位置

所有输出文件保存在：
```
task2_detection/ckpt/
├── {timestamp}_nuts_logging.log          # 或 zipper
├── {timestamp}_nuts_pred.csv              # 或 zipper
└── {timestamp}_tsne_visualization.png
```

其中 `{timestamp}` 格式为 `YYYYMMDD_HHMMSS`，例如 `20251218_232252`。

## 评估指标说明

- **准确率 (Accuracy)**: 就是正确分类的样本占总数多少，越高越好
- 标签说明：`good=0`（正常），`bad=1`（异常）

## 注意事项

1. **数据集路径**: 记得把数据放在 `dataset/` 目录下，要有 `hazelnut` 或 `zipper` 文件夹，然后每个数据集下面要有 `train/` 和 `test/`，每个里面再分 `good/` 和 `bad/` 文件夹
2. **模型选择**: 
   - `ResNet18`: 默认使用该模型，从头训练，效果较好，但是运行速度非常慢
   - `CNN`: 折中选项，速度一般，效果也一般
   - `MLP`: 速度快，但是效果不好，基本无法识别异常样本
3. **训练时间**: 看数据集大小和 GPU，一般几分钟到几十分钟不等
4. **随机种子**: 固定了随机种子（42），保证结果可以复现
