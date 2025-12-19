这是一个基于您提供的代码文件 (`nuts_only.py`, `zipper_only.py`, `model/*`, `data_loader/*`) 和目录结构生成的 `README.md`。

文档内容已根据您实际实现的\*\*有监督二分类（Good vs Bad）\*\*方案进行了重写，同时严格保持了您提供的 `README.md` 的格式结构。

-----

# 图像异常检测任务解决方案 (监督学习版)

## 作业要求

**Task 2: Image Anomaly Detection Task (图像异常检测任务)**

  - [cite\_start]使用监督学习方法（二分类）进行图像异常检测 [cite: 111, 112]
  - [cite\_start]基于“正常(Good)”和“异常(Bad)”样本训练分类模型 [cite: 25, 111]
  - [cite\_start]生成特征层 t-SNE 可视化图以分析类间分离度 [cite: 111]
  - [cite\_start]评估分类准确率 (Accuracy) [cite: 111]

## 数据集说明

  - **数据集类型**: MVTec AD 风格数据集
  - **类别**:
      - [cite\_start]`hazelnut` (榛子) [cite: 146]
      - [cite\_start]`zipper` (拉链) [cite: 112]
  - **数据结构**:
    ```
    dataset/
    ├── hazelnut/
    │   ├── train/
    │   │   ├── good/          # 正常样本 (Label=0)
    │   │   └── bad/           # 异常样本 (Label=1)
    │   └── test/              # 测试样本
    └── zipper/
        ├── train/
        │   ├── good/
        │   └── bad/
        └── test/
    ```
  - **标签定义**:
      - `Good`: 0
      - [cite\_start]`Bad`: 1 [cite: 29]

## 文件结构

```
Image_Anomaly_Detection/
[cite_start]├── ckpt/                        # 输出目录：日志、CSV预测结果、t-SNE图 [cite: 111]
├── dataset/                     # 数据集根目录
│   ├── hazelnut/
│   └── zipper/
├── data_loader/                 # 数据加载模块
[cite_start]│   ├── data_loader_nuts.py      # 榛子数据加载器 [cite: 146]
[cite_start]│   └── data_loader_zipper.py    # 拉链数据加载器 [cite: 112]
├── model/                       # 模型定义
[cite_start]│   ├── CNN.py                   # 简单 CNN 模型 [cite: 61]
[cite_start]│   ├── MLP.py                   # MLP 模型 [cite: 1]
[cite_start]│   └── ResNet18.py              # ResNet18 模型 (主模型) [cite: 90]
[cite_start]├── nuts_only.py                 # 榛子训练主程序 [cite: 111]
[cite_start]├── zipper_only.py               # 拉链训练主程序 [cite: 112]
└── README.md                    # 说明文档
```

## 安装依赖

```bash
pip install torch torchvision numpy matplotlib scikit-learn tqdm pillow
```

## 使用方法

### 基本使用

1.  **配置参数**（在 `nuts_only.py` 或 `zipper_only.py` 的 `main` 函数中修改）:

    ```python
    # 示例：nuts_only.py
    train_dataset = ClassDataset("dataset", split="train", img_size=64)
    # ...
    model = ResNet18(num_classes=2, pretrained=False, freeze_backbone=False)
    # ...
    [cite_start]trainer.train(num_epochs=50) # 调整训练轮数 [cite: 111]
    ```

2.  **运行程序**:

    ```bash
    # 训练榛子模型
    python nuts_only.py

    # 训练拉链模型
    python zipper_only.py
    ```

### 处理不同类别

由于代码逻辑已分离，请直接运行对应的脚本：

  - `nuts_only.py` - 专门处理榛子类别
  - `zipper_only.py` - 专门处理拉链类别

## 解决方案详解

### 核心方法

本项目采用**有监督深度学习分类**方法，直接训练模型区分正常与异常图像：

1.  [cite\_start]**数据增强 (Data Augmentation)** [cite: 44, 149]

      - 为了在小样本下防止过拟合，训练集采用了强增强策略：
      - 随机裁剪 (`RandomResizedCrop`)
      - 随机水平翻转 (`RandomHorizontalFlip`)
      - 随机旋转 (`RandomRotation`, 12度)
      - 颜色抖动 (`ColorJitter`: 亮度/对比度/饱和度/色相)

2.  [cite\_start]**模型架构** [cite: 90]

      - **Backbone**: ResNet18 (包含 Encoder 和 Classifier)
      - **Encoder**: 提取图像的高维特征 (512维)
      - **Classifier**: 全连接层将特征映射到 2 个类别 (Good/Bad)

3.  [cite\_start]**训练策略** [cite: 111]

      - 损失函数: 交叉熵损失 (`CrossEntropyLoss`)
      - 优化器: Adam (学习率 0.001)
      - 训练过程中利用 `tqdm` 监控 Loss 变化

4.  [cite\_start]**特征可视化** [cite: 111]

      - 使用 t-SNE 算法对 Encoder 提取的高维特征进行降维
      - 在 2D 平面上展示 Good 和 Bad 样本的分布情况，直观评估模型的特征分离能力

### 技术特点

  - **监督学习**: 利用异常样本标签进行直接监督，分类边界更明确
  - **强数据增强**: 提升模型对光照、位置变化的鲁棒性
  - **特征降维分析**: 通过 t-SNE 验证模型是否学到了具有判别性的特征
  - **自动日志记录**: 自动保存训练日志 (`.log`) 和预测结果 (`.csv`)

## 输出结果

程序会生成以下输出（保存在 `ckpt/` 目录）：

1.  **控制台/日志输出**:

      - 训练集/测试集样本数量
      - 每个 Epoch 的平均 Loss
      - 最终测试集的分类准确率

2.  **可视化结果**:

      - [cite\_start]**t-SNE 分布图** (`*_tsne_visualization.png`): [cite: 111]

          - 蓝色点：Good (正常)
          - 红色点：Bad (异常)
          - 展示两类样本在特征空间中的分离程度

      - [cite\_start]**预测结果表** (`*_pred.csv`): [cite: 111]

          - 包含测试集中每一张图片的真实标签 (`true`) 和预测标签 (`pred`)

## 评估指标

  - [cite\_start]**Accuracy (准确率)**: 图像级分类的主要评估指标 [cite: 111]

      - 计算公式: (预测正确的样本数 / 总样本数) \* 100%
      - 范围: [0, 100]

  - **t-SNE 可分性**:

      - 定性指标，观察红蓝点簇是否明显分离

## 预期结果

根据当前 ResNet18 模型配置，预期性能：

  - **Accuracy**: 在经过 50-60 轮训练后，预期达到较高的分类准确率（\>90%）。
  - **可视化**: t-SNE 图中，蓝色簇（正常）和红色簇（异常）应有较清晰的边界。

## 技术细节

### 特征提取流程

1.  [cite\_start]图像预处理：Resize 到 64x64，转换为 Tensor [cite: 150]
2.  [cite\_start]输入 ResNet18 模型，经过 `conv1` -\> `avgpool` 提取特征 [cite: 90]
3.  [cite\_start]获得 512 维的特征向量 (`get_features` 方法) [cite: 108]

### 分类与预测

1.  特征向量输入全连接层 (`fc`)
2.  输出 2 维 Logits
3.  [cite\_start]使用 `torch.max` 获取预测类别 (0 或 1) [cite: 111]

### t-SNE 生成

1.  提取测试集所有样本的 Encoder 特征
2.  使用 `sklearn.manifold.TSNE` 将 512 维降至 2 维
3.  [cite\_start]使用 `matplotlib` 绘制散点图，根据真实标签着色 [cite: 111]

## 注意事项

1.  [cite\_start]**GPU 支持**: 代码会自动检测 CUDA，建议使用 GPU 加速训练 [cite: 111]
2.  [cite\_start]**数据路径**: 确保 `dataset` 目录结构正确，包含 `good` 和 `bad` 子文件夹 [cite: 146]
3.  [cite\_start]**中文显示**: 代码已配置 matplotlib 的中文字体 (SimHei 等)，防止图表乱码 [cite: 111]

## 参数调整建议

  - [cite\_start]**model**: 可在 `main()` 中切换为 `CNN` 或 `MLP` 进行实验 [cite: 111]
  - [cite\_start]**img\_size**: 默认为 64，可调整为 128 或 256 以保留更多细节 [cite: 146]
  - [cite\_start]**num\_epochs**: 如果 Loss 收敛较慢，可增加训练轮数（当前 Nuts=50, Zipper=60） [cite: 111, 112]
  - **learning\_rate**: 默认 0.001，可根据 loss 曲线调整

## 扩展建议

1.  [cite\_start]**模型替换**: 尝试使用预训练权重的 ResNet (`pretrained=True`) 加快收敛 [cite: 102]
2.  [cite\_start]**类别权重**: 如果正负样本极度不平衡，可取消代码中 `class_weights` 的注释 [cite: 111]
3.  **更深的网络**: 对于复杂的纹理异常，可以尝试 ResNet50 或更复杂的 Attention 机制