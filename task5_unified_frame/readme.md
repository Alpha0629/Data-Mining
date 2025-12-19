# Task 5: 跨模态统一检测框架

整合了 task2 和 task4 的检测思路，对于 task2 使用预训练的 ResNet 模型提取图像特征，然后通过 One-Class SVM 进行异常检测。支持 nuts 和 zipper 两种数据集，以及 ResNet18 和 ResNet50 两种特征提取模型。从而实现了 task2 和 task4 的统一框架，详细内容和其他方案请见论文。通过固定随机种子，保证结果可复现。

## 使用方法

### 1. 数据集

确保数据文件结构如下（与 task2 相同）：
```
task5_unified_frame/
├── dataset/              # 可选：如果数据集在这里
│   ├── hazelnut/        # 坚果数据集（可选）
│   │   ├── train/
│   │   │   ├── good/    # 正常样本
│   │   │   └── bad/     # 异常样本
│   │   └── test/
│   │       ├── good/
│   │       └── bad/
│   └── zipper/          # 拉链数据集
│       ├── train/
│       │   ├── good/
│       │   └── bad/
│       └── test/
│           ├── good/
│           └── bad/
├── detection.py
└── ...
```

**注意**：程序会自动尝试多个路径来查找数据集：
1. `task5_unified_frame/dataset/`
2. `task2_detection/dataset/`
3. 当前工作目录下的 `dataset/`

### 2. 运行程序

先切换到项目目录：
```bash
cd task5_unified_frame
```

然后运行主程序：
```bash
python detection.py
```

### 3. 参数配置

如果想调整参数，可以在 `detection.py` 的 `if __name__ == "__main__"` 部分修改：

#### 数据集类型
- `DATASET_TYPE`: 选择要检测的数据集（`"nuts"` 或 `"zipper"`，默认：`"zipper"`）

#### 特征提取模型
- `MODEL_TYPE`: 选择用于特征提取的模型（`"resnet18"` 或 `"resnet50"`，默认：`"resnet50"`）
  - `resnet18`: 特征维度 512，参数量约 11M，速度快，适合快速实验
  - `resnet50`: 特征维度 2048，参数量约 25M，特征提取能力更强，可能获得更好的异常检测效果，但计算量更大

#### 阈值分位数列表
- `THRESHOLD_PERCENTILE_LIST`: 阈值分位数列表，程序会对每个分位数进行网格搜索（默认：`[95, 96, 97, 98, 99, 100]`）
  - 可以修改为其他值列表，如 `[90, 95, 99]`
  - 值越大，阈值越高，检测到的异常越少

#### 网格搜索参数
在 `main()` 函数中可以修改网格搜索的参数范围：
```python
nu_list = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]  # nu 参数列表
kernel_list = ['rbf', 'linear', 'poly', 'sigmoid']  # 核函数类型
gamma_list = ['scale', 'auto', 0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 0.8, 1.0]  # gamma 参数（仅用于 RBF 核）
```

**参数说明**：
- `nu`: 异常值比例的上界（0-1 之间），表示最多多少比例的样本可能是异常
- `kernel`: 核函数类型
  - `rbf`: 径向基函数核（最常用）
  - `linear`: 线性核
  - `poly`: 多项式核
  - `sigmoid`: Sigmoid 核
- `gamma`: RBF 核的参数，控制单个样本的影响范围（仅对 RBF 核有效）

#### 其他参数
- `IMG_SIZE`: 图像尺寸（默认：`64`）

示例配置：
```python
DATASET_TYPE = "zipper"  # 使用拉链数据集
MODEL_TYPE = "resnet50"  # 使用 ResNet50 提取特征
THRESHOLD_PERCENTILE_LIST = [95, 96, 97, 98, 99, 100]  # 阈值分位数列表
```

## 结果输出

### 1. 控制台输出

训练过程中会在控制台打印这些信息：
- 数据集类型和模型类型
- 数据集路径和大小
- 特征提取进度
- 网格搜索进度（当前组合/总组合数）
- 每个参数组合的评估结果
- **最佳结果**（基于准确率）：
  - 最佳参数组合（nu, kernel, gamma, percentile）
  - 阈值值
  - AUC、F1 Score、Accuracy

示例输出：
```
最佳结果（基于准确率）:
  nu: 0.100
  kernel: rbf
  gamma: scale
  Percentile: 97
  Threshold: 0.123456
  AUC: 0.8234
  F1 Score: 0.7891
  Accuracy: 0.8567
```

### 2. 输出文件

训练完成后，所有结果会保存在 `ckpt/` 目录下。

#### 2.1 日志文件 (`training_{dataset_type}_{model_type}_p{percentile}_{timestamp}.log`)
- 记录了整个网格搜索过程
- 包括数据集类型、模型类型、特征提取信息
- 每个参数组合的训练和评估结果
- 最佳结果信息
- 文件保存路径信息

#### 2.2 网格搜索结果汇总 (`grid_search_summary_{dataset_type}_{model_type}_p{min}-{max}_{timestamp}.csv`)
- CSV 格式，包含所有参数组合的评估结果
- 列包括：`nu`, `kernel`, `gamma`, `percentile`, `threshold`, `auc`, `f1_score`, `accuracy`, `predictions_csv`
- 文件名中包含所有搜索的 percentile 范围（如 `p95-100`）
- 可以拿来分析不同参数组合的效果

#### 2.3 最佳预测结果文件 (`predictions_{dataset_type}_{model_type}_p{percentile}_nu{nu}_kernel{kernel}_gamma{gamma}_{timestamp}.csv`)
- CSV 格式，包含三列：
  - `anomaly_score`: 异常分数（分数越高越异常）
  - `true_label`: 真实标签（0=正常/good，1=异常/bad）
  - `predicted_label`: 预测标签（0=正常/good，1=异常/bad）
- 这是最佳参数组合的预测结果
- 程序会自动删除其他参数组合的预测结果文件，只保留最佳结果

### 3. 结果文件位置

所有输出文件保存在：
```
task5_unified_frame/ckpt/
├── training_{dataset_type}_{model_type}_p{percentile}_{timestamp}.log
├── grid_search_summary_{dataset_type}_{model_type}_p{min}-{max}_{timestamp}.csv
└── predictions_{dataset_type}_{model_type}_p{percentile}_nu{nu}_kernel{kernel}_gamma{gamma}_{timestamp}.csv
```

其中 `{timestamp}` 格式为 `YYYYMMDD_HHMMSS`，例如 `20251219_115742`。

## 评估指标说明

- **AUC (Area Under ROC Curve)**: ROC 曲线下面积，范围 [0, 1]，越大越好，1 表示完美分类
- **F1 Score**: F1 分数，精确率和召回率的调和平均，范围 [0, 1]，越大越好
- **Accuracy**: 准确率，正确分类的样本占总数的比例，范围 [0, 1]，越大越好

程序会根据 **准确率 (Accuracy)** 来选择最佳参数组合。

## 注意事项

1. **数据集路径**: 
   - 程序会自动尝试多个路径来查找数据集
   - 如果找不到，会报错提示
   - 数据集结构需要和 task2 一样，包含 `train/` 和 `test/` 子目录，每个子目录下有 `good/` 和 `bad/` 文件夹
2. **模型选择**: 
   - `ResNet18`: 速度快，特征维度 512，适合快速实验
   - `ResNet50`: 速度慢，特征维度 2048，特征提取能力更强，可能效果更好
3. **网格搜索时间**: 
   - 由于要遍历所有参数组合和阈值分位数，网格搜索可能需要很长时间（几小时到十几小时不等，取决于参数组合数量和 GPU 性能）
   - 总组合数 = 阈值分位数数量 × nu 数量 × (RBF 核的 gamma 数量 + 其他核数量)
4. **阈值分位数**: 
   - 值越大，阈值越高，检测到的异常越少（更严格）
   - 值越小，阈值越低，检测到的异常越多（更宽松）
   - 建议从 95-99 之间尝试
5. **参数选择**: 
   - `nu` 值通常选择 0.01-0.3 之间
   - RBF 核通常效果最好，但计算量也最大
   - `gamma='scale'` 通常是 RBF 核的默认选择
6. **随机种子**: 固定了随机种子（42），保证结果可以复现

## 框架特点

这个统一框架整合了 task2 和 task4 的方法：

1. **特征提取阶段**（类似 task2）：
   - 使用预训练的 ResNet 模型（ResNet18 或 ResNet50）提取图像特征
   - 只使用训练集中的正常样本（good）来提取特征

2. **异常检测阶段**（类似 task4）：
   - 使用 One-Class SVM 进行异常检测
   - 通过网格搜索找到最佳参数组合
   - 使用阈值分位数来确定异常阈值

3. **统一性**：
   - 可以处理不同类型的图像数据（nuts 和 zipper）
   - 可以选择不同的特征提取模型
   - 统一的评估指标和输出格式
