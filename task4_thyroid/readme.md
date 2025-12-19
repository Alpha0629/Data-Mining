# Task 4: 无监督疾病判断任务

基于 One-Class SVM 的异常检测项目，用于检测甲状腺数据中的异常样本。包含网格搜索优化、阈值选择和 t-SNE 可视化功能。通过固定随机种子，保证结果可复现。

## 使用方法

### 1. 数据集

确保数据文件结构如下：
```
task4_thyroid/
├── datasets/
│   ├── train-set.csv     # 训练集（只包含正常样本，无标签列）
│   └── test-set.csv      # 测试集（最后一列是标签：0=正常，1=异常）
├── main.py
├── data_loader.py
└── ...
```

**数据格式说明**：
- `train-set.csv`: 训练集只包含正常样本，所有列都是特征，没有标签列
- `test-set.csv`: 测试集包含正常和异常样本，最后一列是标签（0=正常，1=异常），其余列是特征

### 2. 运行程序

先切换到项目目录：
```bash
cd task4_thyroid
```

然后运行主程序：
```bash
python main.py
```

### 3. 参数配置

如果想调整参数，可以在 `main.py` 的 `if __name__ == "__main__"` 部分修改：

#### 阈值分位数
- `THRESHOLD_PERCENTILE`: 用于确定异常阈值的分位数（默认：`97`）
  - 表示使用训练集异常分数的第 97 分位数作为阈值
  - 可以修改为其他值，如 90, 95, 98, 99 等
  - 值越大，阈值越高，检测到的异常越少

#### 网格搜索参数
在 `main()` 函数中可以修改网格搜索的参数范围：
```python
nu_list = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]  # nu 参数列表
kernel_list = ['rbf', 'linear', 'poly', 'sigmoid']  # 核函数类型
gamma_list = ['scale', 'auto', 0.001, 0.01, 0.1, 1.0]  # gamma 参数（仅用于 RBF 核）
```

**参数说明**：
- `nu`: 异常值比例的上界（0-1 之间），表示最多多少比例的样本可能是异常
- `kernel`: 核函数类型
  - `rbf`: 径向基函数核（最常用）
  - `linear`: 线性核
  - `poly`: 多项式核
  - `sigmoid`: Sigmoid 核
- `gamma`: RBF 核的参数，控制单个样本的影响范围（仅对 RBF 核有效）

## 结果输出

### 1. 控制台输出

训练过程中会在控制台打印这些信息：
- 数据集大小和标签分布
- 网格搜索进度（当前组合/总组合数）
- 每个参数组合的评估结果
- **最佳结果**：
  - 最佳参数组合（nu, kernel, gamma）
  - 阈值分位数和阈值值
  - AUC、F1 Score、Accuracy

示例输出：
```
最佳结果:
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

#### 2.1 日志文件 (`oneclass_svm_training_p{percentile}_{timestamp}.log`)
- 记录了整个网格搜索过程
- 包括每个参数组合的训练和评估结果
- 最佳结果信息
- 文件保存路径信息

#### 2.2 网格搜索结果汇总 (`oneclass_svm_grid_search_summary_p{percentile}_{timestamp}.csv`)
- CSV 格式，包含所有参数组合的评估结果
- 列包括：`nu`, `kernel`, `gamma`, `percentile`, `threshold`, `auc`, `f1_score`, `accuracy`, `predictions_csv`
- 可以拿来分析不同参数组合的效果

#### 2.3 最佳预测结果文件 (`oneclass_svm_predictions_p{percentile}_nu{nu}_kernel{kernel}_gamma{gamma}_{timestamp}.csv`)
- CSV 格式，包含三列：
  - `anomaly_score`: 异常分数（分数越高越异常）
  - `true_label`: 真实标签（0=正常，1=异常）
  - `predicted_label`: 预测标签（0=正常，1=异常）
- 这是最佳参数组合的预测结果
- 程序会自动删除其他参数组合的预测结果文件，只保留最佳结果

### 3. 结果文件位置

所有输出文件保存在：
```
task4_thyroid/ckpt/
├── oneclass_svm_training_p{percentile}_{timestamp}.log
├── oneclass_svm_grid_search_summary_p{percentile}_{timestamp}.csv
└── oneclass_svm_predictions_p{percentile}_nu{nu}_kernel{kernel}_gamma{gamma}_{timestamp}.csv
```

其中 `{timestamp}` 格式为 `YYYYMMDD_HHMMSS`，例如 `20251219_002523`。

## 评估指标说明

- **AUC (Area Under ROC Curve)**: ROC 曲线下面积，范围 [0, 1]，越大越好，1 表示完美分类
- **F1 Score**: F1 分数，精确率和召回率的调和平均，范围 [0, 1]，越大越好
- **Accuracy**: 准确率，正确分类的样本占总数的比例，范围 [0, 1]，越大越好

程序会根据 **F1 Score** 来选择最佳参数组合。

## 注意事项

1. **数据格式**: 
   - 训练集只包含正常样本，所有列都是特征
   - 测试集最后一列是标签（0=正常，1=异常），其余列是特征
2. **网格搜索时间**: 由于要遍历所有参数组合，网格搜索可能需要较长时间（几十分钟到几小时不等，取决于参数组合数量）
3. **阈值分位数**: 
   - 值越大，阈值越高，检测到的异常越少（更严格）
   - 值越小，阈值越低，检测到的异常越多（更宽松）
   - 建议从 95-99 之间尝试
4. **参数选择**: 
   - `nu` 值通常选择 0.01-0.3 之间
   - RBF 核通常效果最好，但计算量也最大
   - `gamma='scale'` 通常是 RBF 核的默认选择
5. **随机种子**: 固定了随机种子（42），保证结果可以复现

## 其他工具

### 可视化工具

项目还提供了两个可视化脚本：

#### 1. t-SNE 可视化 (`visualize_tsne.py`)
- 对测试集数据进行 t-SNE 降维可视化
- 可以直观看到正常样本和异常样本在特征空间中的分布
- 运行方式：
  ```bash
  python visualize_tsne.py
  ```

#### 2. 网格搜索结果可视化 (`visualize_grid_search.py`)
- 可视化网格搜索结果，对比不同参数组合的效果
- 可以生成热力图等可视化图表
- 运行方式：
  ```bash
  python visualize_grid_search.py
  ```
