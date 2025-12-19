# Task 1: K-means 聚类项目

使用 K-means 算法对图像数据进行聚类分析，通过 scikit-learn 实现。同时，通过固定随机种子，保证结果可复现。

## 使用方法

### 1. 数据集

确保数据文件结构如下：
```
task1_cluster/
├── datasets/
│   ├── dataset/          # 图像数据所在的目录
│   └── cluster_labels.json  # 编号到label的映射
├── main.py
├── data_loader.py
└── viz.py
```

### 2. 运行程序

执行
```bash
cd task1_cluster
```
切换到当前项目的目录下，然后执行

```bash
python main.py
```

### 3. 参数配置

在 `main.py` 的 `if __name__ == "__main__"` 部分可修改以下参数：

- `dataset_dir`: 数据集目录路径（默认：`"datasets/dataset"`）
- `labels_path`: 标签文件路径（默认：`"datasets/cluster_labels.json"`）
- `n_clusters`: 聚类数量（默认：`6`）
- `max_iters`: 最大迭代次数（默认：`300`）
- `random_state`: 随机种子（默认：`42`）
- `n_init`: K-means 初始化次数（默认：`10`）
- `visualize`: 是否生成可视化（默认：`True`）
- `output_dir`: 可视化结果输出目录（默认：`"visualizations"`）
- `reduction_method`: 降维方法（`"pca"` 或 `"tsne"`，默认：`"pca"`）

示例：
```python
results = run_kmeans(
    dataset_dir="datasets/dataset",
    labels_path="datasets/cluster_labels.json",
    n_clusters=6,
    max_iters=300,
    random_state=42,
    n_init=10,
    visualize=True,
    output_dir="visualizations",
    reduction_method="pca"  # 或 "tsne"（更慢但效果通常更好）
)
```

## 结果输出

### 1. 控制台输出

程序运行时会输出以下信息：
- 数据集大小和类别数量
- 特征提取信息
- K-means 训练过程（迭代次数、惯性值）
- **评估指标**：
  - **ARI (Adjusted Rand Index)**: 调整兰德指数
  - **NMI (Normalized Mutual Information)**: 标准化互信息
  - **Accuracy**: 准确率

示例输出：
```
============================================================
Clustering Results
============================================================
Adjusted Rand Index (ARI): 0.8234
Normalized Mutual Information (NMI): 0.7891
Accuracy: 0.8567
============================================================
```

### 2. 可视化结果

如果 `visualize=True`，程序会在 `visualizations/` 目录下生成以下可视化文件：

#### 2.1 散点图 (`clustering_scatter_pca.png` 或 `clustering_scatter_tsne.png`)
- 左侧：预测的聚类结果（使用 PCA 或 t-SNE 降维到 2D）
- 右侧：真实的类别标签
- 用于直观比较聚类效果

#### 2.2 混淆矩阵 (`confusion_matrix.png`)
- 显示真实类别与预测聚类之间的对应关系

#### 2.3 聚类样本图 (`cluster_samples.png`)
- 展示每个聚类中的代表性样本图像
- 每行对应一个聚类，显示该聚类中的多个样本

### 3. 结果文件位置

所有可视化结果保存在：
```
task1_cluster/visualizations/
├── clustering_scatter_pca.png    # 或 clustering_scatter_tsne.png
├── confusion_matrix.png
└── cluster_samples.png
```

## 评估指标说明

- **ARI (Adjusted Rand Index)**: 范围 [-1, 1]，值越大越好，1 表示完全匹配
- **NMI (Normalized Mutual Information)**: 范围 [0, 1]，值越大越好，1 表示完全匹配
- **Accuracy**: 范围 [0, 1]，值越大越好，表示正确分类的样本比例

## 注意事项

1. **数据路径**: 确保 `datasets/dataset/` 目录中包含所有图像文件，且 `cluster_labels.json` 中的文件名与实际图像文件名匹配
2. **降维方法**: 
   - `pca`: 速度快，适合快速查看结果
   - `tsne`: 速度慢但可视化效果通常更好，能更好地展示数据的内在结构
3. **聚类数量**: `n_clusters` 应该与数据中的真实类别数量一致，以获得最佳效果
4. **随机种子**: 固定 `random_state` 可以确保结果可复现
