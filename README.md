# Data-Mining Project
**2025年数据挖掘导论大作业**

**刘嘉明23371007，曹恕晗 23182615，唐楚然 23182613**

## 环境要求

### 版本
Python 3.12
CUDA  12.8

### 硬件配置
本项目在以下硬件配置下运行和测试：
- **GPU**: NVIDIA RTX 5090 (32GB)
- **CPU**: Intel Xeon Platinum 8470Q (25 vCPU)

### 依赖安装
安装项目所需的所有依赖包：

```bash
pip install -r requirements.txt
```

### 主要依赖
- **numpy**
- **pandas**
- **torch**
- **torchvision**
- **scikit-learn**
- **matplotlib**
- **Pillow**
- **tqdm**
- **joblib**

## 项目结构

### task1_cluster - 聚类任务	
使用 K-means 算法对图像数据进行聚类分析，通过 scikit-learn 实现，包含特征提取、聚类评估（ARI、NMI、准确率）和可视化。

### task2_detection - 图像异常检测任务
基于深度学习的二分类项目，实现对坚果和拉链异常的检测。使用从头训练的 ResNet18 模型，包含训练、评估和 t-SNE 特征可视化功能。

### task3_weather - 时间序列预测任务
使用 LSTM 回归模型进行天气温度预测。支持单层/双层 LSTM 架构，可以选择增加额外的简单注意力机制，预测温度变化量并评估预测效果。

### task4_thyroid - 无监督疾病判断任务
基于 One-Class SVM 的异常检测项目，用于检测甲状腺数据中的异常样本。包含网格搜索优化、阈值选择和 t-SNE 可视化功能。

### task5_unified_frame - 跨模态统一检测框架
整合了 task2 的检测功能，使用预训练的 ResNet 模型提取图像特征，然后通过 One-Class SVM 进行异常检测。支持 nuts 和 zipper 两种数据集，以及 ResNet18 和 ResNet50 两种特征提取模型。从而实现了 task2 和 task4 的统一框架，详细内容和其他方案请见论文。
