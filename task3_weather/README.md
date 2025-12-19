# Task 3: 时间序列预测任务

使用 LSTM 回归模型进行天气温度预测。支持单层/双层 LSTM 架构，可以选择增加额外的简单注意力机制，预测温度变化量并评估预测效果。通过固定随机种子，保证结果可复现。

## 使用方法

### 1. 数据集

确保数据文件结构如下：
```
task3_weather/
├── dataset/
│   └── weather.csv       # 天气数据 CSV 文件
├── main.py
├── preprocess.py
└── ...
```

数据文件 `weather.csv` 应包含天气相关的特征列，其中 `OT` 列是目标变量（温度）。

### 2. 运行程序

先切换到项目目录：
```bash
cd task3_weather
```

然后运行主程序：
```bash
python main.py
```

### 3. 参数配置

如果想调整模型和训练参数，可以在 `main.py` 的 `main()` 函数里修改：

#### 模型配置
- `NUM_LAYERS`: LSTM 层数（`1` 或 `2`，默认：`2`）
  - `1`: 单层 LSTM
  - `2`: 双层 LSTM
- `USE_ATTENTION`: 是否使用注意力机制（`True` 或 `False`，默认：`True`）

#### 数据相关
- `seq_len`: 序列长度，即用多少时间步预测下一个时间步（默认：`12`）
- `target_col`: 目标列名（默认：`"OT"`）
- `path`: 数据文件路径（默认：`"./dataset/weather.csv"`）

#### 训练相关
- `batch_size`: 批次大小（默认：`64`）
- `learning_rate`: 学习率（默认：`1e-3`）
- `epochs`: 训练轮数（默认：`30`）

示例配置：
```python
NUM_LAYERS = 2  # 双层 LSTM
USE_ATTENTION = True  # 使用注意力机制
seq_len = 12  # 序列长度
epochs = 30  # 训练轮数
```

## 结果输出

### 1. 控制台输出

训练过程中会在控制台打印这些信息：
- 模型配置（LSTM 层数、是否使用注意力）
- 设备信息（CPU/GPU）
- 数据加载信息
- 每个 epoch 的训练损失
- **测试集评估结果**：
  - MAE（平均绝对误差）
  - RMSE（均方根误差）
  - 分别显示标准化后和原始尺度的结果

示例输出：
```
====== 测试集结果（OT变化量预测，原始尺度） ======
MAE  = 0.1234
RMSE = 0.2345
```

### 2. 输出文件

训练完成后，所有结果会保存在 `ckpt/{timestamp}/` 目录下（`{timestamp}` 格式为 `YYYYMMDD_HHMMSS`）。

#### 2.1 日志文件 (`log_{timestamp}.txt`)
- 记录了整个训练过程
- 包括模型配置、数据信息、每个 epoch 的损失值
- 测试集评估结果（MAE、RMSE）
- 文件保存路径信息

#### 2.2 模型文件 (`model_{timestamp}.pth`)
- 训练好的模型权重
- 可以用 `torch.load()` 加载用于推理

#### 2.3 预测结果文件 (`predictions_{timestamp}.csv`)
- CSV 格式，包含以下列：
  - `Current_OT`: 当前时刻的 OT 值（温度）
  - `True_Delta_OT`: 真实的 OT 变化量
  - `Predicted_Delta_OT`: 预测的 OT 变化量
  - `Delta_Error`: 变化量的误差
  - `Delta_Absolute_Error`: 变化量的绝对误差
  - `True_OT_Next`: 真实的下一时刻 OT 值
  - `Predicted_OT_Next`: 通过预测变化量计算的下一时刻 OT 值
  - `OT_Error`: OT 值的误差
  - `OT_Absolute_Error`: OT 值的绝对误差
- 可以拿来进一步分析和评估

#### 2.4 可视化图 (`visualization_{timestamp}.png`)
- 包含两个子图：
  - 上图：OT 变化量的预测对比（真实值 vs 预测值）
  - 下图：通过变化量计算的 OT 值对比（真实值 vs 预测值）
- 可以直观看看模型预测效果怎么样

### 3. 结果文件位置

所有输出文件保存在：
```
task3_weather/ckpt/{timestamp}/
├── log_{timestamp}.txt
├── model_{timestamp}.pth
├── predictions_{timestamp}.csv
└── visualization_{timestamp}.png
```

其中 `{timestamp}` 格式为 `YYYYMMDD_HHMMSS`，例如 `20251218_212955`。

## 评估指标说明

- **MAE (Mean Absolute Error)**: 平均绝对误差，就是预测值和真实值差的绝对值的平均，越小越好
- **RMSE (Root Mean Squared Error)**: 均方根误差，对误差平方后求平均再开根号，越小越好，对大误差更敏感

程序会输出两个尺度的评估结果：
1. **标准化后的结果**：模型内部使用的标准化尺度
2. **原始尺度的结果**：还原到原始温度单位的结果（更直观）

## 注意事项

1. **数据格式**: 确保 `weather.csv` 文件格式正确，包含 `OT` 列作为目标变量
2. **模型选择**: 
   - 单层 LSTM：参数少，训练快，但表达能力可能不够
   - 双层 LSTM：参数多，训练慢，但通常效果更好
   - 注意力机制：可以提升模型对重要时间步的关注，但会增加计算量
3. **序列长度**: `seq_len` 表示用过去多少个时间步来预测下一个时间步，可以根据数据特点调整
4. **训练时间**: 看数据量和 GPU，一般几分钟到十几分钟不等
5. **随机种子**: 固定了随机种子（42），保证结果可以复现

## 其他工具

### 模型对比工具

项目还提供了 `gen_table.py` 和 `viz.py` 用于对比不同模型配置的结果：

- `gen_table.py`: 从多个日志文件中提取结果，生成对比表格
- `viz.py`: 可视化多个模型的预测结果对比
