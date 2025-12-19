import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class ThyroidDataset(Dataset):
    """
    甲状腺数据集类，继承自 PyTorch Dataset
    """
    
    def __init__(self, data_dir='datasets', filename='train-set.csv', is_train=True):
        """
        初始化数据集
        
        参数:
            data_dir: 数据集目录路径
            filename: CSV文件名
            is_train: 是否为训练集（训练集的标签全为0）
        """
        file_path = Path(data_dir) / filename
        df = pd.read_csv(file_path)
        
        # 提取特征数据
        if is_train:
            # 训练集：所有列都是特征
            self.features = df.values.astype(np.float32)
            # 训练集的标签全为0
            self.labels = np.zeros(len(df), dtype=np.int64)
        else:
            # 测试集：最后一列是标签，其余是特征
            self.features = df.iloc[:, :-1].values.astype(np.float32)
            self.labels = df.iloc[:, -1].values.astype(np.int64)
        
        # 转换为 PyTorch 张量
        self.features = torch.from_numpy(self.features)
        self.labels = torch.from_numpy(self.labels)
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.labels)
    
    def __getitem__(self, idx):
        """
        获取单个样本
        
        参数:
            idx: 样本索引
        
        返回:
            feature: 特征张量
            label: 标签张量
        """
        return self.features[idx], self.labels[idx]


def get_train_dataset(data_dir='datasets', filename='train-set.csv'):
    """
    获取训练集 Dataset
    
    参数:
        data_dir: 数据集目录路径
        filename: 训练集文件名
    
    返回:
        ThyroidDataset: 训练集 Dataset 实例
    """
    return ThyroidDataset(data_dir=data_dir, filename=filename, is_train=True)


def get_test_dataset(data_dir='datasets', filename='test-set.csv'):
    """
    获取测试集 Dataset
    
    参数:
        data_dir: 数据集目录路径
        filename: 测试集文件名
    
    返回:
        ThyroidDataset: 测试集 Dataset 实例
    """
    return ThyroidDataset(data_dir=data_dir, filename=filename, is_train=False)
