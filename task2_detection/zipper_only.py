import os
import csv
import logging
import random
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from model.ResNet18 import ResNet18
from model.CNN import CNN, CNN_EncoderOnly
from model.MLP import MLP
from data_loader.data_loader_zipper import ClassDataset

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class Trainer:
    """训练器类，用于训练 good/bad 分类模型"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        device: str = None,
        learning_rate: float = 0.001,
    ):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        
        self.ckpt_dir = Path("ckpt")
        self.ckpt_dir.mkdir(exist_ok=True)
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 配置日志
        self._setup_logging()
        
        # 计算类别权重
        from collections import Counter
        label_counts = Counter()
        for _, labels, _ in self.train_loader:
            label_counts.update(labels.cpu().numpy().tolist())
        
        total_samples = sum(label_counts.values())
        num_classes = len(label_counts)
        class_weights = torch.ones(num_classes)

        for label, count in label_counts.items():
            class_weights[label] = total_samples / (num_classes * count)
        
        self.logger.info(f"类别分布: {dict(label_counts)}")
        self.logger.info(f"类别权重: {class_weights}")
        
        # 损失与优化器
        self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
    
    def _setup_logging(self):
        self.logger = logging.getLogger('zipper_trainer')
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = []
        
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        log_file = self.ckpt_dir / f"{self.timestamp}_zipper_logging.log"
        csv_file_name = f"{self.timestamp}_zipper_pred.csv"
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        self.csv_file_name = csv_file_name
    
    def train(self, num_epochs: int = 10):
        self.logger.info("=" * 60)
        self.logger.info("开始训练 zipper good/bad 分类模型")
        self.logger.info(f"设备: {self.device}")
        self.logger.info(f"训练集大小（原图）: {len(self.train_loader.dataset)}")
        self.logger.info(f"测试集大小: {len(self.test_loader.dataset)}")
        self.logger.info(f"训练轮数: {num_epochs}")
        self.logger.info("=" * 60)
        
        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            num_batches = 0
            
            for batch_idx, (images, labels, domains) in tqdm(
                enumerate(self.train_loader),
                total=len(self.train_loader),
                desc=f"Epoch [{epoch+1}/{num_epochs}]"
            ):
                images = images.to(self.device)
                targets = labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                num_batches += 1
            
            avg_loss = running_loss / num_batches
            
            self.logger.info(
                f"[Epoch {epoch+1}] Loss: {avg_loss:.4f}"
            )
        
        self.logger.info("=" * 60)
        self.logger.info("训练完成！")
        
        # 训练完成后，生成特征层的t-SNE可视化
        self.visualize_tsne()
        
        # 然后进行评估
        self.evaluate()
    
    def visualize_tsne(self, max_samples=1000):
        """
        对模型encoder提取的特征进行t-SNE降维可视化
        
        Args:
            max_samples: 最大采样数量（如果数据太多，可以采样）
        """
        self.model.eval()
        
        all_features = []
        all_labels = []
        
        self.logger.info("=" * 60)
        self.logger.info("生成特征层 t-SNE 可视化图")
        self.logger.info("=" * 60)
        self.logger.info("正在提取特征用于t-SNE可视化...")
        
        with torch.no_grad():
            sample_count = 0
            for images, labels, _ in self.test_loader:
                if sample_count >= max_samples:
                    break
                    
                images = images.to(self.device)
                
                # 提取特征：支持不同的模型接口
                if hasattr(self.model, 'get_features'):
                    # ResNet18_GoodBad, CNN, MLP 等有 get_features 方法
                    feat = self.model.get_features(images)
                elif hasattr(self.model, 'encoder'):
                    # 有 encoder 属性的模型
                    feat = self.model.encoder(images)
                    # 如果encoder输出是特征图，需要展平
                    if len(feat.shape) > 2:
                        feat = torch.flatten(feat, 1)
                else:
                    # 对于没有明确encoder的模型，使用forward但只取中间特征
                    # 这里作为fallback，可能需要根据具体模型调整
                    self.logger.warning("模型没有get_features或encoder方法，跳过t-SNE可视化")
                    return
                
                all_features.append(feat.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                
                sample_count += labels.size(0)
        
        # 合并所有特征和标签
        features = np.vstack(all_features)
        labels = np.hstack(all_labels)
        
        # 如果样本太多，随机采样
        if len(features) > max_samples:
            indices = np.random.choice(len(features), max_samples, replace=False)
            features = features[indices]
            labels = labels[indices]
        
        self.logger.info(f"提取了 {len(features)} 个样本的特征，特征维度: {features.shape[1]}")
        self.logger.info("开始t-SNE降维...")
        
        # t-SNE降维
        perplexity = min(30, len(features) - 1)  # perplexity不能大于样本数-1
        tsne = TSNE(
            n_components=2, 
            random_state=42, 
            perplexity=perplexity, 
            verbose=0
        )
        features_2d = tsne.fit_transform(features)
        
        # 绘制散点图
        plt.figure(figsize=(10, 8))
        
        # 根据标签着色（good=0, bad=1）
        colors = ['blue', 'red']
        labels_str = ['good', 'bad']
        
        for i in range(2):
            mask = labels == i
            plt.scatter(
                features_2d[mask, 0], 
                features_2d[mask, 1], 
                c=colors[i], 
                label=labels_str[i],
                alpha=0.6,
                s=20
            )
        
        plt.title('t-SNE Visualization of Encoder Features', fontsize=14, fontweight='bold')
        plt.xlabel('t-SNE Dimension 1', fontsize=12)
        plt.ylabel('t-SNE Dimension 2', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 保存图片
        tsne_path = self.ckpt_dir / f"{self.timestamp}_tsne_visualization.png"
        plt.savefig(tsne_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"t-SNE可视化图已保存到: {tsne_path}")
        self.logger.info("=" * 60)
    
    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0
        
        all_true = []
        all_pred = []
        
        with torch.no_grad():
            for images, labels, domains in self.test_loader:
                images = images.to(self.device)
                targets = labels.to(self.device)
                
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                
                all_true.extend(targets.cpu().numpy().tolist())
                all_pred.extend(predicted.cpu().numpy().tolist())
                
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        accuracy = 100 * correct / total
        
        csv_file = self.ckpt_dir / self.csv_file_name
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['true', 'pred'])
            for t, p in zip(all_true, all_pred):
                writer.writerow([t, p])
        
        self.logger.info("=" * 60)
        self.logger.info("测试集评估结果")
        self.logger.info(f"准确率: {accuracy:.2f}% ({correct}/{total})")
        self.logger.info("分类: good=0, bad=1")
        self.logger.info(f"预测结果已保存到: {csv_file}")
        self.logger.info("=" * 60)


def set_seed(seed=42):
    """固定随机种子以确保结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def main():
    # 固定随机种子
    set_seed(42)
    
    print("=" * 60)
    print("开始训练 zipper good/bad 分类模型")
    print("=" * 60)
    
    # TwoClassDataset 已经只加载 hazelnut，不需要 DomainFilteredDataset
    train_dataset = ClassDataset("dataset", split="train", img_size=64)
    test_dataset  = ClassDataset("dataset", split="test", img_size=64)
    
    # 直接加载
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                              worker_init_fn=lambda worker_id: np.random.seed(42 + worker_id))
    test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False,
                              worker_init_fn=lambda worker_id: np.random.seed(42 + worker_id))
    
    print(f"\n训练集(原图): {len(train_dataset)} 个样本")
    print(f"测试集: {len(test_dataset)} 个样本")
    
    model = ResNet18(num_classes=2, pretrained=False, freeze_backbone=False)
    # model = CNN(num_classes=2, dropout=0.1)
    # model = MLP(num_classes=2, dropout=0.1)
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        learning_rate=0.001
    )
    
    trainer.train(num_epochs=60)
    
    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)
    print(f"\n所有日志和预测结果已保存到 ckpt/ 目录")



if __name__ == "__main__":
    main()
