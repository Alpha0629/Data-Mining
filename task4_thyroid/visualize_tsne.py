import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from data_loader import get_test_dataset
from torch.utils.data import DataLoader
import torch
from pathlib import Path
from datetime import datetime

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def plot_tsne(test_dataset, output_dir="ckpt", perplexity=30, n_iter=1000, random_state=42):
    """
    对测试集数据进行t-SNE降维可视化
    
    Args:
        test_dataset: 测试集Dataset
        output_dir: 输出目录
        perplexity: t-SNE的困惑度参数（建议值：5-50，默认30）
        n_iter: 迭代次数（默认1000）
        random_state: 随机种子
    """
    print("开始加载测试集数据...")
    
    # 获取所有测试数据
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    features, labels = next(iter(test_loader))
    
    # 转换为numpy数组
    X = features.numpy()
    y = labels.numpy()
    
    print(f"数据形状: {X.shape}")
    print(f"标签分布: 正常={np.sum(y==0)}, 异常={np.sum(y==1)}")
    
    # 执行t-SNE降维
    print(f"\n开始t-SNE降维 (perplexity={perplexity}, n_iter={n_iter})...")
    print("这可能需要一些时间，请耐心等待...")
    
    tsne = TSNE(n_components=2, perplexity=perplexity,
                random_state=random_state, verbose=1)
    X_tsne = tsne.fit_transform(X)
    
    print("t-SNE降维完成！")
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('white')  # 设置背景为白色
    ax.set_facecolor('#fafafa')  # 设置图表背景为浅灰色
    
    fig.suptitle('测试集数据 t-SNE 降维可视化', fontsize=16, fontweight='bold', y=0.98)
    
    # 按真实标签着色 - 使用柔和的颜色
    normal_mask = y == 0
    anomaly_mask = y == 1
    
    # 使用柔和的颜色：青绿色和浅橙色
    normal_color = '#f79059'  # 柔和的青绿色
    anomaly_color = '#9573A6'  # 柔和的沙棕色/橙色
    
    scatter_normal = ax.scatter(X_tsne[normal_mask, 0], X_tsne[normal_mask, 1],
                               c=normal_color, alpha=0.7, s=40, 
                               label='正常样本', edgecolors='white', linewidths=0.8)
    scatter_anomaly = ax.scatter(X_tsne[anomaly_mask, 0], X_tsne[anomaly_mask, 1],
                                c=anomaly_color, alpha=0.7, s=40, 
                                label='异常样本', edgecolors='white', linewidths=0.8)
    
    ax.set_xlabel('t-SNE 维度 1', fontsize=12, fontweight='medium')
    ax.set_ylabel('t-SNE 维度 2', fontsize=12, fontweight='medium')
    ax.set_title('按真实标签分类', fontsize=14, fontweight='bold', pad=15)
    
    # 美化图例
    legend = ax.legend(loc='best', fontsize=11, frameon=True, 
                      fancybox=True, shadow=True, framealpha=0.9)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('#cccccc')
    
    ax.grid(False)  # 去掉网格背景
    
    # 美化坐标轴
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')
    
    plt.tight_layout()
    
    # 保存图片
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path(output_dir) / f"tsne_visualization_{timestamp}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n图表已保存到: {output_file}")
    
    # 保存t-SNE坐标数据
    tsne_data = pd.DataFrame({
        'tsne_dim1': X_tsne[:, 0],
        'tsne_dim2': X_tsne[:, 1],
        'true_label': y
    })
    tsne_csv = Path(output_dir) / f"tsne_coordinates_{timestamp}.csv"
    tsne_data.to_csv(tsne_csv, index=False, encoding='utf-8-sig')
    print(f"t-SNE坐标数据已保存到: {tsne_csv}")
    
    # 显示图表
    plt.show()
    
    # 打印统计信息
    print("\n" + "="*60)
    print("统计信息:")
    print("="*60)
    print(f"正常样本数量: {np.sum(normal_mask)}")
    print(f"异常样本数量: {np.sum(anomaly_mask)}")
    print(f"正常样本在t-SNE空间中的中心: ({X_tsne[normal_mask, 0].mean():.4f}, {X_tsne[normal_mask, 1].mean():.4f})")
    print(f"异常样本在t-SNE空间中的中心: ({X_tsne[anomaly_mask, 0].mean():.4f}, {X_tsne[anomaly_mask, 1].mean():.4f})")
    print("="*60)


def plot_tsne_with_model_scores(model, c, test_dataset, device, output_dir="ckpt", 
                                  perplexity=30, n_iter=1000, random_state=42):
    """
    对测试集数据进行t-SNE降维可视化，并使用模型计算的异常分数着色
    
    Args:
        model: 训练好的EmbeddingNet模型
        c: 中心点
        test_dataset: 测试集Dataset
        device: 设备
        output_dir: 输出目录
        perplexity: t-SNE的困惑度参数
        n_iter: 迭代次数
        random_state: 随机种子
    """
    print("开始加载测试集数据...")
    
    # 获取所有测试数据
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    features, labels = next(iter(test_loader))
    
    # 转换为numpy数组
    X = features.numpy()
    y = labels.numpy()
    
    print(f"数据形状: {X.shape}")
    print(f"标签分布: 正常={np.sum(y==0)}, 异常={np.sum(y==1)}")
    
    # 计算模型异常分数
    print("\n开始计算模型异常分数...")
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(device)
        z = model(X_tensor)
        c_tensor = c.to(device) if isinstance(c, torch.Tensor) else torch.FloatTensor(c).to(device)
        anomaly_scores = torch.norm(z - c_tensor, dim=1).cpu().numpy()
    
    print("异常分数计算完成！")
    
    # 执行t-SNE降维
    print(f"\n开始t-SNE降维 (perplexity={perplexity}, n_iter={n_iter})...")
    print("这可能需要一些时间，请耐心等待...")
    
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, 
                random_state=random_state, verbose=1)
    X_tsne = tsne.fit_transform(X)
    
    print("t-SNE降维完成！")
    
    # 创建图表
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('测试集数据 t-SNE 降维可视化（使用模型异常分数）', fontsize=16, fontweight='bold')
    
    # 图1: 按真实标签着色
    ax1 = axes[0]
    normal_mask = y == 0
    anomaly_mask = y == 1
    
    # 使用柔和的颜色
    normal_color = '#9bbf8a'  # 柔和的蓝色
    anomaly_color = '#82afda' # 柔和的沙棕色/橙色
    
    ax1.set_facecolor('#fafafa')  # 设置背景为浅灰色
    
    scatter1_normal = ax1.scatter(X_tsne[normal_mask, 0], X_tsne[normal_mask, 1],
                                  c=normal_color, alpha=0.7, s=40, 
                                  label='正常样本', edgecolors='white', linewidths=0.8)
    scatter1_anomaly = ax1.scatter(X_tsne[anomaly_mask, 0], X_tsne[anomaly_mask, 1],
                                   c=anomaly_color, alpha=0.7, s=40, 
                                   label='异常样本', edgecolors='white', linewidths=0.8)
    
    ax1.set_xlabel('t-SNE 维度 1', fontsize=12, fontweight='medium')
    ax1.set_ylabel('t-SNE 维度 2', fontsize=12, fontweight='medium')
    ax1.set_title('按真实标签分类', fontsize=14, fontweight='bold', pad=15)
    
    # 美化图例
    legend1 = ax1.legend(loc='best', fontsize=10, frameon=True, 
                        fancybox=True, shadow=True, framealpha=0.9)
    legend1.get_frame().set_facecolor('white')
    legend1.get_frame().set_edgecolor('#cccccc')
    
    ax1.grid(False)  # 去掉网格背景
    
    # 美化坐标轴
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_color('#cccccc')
    ax1.spines['bottom'].set_color('#cccccc')
    
    # 图2: 按模型异常分数着色
    ax2 = axes[1]
    ax2.set_facecolor('#fafafa')  # 设置背景为浅灰色
    
    scatter2 = ax2.scatter(X_tsne[:, 0], X_tsne[:, 1], c=anomaly_scores,
                          cmap='plasma', alpha=0.75, s=40, edgecolors='white', linewidths=0.6)
    
    ax2.set_xlabel('t-SNE 维度 1', fontsize=12, fontweight='medium')
    ax2.set_ylabel('t-SNE 维度 2', fontsize=12, fontweight='medium')
    ax2.set_title('按模型异常分数着色', fontsize=14, fontweight='bold', pad=15)
    ax2.grid(False)  # 去掉网格背景
    
    # 美化坐标轴
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_color('#cccccc')
    ax2.spines['bottom'].set_color('#cccccc')
    
    cbar2 = plt.colorbar(scatter2, ax=ax2, pad=0.02)
    cbar2.set_label('异常分数（到中心的距离）', fontsize=10, fontweight='medium')
    cbar2.ax.tick_params(labelsize=9)
    
    # 图3: 按模型异常分数着色，并用点的大小表示分数
    ax3 = axes[2]
    ax3.set_facecolor('#fafafa')  # 设置背景为浅灰色
    
    # 归一化异常分数用于点的大小（范围在25-100之间）
    sizes = 25 + (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min()) * 75
    
    scatter3 = ax3.scatter(X_tsne[:, 0], X_tsne[:, 1], c=anomaly_scores,
                          cmap='plasma', alpha=0.75, s=sizes, edgecolors='white', linewidths=0.5)
    
    ax3.set_xlabel('t-SNE 维度 1', fontsize=12, fontweight='medium')
    ax3.set_ylabel('t-SNE 维度 2', fontsize=12, fontweight='medium')
    ax3.set_title('按异常分数着色和大小', fontsize=14, fontweight='bold', pad=15)
    ax3.grid(False)  # 去掉网格背景
    
    # 美化坐标轴
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['left'].set_color('#cccccc')
    ax3.spines['bottom'].set_color('#cccccc')
    
    cbar3 = plt.colorbar(scatter3, ax=ax3, pad=0.02)
    cbar3.set_label('异常分数（到中心的距离）', fontsize=10, fontweight='medium')
    cbar3.ax.tick_params(labelsize=9)
    
    plt.tight_layout()
    
    # 保存图片
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path(output_dir) / f"tsne_with_model_scores_{timestamp}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n图表已保存到: {output_file}")
    
    # 保存t-SNE坐标数据和异常分数
    tsne_data = pd.DataFrame({
        'tsne_dim1': X_tsne[:, 0],
        'tsne_dim2': X_tsne[:, 1],
        'true_label': y,
        'model_anomaly_score': anomaly_scores
    })
    tsne_csv = Path(output_dir) / f"tsne_with_scores_{timestamp}.csv"
    tsne_data.to_csv(tsne_csv, index=False, encoding='utf-8-sig')
    print(f"t-SNE坐标和异常分数数据已保存到: {tsne_csv}")
    
    # 显示图表
    plt.show()
    
    # 打印统计信息
    print("\n" + "="*60)
    print("统计信息:")
    print("="*60)
    print(f"正常样本数量: {np.sum(normal_mask)}")
    print(f"异常样本数量: {np.sum(anomaly_mask)}")
    print(f"正常样本平均异常分数: {anomaly_scores[normal_mask].mean():.4f}")
    print(f"异常样本平均异常分数: {anomaly_scores[anomaly_mask].mean():.4f}")
    print(f"异常分数范围: [{anomaly_scores.min():.4f}, {anomaly_scores.max():.4f}]")
    print("="*60)


if __name__ == "__main__":
    import sys
    from Prototype import EmbeddingNet
    
    # 加载测试集
    test_dataset = get_test_dataset(data_dir="datasets", filename="test-set.csv")
    
    # 如果提供了模型路径，使用模型异常分数
    if len(sys.argv) > 1 and sys.argv[1] == "--with-model":
        print("使用模型异常分数进行可视化...")
        # 这里需要加载训练好的模型，暂时使用默认参数
        # 实际使用时，应该从保存的模型文件中加载
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = EmbeddingNet(embedding_dim=3).to(device)
        
        # 注意：这里需要实际的模型权重和中心点
        # 为了演示，我们使用随机初始化的模型
        print("警告：使用随机初始化的模型，实际使用时请加载训练好的模型！")
        
        # 创建一个随机的中心点（实际应该从训练中获取）
        c = torch.randn(3).to(device)
        
        plot_tsne_with_model_scores(model, c, test_dataset, device)
    else:
        print("使用原始特征进行可视化...")
        plot_tsne(test_dataset)

