import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import glob

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def plot_grid_search_results(csv_file=None):
    """
    根据网格搜索结果CSV文件生成三个柱状图
    
    Args:
        csv_file: CSV文件路径，如果为None则自动查找最新的文件
    """
    # 如果没有指定文件，查找最新的网格搜索结果文件
    if csv_file is None:
        ckpt_dir = Path("ckpt")
        csv_files = list(ckpt_dir.glob("prototype_grid_search_summary_*.csv"))
        if not csv_files:
            print("未找到网格搜索结果文件！")
            return
        csv_file = max(csv_files, key=lambda p: p.stat().st_mtime)
        print(f"使用最新的文件: {csv_file}")
    
    # 读取CSV文件，跳过空格
    df = pd.read_csv(csv_file, skipinitialspace=True)
    
    # 清理列名（去除空格）
    df.columns = df.columns.str.strip()
    
    # 清理数据（去除空格）
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.strip()
            # 尝试转换为数值类型
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 确保数据类型正确
    df['ema_alpha'] = pd.to_numeric(df['ema_alpha'], errors='coerce')
    df['auc'] = pd.to_numeric(df['auc'], errors='coerce')
    df['best_f1_score'] = pd.to_numeric(df['best_f1_score'], errors='coerce')
    df['accuracy'] = pd.to_numeric(df['accuracy'], errors='coerce')
    
    # 删除包含NaN的行
    df = df.dropna()
    
    # 按EMA Alpha排序
    df = df.sort_values('ema_alpha', ascending=False)
    
    # 准备数据
    ema_alphas = df['ema_alpha'].values
    auc_values = df['auc'].values
    f1_values = df['best_f1_score'].values
    acc_values = df['accuracy'].values
    
    # 设置x轴标签
    x_labels = [f'{alpha:.3f}' if alpha < 1.0 else f'{alpha:.1f}' for alpha in ema_alphas]
    x_pos = np.arange(len(ema_alphas))
    
    # 为每个EMA值分配不同颜色（使用colormap）
    n_colors = len(ema_alphas)
    cmap = plt.cm.get_cmap('tab20', n_colors) if n_colors <= 20 else plt.cm.get_cmap('viridis', n_colors)
    colors = [cmap(i) for i in range(n_colors)]
    
    # 输出目录
    output_dir = Path("ckpt")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = Path(csv_file).stem.split('_')[-1]
    
    # 1. AUC 柱状图（独立图表）
    fig1, ax1 = plt.subplots(figsize=(14, 6))
    bars1 = ax1.bar(x_pos, auc_values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('EMA Alpha', fontsize=12)
    ax1.set_ylabel('AUC', fontsize=12)
    ax1.set_title('AUC vs EMA Alpha', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    
    # 标记最佳值（加粗x轴标签字体）
    best_auc_idx = np.argmax(auc_values)
    ax1.set_xticklabels(x_labels, rotation=45, ha='right')
    # 获取所有标签并设置最佳值的字体加粗
    for i, label in enumerate(ax1.get_xticklabels()):
        if i == best_auc_idx:
            label.set_fontweight('bold')
            label.set_fontsize(11)  # 稍微增大字体使其更明显
    
    ax1.grid(False)  # 去掉网格背景
    ax1.set_ylim([0, 1.0])
    
    # 在柱状图上添加数值标签（只显示最佳值和前3个值）
    sorted_indices = np.argsort(auc_values)[::-1]
    for idx in sorted_indices[:3]:
        bar = bars1[idx]
        value = auc_values[idx]
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    output_file1 = output_dir / f"grid_search_auc_{timestamp}.png"
    plt.savefig(output_file1, dpi=300, bbox_inches='tight')
    print(f"AUC图表已保存到: {output_file1}")
    plt.show(block=False)
    
    # 2. F1 Score 柱状图（独立图表）
    fig2, ax2 = plt.subplots(figsize=(14, 6))
    bars2 = ax2.bar(x_pos, f1_values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('EMA Alpha', fontsize=12)
    ax2.set_ylabel('F1 Score', fontsize=12)
    ax2.set_title('F1 Score vs EMA Alpha', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos)
    
    # 标记最佳值（加粗x轴标签字体）
    best_f1_idx = np.argmax(f1_values)
    ax2.set_xticklabels(x_labels, rotation=45, ha='right')
    # 获取所有标签并设置最佳值的字体加粗
    for i, label in enumerate(ax2.get_xticklabels()):
        if i == best_f1_idx:
            label.set_fontweight('bold')
            label.set_fontsize(11)  # 稍微增大字体使其更明显
    
    ax2.grid(False)  # 去掉网格背景
    ax2.set_ylim([0, 1.0])
    
    # 在柱状图上添加数值标签（只显示最佳值和前3个值）
    sorted_indices = np.argsort(f1_values)[::-1]
    for idx in sorted_indices[:3]:
        bar = bars2[idx]
        value = f1_values[idx]
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    output_file2 = output_dir / f"grid_search_f1_{timestamp}.png"
    plt.savefig(output_file2, dpi=300, bbox_inches='tight')
    print(f"F1 Score图表已保存到: {output_file2}")
    plt.show(block=False)
    
    # 3. Accuracy 柱状图（独立图表）
    fig3, ax3 = plt.subplots(figsize=(14, 6))
    bars3 = ax3.bar(x_pos, acc_values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax3.set_xlabel('EMA Alpha', fontsize=12)
    ax3.set_ylabel('Accuracy', fontsize=12)
    ax3.set_title('Accuracy vs EMA Alpha', fontsize=14, fontweight='bold')
    ax3.set_xticks(x_pos)
    
    # 标记最佳值（加粗x轴标签字体）
    best_acc_idx = np.argmax(acc_values)
    ax3.set_xticklabels(x_labels, rotation=45, ha='right')
    # 获取所有标签并设置最佳值的字体加粗
    for i, label in enumerate(ax3.get_xticklabels()):
        if i == best_acc_idx:
            label.set_fontweight('bold')
            label.set_fontsize(11)  # 稍微增大字体使其更明显
    
    ax3.grid(False)  # 去掉网格背景
    ax3.set_ylim([0, 1.0])
    
    # 在柱状图上添加数值标签（只显示最佳值和前3个值）
    sorted_indices = np.argsort(acc_values)[::-1]
    for idx in sorted_indices[:3]:
        bar = bars3[idx]
        value = acc_values[idx]
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    output_file3 = output_dir / f"grid_search_accuracy_{timestamp}.png"
    plt.savefig(output_file3, dpi=300, bbox_inches='tight')
    print(f"Accuracy图表已保存到: {output_file3}")
    plt.show()
    
    # 打印统计信息
    print("\n" + "="*60)
    print("统计信息:")
    print("="*60)
    print(f"最佳 AUC: {auc_values[best_auc_idx]:.4f} (EMA Alpha = {ema_alphas[best_auc_idx]:.3f})")
    print(f"最佳 F1 Score: {f1_values[best_f1_idx]:.4f} (EMA Alpha = {ema_alphas[best_f1_idx]:.3f})")
    print(f"最佳 Accuracy: {acc_values[best_acc_idx]:.4f} (EMA Alpha = {ema_alphas[best_acc_idx]:.3f})")
    print("="*60)


if __name__ == "__main__":
    import sys
    
    # 如果提供了命令行参数，使用指定的文件
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        csv_file = None
    
    plot_grid_search_results(csv_file)

