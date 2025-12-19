import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from pathlib import Path

# 设置中文字体和科研论文风格
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-paper')  # 科研论文风格


def parse_log_file(log_path):
    """
    解析 log 文件，提取模型配置信息
    
    Args:
        log_path: log 文件路径
        
    Returns:
        dict: 包含模型配置信息的字典
    """
    config = {
        'num_layers': None,
        'use_attention': None,
        'mae': None,
        'rmse': None,
        'timestamp': None
    }
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 提取时间戳（从文件名或路径中）
        timestamp_match = re.search(r'(\d{8}_\d{6})', log_path)
        if timestamp_match:
            config['timestamp'] = timestamp_match.group(1)
        
        # 解析每一行
        for i, line in enumerate(lines):
            # 提取 LSTM 层数
            if 'LSTM层数:' in line:
                match = re.search(r'LSTM层数:\s*(\d+)', line)
                if match:
                    config['num_layers'] = int(match.group(1))
            
            # 提取注意力机制
            if '使用注意力机制:' in line:
                match = re.search(r'使用注意力机制:\s*(True|False)', line)
                if match:
                    config['use_attention'] = match.group(1) == 'True'
            
            # 提取 MAE 和 RMSE（原始尺度）
            if '测试集结果（OT变化量预测，原始尺度）' in line:
                # 读取接下来的两行
                if i + 1 < len(lines):
                    mae_line = lines[i + 1]
                    mae_match = re.search(r'MAE\s*=\s*([\d.]+)', mae_line)
                    if mae_match:
                        config['mae'] = float(mae_match.group(1))
                
                if i + 2 < len(lines):
                    rmse_line = lines[i + 2]
                    rmse_match = re.search(r'RMSE\s*=\s*([\d.]+)', rmse_line)
                    if rmse_match:
                        config['rmse'] = float(rmse_match.group(1))
    
    except Exception as e:
        print(f"解析 log 文件 {log_path} 时出错: {e}")
    
    return config


def load_predictions(csv_path):
    """
    加载预测结果 CSV 文件
    
    Args:
        csv_path: CSV 文件路径
        
    Returns:
        pd.DataFrame: 预测结果数据框
    """
    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        print(f"加载 CSV 文件 {csv_path} 时出错: {e}")
        return None


def find_csv_from_log(log_path):
    """
    根据 log 文件路径找到对应的 predictions CSV 文件
    
    Args:
        log_path: log 文件路径
        
    Returns:
        str: CSV 文件路径，如果找不到则返回 None
    """
    # 从 log 路径中提取时间戳
    timestamp_match = re.search(r'(\d{8}_\d{6})', log_path)
    if not timestamp_match:
        return None
    
    timestamp = timestamp_match.group(1)
    
    # 获取 log 文件所在目录
    log_dir = os.path.dirname(os.path.abspath(log_path))
    
    # 构建 CSV 文件路径（优先在同目录下查找）
    csv_path = os.path.join(log_dir, f"predictions_{timestamp}.csv")
    
    if os.path.exists(csv_path):
        return csv_path
    
    # 如果直接路径不存在，尝试在 ckpt 目录下查找
    csv_path = os.path.join("ckpt", timestamp, f"predictions_{timestamp}.csv")
    if os.path.exists(csv_path):
        return csv_path
    
    # 尝试相对路径
    csv_path = os.path.join(os.path.dirname(log_path), f"predictions_{timestamp}.csv")
    if os.path.exists(csv_path):
        return csv_path
    
    return None


def generate_model_label(num_layers, use_attention):
    """
    生成模型标签（英文）
    
    Args:
        num_layers: LSTM 层数
        use_attention: 是否使用注意力机制
        
    Returns:
        str: 模型标签
    """
    layer_str = f"{num_layers}-Layer LSTM" if num_layers == 2 else "1-Layer LSTM"
    attention_str = " + Attention" if use_attention else ""
    return f"{layer_str}{attention_str}"


def plot_comparison(log_files, save_path=None, show_plot=True, max_points=500):
    """
    绘制多个模型的对比图
    
    Args:
        log_files: log 文件路径列表
        save_path: 保存图片的路径（可选）
        show_plot: 是否显示图片
        max_points: 最多显示的数据点数量
    """
    # 存储所有模型的数据
    all_data = []
    
    # 解析每个 log 文件
    for log_path in log_files:
        # 转换为绝对路径并检查文件是否存在
        log_path_abs = os.path.abspath(log_path)
        if not os.path.exists(log_path_abs):
            print(f"警告: 文件 {log_path} 不存在，跳过")
            continue
        
        log_path = log_path_abs
        
        # 解析 log 文件
        config = parse_log_file(log_path)
        
        # 检查是否成功解析
        if config['num_layers'] is None:
            print(f"警告: 无法从 {log_path} 解析模型配置，跳过")
            continue
        
        # 找到对应的 CSV 文件
        csv_path = find_csv_from_log(log_path)
        if csv_path is None or not os.path.exists(csv_path):
            print(f"警告: 找不到 {log_path} 对应的 predictions CSV 文件，跳过")
            continue
        
        # 加载预测结果
        df = load_predictions(csv_path)
        if df is None:
            continue
        
        # 生成模型标签（不包含误差指标）
        model_label = generate_model_label(config['num_layers'], config['use_attention'])
        
        all_data.append({
            'label': model_label,
            'config': config,
            'df': df
        })
    
    if len(all_data) == 0:
        print("错误: 没有成功加载任何数据")
        return
    
    # 定义颜色和线型（使用自定义颜色列表）
    custom_colors = [
        '#9bbf8a', '#82afda', '#c2bdde', '#f79059', '#e7dbd3',
        '#8dcec8', '#add3e2', '#3480b8', '#ffbe7a', '#fa8878',
        '#c82423'
    ]
    # 如果模型数量超过颜色列表长度，循环使用颜色
    colors = [custom_colors[i % len(custom_colors)] for i in range(len(all_data))]
    linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 1)), (0, (1, 1))][:len(all_data)]
    
    # ========== 图1: OT变化量预测对比 ==========
    fig1, ax1 = plt.subplots(1, 1, figsize=(12, 10))
    # fig1.suptitle('Model Comparison: OT Change Prediction', fontsize=16, fontweight='bold')
    
    for i, data in enumerate(all_data):
        df = data['df']
        n_points = min(len(df), max_points)
        
        # 绘制真实值（只绘制一次）
        if i == 0:
            ax1.plot(df['True_Delta_OT'].values[:n_points], 
                    label='True Delta OT', 
                    color='gray', 
                    linewidth=2.5, 
                    linestyle='-',
                    alpha=0.8)
        
        # 绘制预测值
        ax1.plot(df['Predicted_Delta_OT'].values[:n_points], 
                label=data['label'], 
                color=colors[i], 
                linewidth=1.5, 
                linestyle='-',
                alpha=0.8)
    
    ax1.set_xlabel('Time Index', fontsize=12)
    ax1.set_ylabel('Delta Temperature', fontsize=12)
    # ax1.set_title('OT Change Prediction', fontsize=13, fontweight='bold')
    legend1 = ax1.legend(loc='best', fontsize=9, frameon=True, fancybox=False, edgecolor='gray', facecolor='white')
    legend1.get_frame().set_alpha(1.0)
    ax1.grid(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # 保存第一张图片
    if save_path:
        # 生成第一张图的文件名
        if save_path.endswith('.png'):
            save_path1 = save_path.replace('.png', '_delta.png')
        else:
            save_path1 = save_path + '_delta.png'
        plt.savefig(save_path1, dpi=300, bbox_inches='tight')
        print(f"OT变化量对比图已保存到: {save_path1}")
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig1)
    
    # ========== 图2: OT值预测对比 ==========
    fig2, ax2 = plt.subplots(1, 1, figsize=(12, 10))
    # fig2.suptitle('Model Comparison: OT Value Prediction via Delta', fontsize=16, fontweight='bold')
    
    for i, data in enumerate(all_data):
        df = data['df']
        n_points = min(len(df), max_points)
        
        # 绘制真实值（只绘制一次）
        if i == 0:
            ax2.plot(df['True_OT_Next'].values[:n_points], 
                    label='True OT', 
                    color='gray', 
                    linewidth=2.5, 
                    linestyle='-',
                    alpha=0.8)
        
        # 绘制预测值
        ax2.plot(df['Predicted_OT_Next'].values[:n_points], 
                label=data['label'], 
                color=colors[i], 
                linewidth=1.5, 
                linestyle='-',
                alpha=0.8)
    
    ax2.set_xlabel('Time Index', fontsize=12)
    ax2.set_ylabel('Temperature', fontsize=12)
    # ax2.set_title('OT Value Prediction via Delta', fontsize=13, fontweight='bold')
    legend2 = ax2.legend(loc='best', fontsize=9, frameon=True, fancybox=False, edgecolor='gray', facecolor='white')
    legend2.get_frame().set_alpha(1.0)
    ax2.grid(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # 保存第二张图片
    if save_path:
        # 生成第二张图的文件名
        if save_path.endswith('.png'):
            save_path2 = save_path.replace('.png', '_value.png')
        else:
            save_path2 = save_path + '_value.png'
        plt.savefig(save_path2, dpi=300, bbox_inches='tight')
        print(f"OT值对比图已保存到: {save_path2}")
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig2)


if __name__ == "__main__":
    # =========================================================
    #  使用说明：
    #  在这里添加要对比的 log 文件路径列表
    #  程序会自动找到对应的 predictions CSV 文件并生成对比图
    # =========================================================
    
    # 示例：对比多个模型的 log 文件
    # 在这里添加要对比的 log 文件路径列表
    log_files = [
        "ckpt/20251218_192941/log_20251218_192941.txt",  # 单层LSTM，无注意力
        "ckpt/20251218_193045/log_20251218_193045.txt",  # 单层LSTM，有注意力
        "ckpt/20251218_192804/log_20251218_192804.txt",  # 双层LSTM，无注意力
        "ckpt/20251218_193228/log_20251218_193228.txt",  # 双层LSTM，有注意力
    ]
    
    # 生成对比图
    plot_comparison(
        log_files=log_files,
        save_path="model_comparison.png",  # 保存路径
        show_plot=True,  # 是否显示图片
        max_points=500   # 最多显示的数据点数量
    )
