import os
import re
from typing import List, Dict


def parse_log_file(log_path: str) -> Dict:
    """
    解析 log 文件，提取模型配置信息和评估指标
    
    Args:
        log_path: log 文件路径
        
    Returns:
        dict: 包含模型配置信息和评估指标的字典
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


def generate_model_name(num_layers: int, use_attention: bool) -> str:
    """
    生成模型名称（英文）
    
    Args:
        num_layers: LSTM 层数
        use_attention: 是否使用注意力机制
        
    Returns:
        str: 模型名称
    """
    layer_str = f"{num_layers}-Layer LSTM" if num_layers == 2 else "1-Layer LSTM"
    attention_str = " + Attention" if use_attention else ""
    return f"{layer_str}{attention_str}"


def generate_latex_table(log_files: List[str], output_path: str = "table.tex"):
    """
    生成LaTeX格式的对比表格
    
    Args:
        log_files: log 文件路径列表
        output_path: 输出文件路径
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
        
        # 生成模型名称
        model_name = generate_model_name(config['num_layers'], config['use_attention'])
        
        all_data.append({
            'model_name': model_name,
            'num_layers': config['num_layers'],
            'use_attention': config['use_attention'],
            'mae': config['mae'],
            'rmse': config['rmse']
        })
    
    if len(all_data) == 0:
        print("错误: 没有成功加载任何数据")
        return
    
    # 保持输入顺序，不进行排序
    
    # 找出MAE和RMSE的最小值和第二小值（用于加粗和下划线）
    mae_values = [d['mae'] for d in all_data if d['mae'] is not None]
    rmse_values = [d['rmse'] for d in all_data if d['rmse'] is not None]
    
    mae_sorted = sorted(set(mae_values)) if mae_values else []
    rmse_sorted = sorted(set(rmse_values)) if rmse_values else []
    
    mae_min = mae_sorted[0] if len(mae_sorted) > 0 else None
    mae_second = mae_sorted[1] if len(mae_sorted) > 1 else None
    rmse_min = rmse_sorted[0] if len(rmse_sorted) > 0 else None
    rmse_second = rmse_sorted[1] if len(rmse_sorted) > 1 else None
    
    # 生成LaTeX表格
    latex_content = []
    latex_content.append("\\documentclass{article}")
    latex_content.append("\\usepackage{booktabs}")
    latex_content.append("")
    latex_content.append("\\begin{document}")
    latex_content.append("")
    latex_content.append("\\begin{table}[htbp]")
    latex_content.append("\\centering")
    latex_content.append("\\caption{Comparison of different LSTM models on weather prediction task}")
    latex_content.append("\\label{tab:model_comparison}")
    latex_content.append("\\begin{tabular}{lcc}")
    latex_content.append("\\toprule")
    latex_content.append("Model & MAE & RMSE \\\\")
    latex_content.append("\\midrule")
    
    for data in all_data:
        model_name = data['model_name']
        mae = data['mae']
        rmse = data['rmse']
        
        # 格式化数值并添加格式
        if mae is not None:
            mae_str = f"{mae:.4f}"
            # 如果是最小值，加粗
            if mae == mae_min:
                mae_str = f"\\textbf{{{mae_str}}}"
            # 如果是第二小值，加下划线
            elif mae == mae_second:
                mae_str = f"\\underline{{{mae_str}}}"
        else:
            mae_str = "N/A"
        
        if rmse is not None:
            rmse_str = f"{rmse:.4f}"
            # 如果是最小值，加粗
            if rmse == rmse_min:
                rmse_str = f"\\textbf{{{rmse_str}}}"
            # 如果是第二小值，加下划线
            elif rmse == rmse_second:
                rmse_str = f"\\underline{{{rmse_str}}}"
        else:
            rmse_str = "N/A"
        
        # LaTeX转义特殊字符
        model_name_escaped = model_name.replace('&', '\\&')
        
        latex_content.append(f"{model_name_escaped} & {mae_str} & {rmse_str} \\\\")
    
    latex_content.append("\\bottomrule")
    latex_content.append("\\end{tabular}")
    latex_content.append("\\end{table}")
    latex_content.append("\\end{document}")
    
    # 写入文件（使用 'w' 模式覆盖，避免编辑器自动添加的内容）
    with open(output_path, 'w', encoding='utf-8') as f:
        # 先写入 BOM 标记（如果需要），然后写入内容
        f.write('\n'.join(latex_content))
        f.write('\n')  # 文件末尾换行
    
    print(f"LaTeX表格已保存到: {output_path}")
    
    # 同时输出到控制台
    print("\n生成的表格内容：")
    print('\n'.join(latex_content))


def generate_markdown_table(log_files: List[str], output_path: str = "table.md"):
    """
    生成Markdown格式的对比表格
    
    Args:
        log_files: log 文件路径列表
        output_path: 输出文件路径
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
        
        # 生成模型名称
        model_name = generate_model_name(config['num_layers'], config['use_attention'])
        
        all_data.append({
            'model_name': model_name,
            'num_layers': config['num_layers'],
            'use_attention': config['use_attention'],
            'mae': config['mae'],
            'rmse': config['rmse']
        })
    
    if len(all_data) == 0:
        print("错误: 没有成功加载任何数据")
        return
    
    # 保持输入顺序，不进行排序
    
    # 生成Markdown表格
    md_content = []
    md_content.append("## Model Comparison Table")
    md_content.append("")
    md_content.append("| Model | MAE | RMSE |")
    md_content.append("|-------|-----|------|")
    
    for data in all_data:
        model_name = data['model_name']
        mae = data['mae']
        rmse = data['rmse']
        
        # 格式化数值
        if mae is not None:
            mae_str = f"{mae:.4f}"
        else:
            mae_str = "N/A"
        
        if rmse is not None:
            rmse_str = f"{rmse:.4f}"
        else:
            rmse_str = "N/A"
        
        md_content.append(f"| {model_name} | {mae_str} | {rmse_str} |")
    
    # 写入文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_content))
    
    print(f"Markdown表格已保存到: {output_path}")


if __name__ == "__main__":
    # =========================================================
    #  使用说明：
    #  在这里添加要对比的 log 文件路径列表
    #  程序会自动解析并生成科研论文格式的表格
    # =========================================================
    
    # 示例：对比多个模型的 log 文件
    log_files = [
        "ckpt/20251218_192941/log_20251218_192941.txt",  # 单层LSTM，无注意力
        "ckpt/20251218_193045/log_20251218_193045.txt",  # 单层LSTM，有注意力
        "ckpt/20251218_192804/log_20251218_192804.txt",  # 双层LSTM，无注意力
        "ckpt/20251218_193228/log_20251218_193228.txt",  # 双层LSTM，有注意力
    ]
    
    # 生成LaTeX表格（科研论文常用格式）
    generate_latex_table(log_files, output_path="table.tex")
    
    # 生成Markdown表格（可选，用于文档或README）
    generate_markdown_table(log_files, output_path="table.md")
    
    print("\n完成！")
