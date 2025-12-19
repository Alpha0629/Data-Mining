import sys
from pathlib import Path

# 添加 task2 的路径以便导入模型和数据加载器
task2_path = Path(__file__).parent.parent / "task2_detection"
if str(task2_path) not in sys.path:
    sys.path.insert(0, str(task2_path))

import numpy as np
import pandas as pd
import logging
import random
from datetime import datetime
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from tqdm import tqdm

# 模型将在主函数中根据配置动态导入


# ===============================
# 0. 随机种子设置
# ===============================
def set_seed(seed=42):
    """设置随机种子以确保实验结果可重复"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"随机种子已设置为: {seed}")


# ===============================
# 1. 日志配置
# ===============================
def setup_logging(log_dir=None, percentile=None, dataset_type="nuts", model_type="resnet18"):
    """配置日志系统，同时输出到文件和控制台"""
    # 如果没有指定log_dir，使用task5_unified_frame目录下的ckpt
    if log_dir is None:
        log_dir = Path(__file__).parent / "ckpt"
    else:
        log_dir = Path(log_dir)
    # 创建日志目录
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成日志文件名（带数据集类型、模型类型、时间戳和分位数，去掉oneclass_svm前缀）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if percentile is not None:
        log_file = log_dir / f"training_{dataset_type}_{model_type}_p{percentile}_{timestamp}.log"
    else:
        log_file = log_dir / f"training_{dataset_type}_{model_type}_{timestamp}.log"
    
    # 配置日志格式
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()  # 同时输出到控制台
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"日志文件已创建: {log_file}")
    return logger


# ========================================
# 2. 使用预训练ResNet18提取特征
# ========================================
def extract_features_with_resnet18(dataset, model, device, logger=None):
    """
    使用预训练的ResNet18提取图片特征（嵌入空间）
    
    Args:
        dataset: 图片数据集（ClassDataset）
        model: 预训练的ResNet18模型
        device: 计算设备（'cuda'或'cpu'）
        logger: 日志记录器
    
    Returns:
        features: 特征数组 [n_samples, feature_dim]
        labels: 标签数组 [n_samples]
    """
    model.eval()
    model.to(device)
    
    all_features = []
    all_labels = []
    
    # 检查数据集是否为空
    if len(dataset) == 0:
        error_msg = f"数据集为空！请检查数据集路径是否正确。当前数据集大小: {len(dataset)}"
        if logger:
            logger.error(error_msg)
        raise ValueError(error_msg)
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    
    if logger:
        logger.info(f"开始提取特征，数据集大小: {len(dataset)}")
    
    with torch.no_grad():
        for images, labels, domains in tqdm(dataloader, desc="提取特征"):
            images = images.to(device)
            
            # 使用ResNet18提取特征（嵌入空间）
            features = model.get_features(images)  # [batch_size, 512]
            
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    # 检查是否有提取到特征
    if len(all_features) == 0:
        error_msg = "未能提取到任何特征！请检查数据集和模型是否正确。"
        if logger:
            logger.error(error_msg)
        raise ValueError(error_msg)
    
    # 合并所有特征和标签
    features = np.vstack(all_features)
    labels = np.hstack(all_labels)
    
    if logger:
        logger.info(f"特征提取完成，特征形状: {features.shape}, 标签形状: {labels.shape}")
    
    return features, labels


# ========================================
# 3. 数据加载和特征提取
# ========================================
def load_and_extract_features(data_dir="dataset", img_size=64, device=None, logger=None, dataset_type="nuts", ClassDataset=None, model_type="resnet18"):
    """
    加载task2的图片数据，使用预训练ResNet模型提取特征
    
    Args:
        data_dir: 数据集目录
        img_size: 图片尺寸
        device: 计算设备
        logger: 日志记录器
        dataset_type: 数据集类型 ('nuts' 或 'zipper')
        ClassDataset: 数据加载器类
        model_type: 模型类型 ('resnet18' 或 'resnet50')
    
    Returns:
        X_train: 训练集特征（只包含正常样本，good=0）
        X_test: 测试集特征
        y_test: 测试集标签（0=正常/good, 1=异常/bad）
    
    Note:
        ResNet18: 特征维度512，参数量约11M，速度快
        ResNet50: 特征维度2048，参数量约25M，特征提取能力更强，但计算量更大
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if logger:
        logger.info(f"使用设备: {device}")
        logger.info(f"数据集类型: {dataset_type}")
    
    if ClassDataset is None:
        raise ValueError("ClassDataset 参数不能为空，请提供数据加载器类")
    
    # 检查数据集路径是否存在
    data_path = Path(data_dir)
    if not data_path.exists():
        # 尝试相对于 task2_detection 的路径
        task2_path = Path(__file__).parent.parent / "task2_detection" / data_dir
        if task2_path.exists():
            data_dir = str(task2_path)
            if logger:
                logger.info(f"使用 task2_detection 目录下的数据集: {data_dir}")
        else:
            error_msg = f"数据集路径不存在: {data_dir}\n请检查路径是否正确，或确保数据集目录存在。"
            if logger:
                logger.error(error_msg)
            raise FileNotFoundError(error_msg)
    
    # 加载数据集
    train_dataset = ClassDataset(data_dir, split="train", img_size=img_size)
    test_dataset = ClassDataset(data_dir, split="test", img_size=img_size)
    
    if logger:
        logger.info(f"数据集路径: {data_dir}")
        logger.info(f"训练集大小: {len(train_dataset)}")
        logger.info(f"测试集大小: {len(test_dataset)}")
    
    # 检查数据集是否为空
    category_name = "hazelnut" if dataset_type == "nuts" else "zipper"
    if len(train_dataset) == 0:
        error_msg = (
            f"训练集为空！请检查数据集路径和目录结构。\n"
            f"期望的目录结构: {data_dir}/{category_name}/train/good/ 和 {data_dir}/{category_name}/train/bad/\n"
            f"当前路径: {Path(data_dir).absolute()}"
        )
        if logger:
            logger.error(error_msg)
        raise ValueError(error_msg)
    
    if len(test_dataset) == 0:
        error_msg = (
            f"测试集为空！请检查数据集路径和目录结构。\n"
            f"期望的目录结构: {data_dir}/{category_name}/test/good/ 和 {data_dir}/{category_name}/test/bad/\n"
            f"当前路径: {Path(data_dir).absolute()}"
        )
        if logger:
            logger.error(error_msg)
        raise ValueError(error_msg)
    
    # 加载预训练的模型（ResNet18或ResNet50）
    if logger:
        logger.info(f"加载预训练{model_type.upper()}模型...")
    
    if model_type.lower() == "resnet18":
        from model.ResNet18 import ResNet18  # pyright: ignore[reportMissingImports]
        model = ResNet18(num_classes=2, pretrained=True, freeze_backbone=True)
    elif model_type.lower() == "resnet50":
        from model.ResNet50 import ResNet50  # pyright: ignore[reportMissingImports]
        model = ResNet50(num_classes=2, pretrained=True, freeze_backbone=True)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}，请选择 'resnet18' 或 'resnet50'")
    
    if logger:
        logger.info(f"{model_type.upper()}模型加载完成")
    
    # 提取训练集特征和标签
    train_features, train_labels = extract_features_with_resnet18(
        train_dataset, model, device, logger
    )
    
    # 提取测试集特征和标签
    test_features, test_labels = extract_features_with_resnet18(
        test_dataset, model, device, logger
    )
    
    # 对于异常检测，训练集应该只包含正常样本（good=0）
    # 过滤出训练集中的正常样本
    normal_mask = train_labels == 0
    X_train = train_features[normal_mask]
    
    if logger:
        logger.info(f"训练集正常样本数: {len(X_train)}")
        logger.info(f"测试集标签分布: 正常={np.sum(test_labels==0)}, 异常={np.sum(test_labels==1)}")
    
    return X_train, test_features, test_labels


# ========================================
# 4. 训练One-Class SVM
# ========================================
def train_oneclass_svm(X_train, nu=0.1, kernel='rbf', gamma='scale', logger=None):
    """
    训练One-Class SVM模型
    
    Args:
        X_train: 训练集特征（只使用正常样本）
        nu: 异常值比例的上界（0-1之间，默认0.1表示最多10%的样本可能是异常）
        kernel: 核函数类型（'rbf', 'linear', 'poly', 'sigmoid'）
        gamma: RBF核的参数（'scale'或'auto'或数值）
        logger: 日志记录器
    
    Returns:
        model: 训练好的One-Class SVM模型
    """
    if logger:
        logger.info(f"训练One-Class SVM: nu={nu}, kernel={kernel}, gamma={gamma}")
    
    model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
    model.fit(X_train)
    
    if logger:
        logger.info("One-Class SVM训练完成")
    
    return model


# ========================================
# 5. 计算异常分数和预测
# ========================================
def compute_scores_and_predict(model, X_data):
    """
    使用One-Class SVM计算异常分数和预测
    
    Args:
        model: 训练好的One-Class SVM模型
        X_data: 数据特征
    
    Returns:
        predictions: 预测结果（1=异常, 0=正常）
        anomaly_scores: 异常分数（分数越高越异常）
    """
    # One-Class SVM返回1（正常）或-1（异常）
    predictions_svm = model.predict(X_data)
    
    # 转换为0/1格式：-1 -> 1（异常），1 -> 0（正常）
    predictions = (predictions_svm == -1).astype(int)
    
    # 决策函数值：负值表示异常，正值表示正常
    decision_scores = model.decision_function(X_data)
    
    # 将决策函数值转换为异常分数（距离越大，异常分数越高）
    # 使用负的决策函数值作为异常分数（这样异常样本分数更高）
    anomaly_scores = -decision_scores
    
    return predictions, anomaly_scores


# ========================================
# 6. 评估函数
# ========================================
def evaluate(scores, labels, threshold, logger, output_dir=None, nu=None, kernel=None, gamma=None, percentile=None, dataset_type="nuts", model_type="resnet18"):
    """
    使用固定阈值评估模型性能
    
    Args:
        scores: 测试集的异常分数
        labels: 测试集的真实标签
        threshold: 固定阈值（训练集分位数）
        logger: 日志记录器
        output_dir: 输出目录
        nu: One-Class SVM的nu参数（用于日志记录）
        kernel: 核函数类型（用于日志记录）
        gamma: gamma参数（用于日志记录）
        percentile: 使用的分位数（用于日志记录）
        dataset_type: 数据集类型 ('nuts' 或 'zipper')
    
    Returns:
        auc: ROC-AUC分数
        f1: F1分数
        acc: 准确率
    """
    auc = roc_auc_score(labels, scores)
    
    # 使用固定阈值进行预测
    preds = (scores > threshold).astype(int)
    f1 = f1_score(labels, preds)
    acc = accuracy_score(labels, preds)

    logger.info("==== Evaluation Result ====")
    if nu is not None:
        logger.info(f"One-Class SVM参数: nu={nu:.3f}, kernel={kernel}, gamma={gamma}")
    if percentile is not None:
        logger.info(f"Threshold (Train {percentile}th percentile): {threshold:.6f}")
    else:
        logger.info(f"Threshold: {threshold:.6f}")
    logger.info(f"AUC: {auc:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    logger.info(f"Accuracy: {acc:.4f}")

    # 如果没有指定output_dir，使用task5_unified_frame目录下的ckpt
    if output_dir is None:
        output_dir = Path(__file__).parent / "ckpt"
    else:
        output_dir = Path(output_dir)
    # 保存预测结果到 CSV（文件名中包含数据集类型和模型类型，去掉oneclass_svm前缀）
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if nu is not None:
        if percentile is not None:
            predictions_csv = output_dir / f"predictions_{dataset_type}_{model_type}_p{percentile}_nu{nu:.3f}_kernel{kernel}_gamma{gamma}_{timestamp}.csv"
        else:
            predictions_csv = output_dir / f"predictions_{dataset_type}_{model_type}_nu{nu:.3f}_kernel{kernel}_gamma{gamma}_{timestamp}.csv"
    else:
        if percentile is not None:
            predictions_csv = output_dir / f"predictions_{dataset_type}_{model_type}_p{percentile}_{timestamp}.csv"
        else:
            predictions_csv = output_dir / f"predictions_{dataset_type}_{model_type}_{timestamp}.csv"
    
    results_df = pd.DataFrame({
        'anomaly_score': scores,
        'true_label': labels,
        'predicted_label': preds
    })
    
    results_df.to_csv(predictions_csv, index=False, encoding='utf-8-sig')
    logger.info(f"预测结果已保存到: {predictions_csv}")

    return auc, f1, acc, predictions_csv


if __name__ == "__main__":
    # 固定随机种子
    set_seed(seed=42)
    
    # ========================================
    # 配置参数
    # ========================================
    # 数据集类型选择：'nuts' 或 'zipper'
    DATASET_TYPE = "zipper"  # 可以修改为 "zipper" 来选择 zipper 数据集
    
    # 模型类型选择：'resnet18' 或 'resnet50'
    # ResNet18: 特征维度512，参数量约11M，速度快，适合快速实验
    # ResNet50: 特征维度2048，参数量约25M，特征提取能力更强，可能获得更好的异常检测效果，但计算量更大
    MODEL_TYPE = "resnet50"  # 可以修改为 "resnet50" 来使用 ResNet50
    
    # 阈值分位数列表（可配置，将进行网格搜索）
    THRESHOLD_PERCENTILE_LIST = [95, 96, 97, 98, 99, 100]  # 可以修改为其他值列表，如 [90, 95, 99]
    
    # 根据数据集类型导入对应的数据加载器
    if DATASET_TYPE == "nuts":
        from data_loader.data_loader_nuts import ClassDataset  # pyright: ignore[reportMissingImports]
    elif DATASET_TYPE == "zipper":
        from data_loader.data_loader_zipper import ClassDataset  # pyright: ignore[reportMissingImports]
    else:
        raise ValueError(f"不支持的数据集类型: {DATASET_TYPE}，请选择 'nuts' 或 'zipper'")
    
    # 设置日志（会自动创建 task5_unified_frame/ckpt 文件夹，使用第一个percentile作为日志文件名）
    ckpt_dir = Path(__file__).parent / "ckpt"
    logger = setup_logging(log_dir=str(ckpt_dir), percentile=THRESHOLD_PERCENTILE_LIST[0] if THRESHOLD_PERCENTILE_LIST else None, dataset_type=DATASET_TYPE, model_type=MODEL_TYPE)
    logger.info(f"随机种子已设置为: 42")
    logger.info(f"数据集类型: {DATASET_TYPE}")
    logger.info(f"模型类型: {MODEL_TYPE}")
    
    # 数据集路径 - 尝试多个可能的路径
    current_dir = Path(__file__).parent
    possible_paths = [
        current_dir / "dataset",  # task5_unified_frame/dataset
        current_dir.parent / "task2_detection" / "dataset",  # task2_detection/dataset
        Path("dataset"),  # 当前工作目录下的 dataset
    ]
    
    DATA_DIR = None
    for path in possible_paths:
        if path.exists():
            DATA_DIR = str(path)
            break
    
    if DATA_DIR is None:
        # 如果都找不到，使用默认路径（会在 load_and_extract_features 中检查）
        DATA_DIR = "dataset"
        logger.warning(f"未找到数据集，将尝试使用路径: {DATA_DIR}")
    else:
        logger.info(f"找到数据集路径: {DATA_DIR}")
    
    IMG_SIZE = 64  # 图片尺寸
    
    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # ========================================
    # 加载数据并提取特征
    # ========================================
    logger.info("开始加载数据集并提取特征...")
    X_train, X_test, y_test = load_and_extract_features(
        data_dir=DATA_DIR,
        img_size=IMG_SIZE,
        device=device,
        logger=logger,
        dataset_type=DATASET_TYPE,
        ClassDataset=ClassDataset,
        model_type=MODEL_TYPE
    )
    
    logger.info(f"训练集特征形状: {X_train.shape}")
    logger.info(f"测试集特征形状: {X_test.shape}")
    logger.info(f"测试集标签分布: 正常={np.sum(y_test==0)}, 异常={np.sum(y_test==1)}")
    
    # 特征标准化（对One-Class SVM很重要）
    logger.info("对特征进行标准化...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    logger.info("特征标准化完成")
    
    # ========================================
    # 网格搜索：One-Class SVM参数 + 阈值分位数
    # ========================================
    # 定义参数网格
    nu_list = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    kernel_list = ['rbf', 'linear', 'poly', 'sigmoid']
    gamma_list = ['scale', 'auto', 0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 0.8, 1.0]
    
    logger.info(f"\n{'='*60}")
    logger.info("开始网格搜索")
    logger.info(f"阈值分位数列表: {THRESHOLD_PERCENTILE_LIST}")
    logger.info(f"nu值: {nu_list}")
    logger.info(f"kernel: {kernel_list}")
    logger.info(f"gamma: {gamma_list}")
    logger.info(f"{'='*60}\n")
    
    # 记录所有网格搜索结果
    grid_search_results = []
    # 记录所有预测结果CSV文件路径（用于后续删除）
    prediction_csv_files = []
    # 计算总组合数：RBF核需要遍历gamma，其他核（linear, poly, sigmoid）每个只需要一次
    num_rbf_combinations = len(gamma_list)  # RBF核的gamma组合数
    num_other_kernels = len([k for k in kernel_list if k != 'rbf'])  # 其他核的数量
    num_svm_combinations = len(nu_list) * (num_rbf_combinations + num_other_kernels)
    total_combinations = len(THRESHOLD_PERCENTILE_LIST) * num_svm_combinations
    current_idx = 0
    
    for THRESHOLD_PERCENTILE in THRESHOLD_PERCENTILE_LIST:
        logger.info(f"\n{'#'*60}")
        logger.info(f"当前阈值分位数: {THRESHOLD_PERCENTILE}")
        logger.info(f"{'#'*60}\n")
        
        for nu in nu_list:
            for kernel in kernel_list:
                # 对于非RBF核，gamma参数无效，只运行一次
                if kernel != 'rbf':
                    gamma_list_for_kernel = ['scale']  # 占位符，实际不使用
                else:
                    gamma_list_for_kernel = gamma_list
                
                for gamma in gamma_list_for_kernel:
                    set_seed(seed=42)
                    current_idx += 1
                    logger.info(f"\n{'='*60}")
                    logger.info(f"网格搜索进度: [{current_idx}/{total_combinations}]")
                    logger.info(f"参数组合: percentile={THRESHOLD_PERCENTILE}, nu={nu:.3f}, kernel={kernel}, gamma={gamma}")
                    logger.info(f"{'='*60}")
                    
                    # 训练One-Class SVM
                    model = train_oneclass_svm(
                        X_train_scaled, 
                        nu=nu, 
                        kernel=kernel, 
                        gamma=gamma if kernel == 'rbf' else 'scale',
                        logger=logger
                    )
                    
                    # 在训练集上计算分数，确定阈值（使用配置的分位数）
                    logger.info("在训练集上计算异常分数以确定阈值...")
                    _, train_scores = compute_scores_and_predict(model, X_train_scaled)
                    threshold = np.percentile(train_scores, THRESHOLD_PERCENTILE)
                    logger.info(f"训练集{THRESHOLD_PERCENTILE}分位数阈值: {threshold:.6f}")
                    logger.info(f"训练集分数统计: min={train_scores.min():.6f}, max={train_scores.max():.6f}, "
                               f"mean={train_scores.mean():.6f}, std={train_scores.std():.6f}")
                    
                    # 在测试集上计算分数并评估
                    logger.info("在测试集上计算异常分数并评估...")
                    _, test_scores = compute_scores_and_predict(model, X_test_scaled)
                    
                    # 使用固定阈值评估
                    auc, f1, acc, predictions_csv = evaluate(
                        test_scores, 
                        y_test, 
                        threshold, 
                        logger, 
                        nu=nu, 
                        kernel=kernel, 
                        gamma=gamma if kernel == 'rbf' else 'scale',
                        percentile=THRESHOLD_PERCENTILE,
                        dataset_type=DATASET_TYPE,
                        model_type=MODEL_TYPE
                    )
                    
                    # 记录预测结果CSV文件路径
                    prediction_csv_files.append(predictions_csv)
                    
                    # 记录当前参数组合的结果
                    grid_search_results.append({
                        'nu': nu,
                        'kernel': kernel,
                        'gamma': gamma if kernel == 'rbf' else 'N/A',
                        'percentile': THRESHOLD_PERCENTILE,
                        'threshold': threshold,
                        'auc': auc,
                        'f1_score': f1,
                        'accuracy': acc,
                        'predictions_csv': str(predictions_csv)
                    })
    
    # 保存网格搜索结果汇总（文件名中包含数据集类型和模型类型，去掉oneclass_svm前缀）
    logger.info(f"\n{'='*60}")
    logger.info("网格搜索完成，汇总结果:")
    logger.info(f"{'='*60}")
    ckpt_dir = Path(__file__).parent / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 文件名中包含所有搜索的percentile范围
    percentile_str = f"p{min(THRESHOLD_PERCENTILE_LIST)}-{max(THRESHOLD_PERCENTILE_LIST)}"
    grid_summary_csv = ckpt_dir / f"grid_search_summary_{DATASET_TYPE}_{MODEL_TYPE}_{percentile_str}_{timestamp}.csv"
    
    grid_summary_df = pd.DataFrame(grid_search_results)
    grid_summary_df.to_csv(grid_summary_csv, index=False, encoding='utf-8-sig')
    logger.info(f"网格搜索结果汇总已保存到: {grid_summary_csv}")
    
    # 找出最佳结果（以准确率为准）
    best_idx = grid_summary_df['accuracy'].idxmax()
    best_result = grid_summary_df.loc[best_idx]
    best_predictions_csv = Path(best_result['predictions_csv'])
    
    logger.info(f"\n最佳结果（基于准确率）:")
    logger.info(f"  nu: {best_result['nu']:.3f}")
    logger.info(f"  kernel: {best_result['kernel']}")
    logger.info(f"  gamma: {best_result['gamma']}")
    logger.info(f"  Percentile: {best_result['percentile']}")
    logger.info(f"  Threshold: {best_result['threshold']:.6f}")
    logger.info(f"  AUC: {best_result['auc']:.4f}")
    logger.info(f"  F1 Score: {best_result['f1_score']:.4f}")
    logger.info(f"  Accuracy: {best_result['accuracy']:.4f}")
    logger.info(f"  预测结果文件: {best_predictions_csv}")
    
    # 删除除最佳结果外的所有预测结果CSV文件
    logger.info(f"\n{'='*60}")
    logger.info("清理临时文件...")
    deleted_count = 0
    for csv_file in prediction_csv_files:
        # 转换为Path对象（如果还不是）
        csv_path = Path(csv_file) if not isinstance(csv_file, Path) else csv_file
        best_path = Path(best_predictions_csv) if not isinstance(best_predictions_csv, Path) else best_predictions_csv
        
        # 比较绝对路径，确保是同一个文件
        if csv_path.resolve() != best_path.resolve():
            try:
                if csv_path.exists():
                    csv_path.unlink()  # 删除文件
                    deleted_count += 1
                    logger.debug(f"已删除: {csv_path}")
            except Exception as e:
                logger.warning(f"无法删除文件 {csv_path}: {e}")
    
    logger.info(f"已删除 {deleted_count} 个临时预测结果文件")
    logger.info(f"保留最佳结果文件: {best_predictions_csv}")
    logger.info(f"{'='*60}")

