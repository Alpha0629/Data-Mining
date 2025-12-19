from data_loader import get_train_dataset, get_test_dataset
import numpy as np
import pandas as pd
import logging
import random
from datetime import datetime
from pathlib import Path
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import joblib


# ===============================
# 0. 随机种子设置
# ===============================
def set_seed(seed=42):
    """设置随机种子以确保实验结果可重复"""
    random.seed(seed)
    np.random.seed(seed)
    print(f"随机种子已设置为: {seed}")


# ===============================
# 1. 日志配置
# ===============================
def setup_logging(log_dir="ckpt", percentile=None):
    """配置日志系统，同时输出到文件和控制台"""
    # 创建日志目录
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # 生成日志文件名（带时间戳和分位数）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if percentile is not None:
        log_file = Path(log_dir) / f"oneclass_svm_training_p{percentile}_{timestamp}.log"
    else:
        log_file = Path(log_dir) / f"oneclass_svm_training_{timestamp}.log"
    
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
# 2. 数据加载：从Dataset转换为numpy数组
# ========================================
def load_data_as_numpy(train_dataset, test_dataset):
    """
    从PyTorch Dataset中提取numpy数组
    
    Args:
        train_dataset: 训练集Dataset
        test_dataset: 测试集Dataset
    
    Returns:
        X_train: 训练集特征 [n_samples, n_features]
        X_test: 测试集特征 [n_samples, n_features]
        y_test: 测试集标签 [n_samples]
    """
    # 提取训练集特征（训练集没有标签，或标签全为0）
    X_train = train_dataset.features.numpy()
    
    # 提取测试集特征和标签
    X_test = test_dataset.features.numpy()
    y_test = test_dataset.labels.numpy()
    
    return X_train, X_test, y_test


# ========================================
# 3. 训练One-Class SVM
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
# 4. 计算异常分数和预测
# ========================================
def compute_scores_and_predict(model, X_data):
    """
    使用One-Class SVM计算异常分数和预测
    
    Args:
        model: 训练好的One-Class SVM模型
        X_data: 数据特征
    
    Returns:
        predictions: 预测结果（1=异常, 0=正常）
        decision_scores: 决策函数值（距离超平面的距离，负值表示异常）
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
# 5. 评估函数
# ========================================
def evaluate(scores, labels, threshold, logger, output_dir="ckpt", nu=None, kernel=None, gamma=None, percentile=None):
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

    # 保存预测结果到 CSV
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if nu is not None:
        if percentile is not None:
            predictions_csv = Path(output_dir) / f"oneclass_svm_predictions_p{percentile}_nu{nu:.3f}_kernel{kernel}_gamma{gamma}_{timestamp}.csv"
        else:
            predictions_csv = Path(output_dir) / f"oneclass_svm_predictions_nu{nu:.3f}_kernel{kernel}_gamma{gamma}_{timestamp}.csv"
    else:
        if percentile is not None:
            predictions_csv = Path(output_dir) / f"oneclass_svm_predictions_p{percentile}_{timestamp}.csv"
        else:
            predictions_csv = Path(output_dir) / f"oneclass_svm_predictions_{timestamp}.csv"
    
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
    # 阈值分位数（可配置）
    THRESHOLD_PERCENTILE = 97  # 使用训练集的95分位数作为阈值，可以修改为其他值（如90, 95, 99等）
    
    # 设置日志（会自动创建 ckpt 文件夹）
    logger = setup_logging(log_dir="ckpt", percentile=THRESHOLD_PERCENTILE)
    logger.info(f"随机种子已设置为: 42")
    
    logger.info("开始加载数据集...")
    train_dataset = get_train_dataset(data_dir="datasets", filename="train-set.csv")
    test_dataset  = get_test_dataset(data_dir="datasets", filename="test-set.csv")
    
    # 转换为numpy数组
    X_train, X_test, y_test = load_data_as_numpy(train_dataset, test_dataset)
    
    logger.info(f"训练集形状: {X_train.shape}")
    logger.info(f"测试集形状: {X_test.shape}")
    logger.info(f"测试集标签分布: 正常={np.sum(y_test==0)}, 异常={np.sum(y_test==1)}")
    
    # 特征标准化（对One-Class SVM很重要）
    logger.info("对特征进行标准化...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    logger.info("特征标准化完成")
    
    # ========================================
    # 网格搜索：One-Class SVM参数
    # ========================================
    # 定义参数网格
    nu_list = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    kernel_list = ['rbf', 'linear', 'poly', 'sigmoid']
    gamma_list = ['scale', 'auto', 0.001, 0.01, 0.1, 1.0]
    
    logger.info(f"\n{'='*60}")
    logger.info("开始网格搜索")
    logger.info(f"阈值分位数: {THRESHOLD_PERCENTILE}")
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
    total_combinations = len(nu_list) * (num_rbf_combinations + num_other_kernels)
    current_idx = 0
    
    for nu in nu_list:
        for kernel in kernel_list:
            # 对于非RBF核，gamma参数无效，只运行一次
            if kernel != 'rbf':
                gamma_list_for_kernel = ['scale']  # 占位符，实际不使用
            else:
                gamma_list_for_kernel = gamma_list
            
            for gamma in gamma_list_for_kernel:
                current_idx += 1
                logger.info(f"\n{'='*60}")
                logger.info(f"网格搜索进度: [{current_idx}/{total_combinations}]")
                logger.info(f"参数组合: nu={nu:.3f}, kernel={kernel}, gamma={gamma}")
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
                    percentile=THRESHOLD_PERCENTILE
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
    
    # 保存网格搜索结果汇总
    logger.info(f"\n{'='*60}")
    logger.info("网格搜索完成，汇总结果:")
    logger.info(f"{'='*60}")
    Path("ckpt").mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    grid_summary_csv = Path("ckpt") / f"oneclass_svm_grid_search_summary_p{THRESHOLD_PERCENTILE}_{timestamp}.csv"
    
    grid_summary_df = pd.DataFrame(grid_search_results)
    grid_summary_df.to_csv(grid_summary_csv, index=False, encoding='utf-8-sig')
    logger.info(f"网格搜索结果汇总已保存到: {grid_summary_csv}")
    
    # 找出最佳结果
    best_idx = grid_summary_df['f1_score'].idxmax()
    best_result = grid_summary_df.loc[best_idx]
    best_predictions_csv = Path(best_result['predictions_csv'])
    
    logger.info(f"\n最佳结果:")
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
