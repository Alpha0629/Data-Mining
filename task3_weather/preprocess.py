import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging

# 获取logger，如果还没有配置则使用默认配置
logger = logging.getLogger(__name__)


# =========================================================
#  函 数 1：读取数据并打印信息
# =========================================================
def load_weather_csv(path='./dataset/weather.csv'):
    df = pd.read_csv(path)

    logger.info(f"数据形状：{df.shape}")
    logger.info(f"\n{df.head()}")

    # 清理列名（去除空格）
    df.columns = df.columns.str.strip()

    logger.info("\n所有列名：")
    for i, col in enumerate(df.columns):
        logger.info(f"{i}: '{col}' (repr: {repr(col)})")

    return df


# =========================================================
#  函 数 2：数据标准化 + 划分训练集、测试集
# =========================================================
def split_and_scale(df, target_col='OT', train_ratio=0.8):
    # 将整张表转换为矩阵
    X_all = df.values

    # 计算划分点
    split_index = int(len(df) * train_ratio)

    # 特征列（移除第一列时间戳）
    train_data = X_all[:split_index, 1:]
    test_data  = X_all[split_index:, 1:]

    logger.info(f"训练集 shape：{train_data.shape}")
    logger.info(f"测试集 shape：{test_data.shape}")

    # 标准化（只在训练集上 fit）
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_data)
    test_scaled  = scaler.transform(test_data)

    # 找目标列 OT 的列索引
    # 注意：train_data 已经去掉了第一列，因此需要 -1
    OT_index = df.columns.get_loc(target_col) - 1

    return train_scaled, test_scaled, OT_index, scaler


# =========================================================
#  函 数 3：滑动窗口（预测OT的变化量）
# =========================================================
def create_sequences(data, seq_len=12, target_col_index=0):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])                # 输入序列 [seq_len, feature_dim]
        # 目标：下一时间点OT的变化量 = OT[t+1] - OT[t]
        ot_current = data[i+seq_len-1][target_col_index]  # OT[t]
        ot_next = data[i+seq_len][target_col_index]       # OT[t+1]
        y.append(ot_next - ot_current)  # 变化量 delta = OT[t+1] - OT[t]
    return np.array(X), np.array(y)


# =========================================================
#  函 数 4：完整的预处理流程（最终你只需要调用这一句）
# =========================================================
def load_weather_dataset(path='./dataset/weather.csv', seq_len=12, target_col='OT'):
    # ① 读取 CSV
    df = load_weather_csv(path)

    # ② 划分 + 标准化
    train_scaled, test_scaled, OT_index, scaler = split_and_scale(df, target_col)

    # ③ 滑动窗口切分
    X_train, y_train = create_sequences(train_scaled, seq_len, OT_index)
    X_test, y_test = create_sequences(test_scaled, seq_len, OT_index)

    logger.info("\n最终数据维度：")
    logger.info(f"X_train: {X_train.shape}")
    logger.info(f"y_train: {y_train.shape}")
    logger.info(f"X_test: {X_test.shape}")
    logger.info(f"y_test: {y_test.shape}")

    return X_train, y_train, X_test, y_test, scaler, OT_index


# =========================================================
#  主 函 数（测试用）
# =========================================================
if __name__ == "__main__":
    X_train, y_train, X_test, y_test, scaler, OT_index = load_weather_dataset()
