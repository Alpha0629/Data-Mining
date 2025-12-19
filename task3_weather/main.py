
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging
import os
from datetime import datetime
import pandas as pd
import random

# 调用你的预处理文件
from preprocess import load_weather_dataset



# =========================================================
#  模 型：LSTM 回归器（支持单层/双层，可选注意力机制）
# =========================================================
class LSTMRegressor(nn.Module):
    def __init__(self, input_dim=21, hidden1=64, hidden2=32, seq_len=12, 
                 num_layers=2, use_attention=False):
        super().__init__()
        self.num_layers = num_layers
        self.use_attention = use_attention
        
        # 第一层LSTM
        self.lstm1 = nn.LSTM(input_dim, hidden1, batch_first=True)
        
        # 第二层LSTM（仅在num_layers=2时使用）
        if num_layers == 2:
            self.lstm2 = nn.LSTM(hidden1, hidden2, batch_first=True)
            lstm_output_dim = hidden2
        else:
            lstm_output_dim = hidden1
        
        # 注意力机制（可选）
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(lstm_output_dim, lstm_output_dim),
                nn.Tanh(),
                nn.Linear(lstm_output_dim, 1)
            )
        
        # 全连接层
        self.fc = nn.Linear(lstm_output_dim, 1)

    def forward(self, x):
        # x: [batch, seq_len, feature_dim]
        out, _ = self.lstm1(x)
        
        # 第二层LSTM（如果使用）
        if self.num_layers == 2:
            out, _ = self.lstm2(out)
        
        # out: [batch, seq_len, lstm_output_dim]
        
        # 使用注意力机制或直接取最后一个时间步
        if self.use_attention:
            # 使用注意力机制计算每个时间步的权重
            attention_scores = self.attention(out)  # [batch, seq_len, 1]
            attention_weights = torch.softmax(attention_scores, dim=1)  # [batch, seq_len, 1]
            # 加权求和
            out = (out * attention_weights).sum(dim=1)  # [batch, lstm_output_dim]
        else:
            # 直接取最后一个时间步的输出
            out = out[:, -1, :]  # [batch, lstm_output_dim]
        
        out = self.fc(out)
        return out.squeeze(1)



# =========================================================
#  训 练：单个 epoch
# =========================================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(X)

    return total_loss / len(loader.dataset)



# =========================================================
#  模 型 测 试 / 评 估
# =========================================================
def evaluate(model, loader, device):
    model.eval()
    preds, trues = [], []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            pred = model(X)

            preds.append(pred.cpu().numpy())
            trues.append(y.cpu().numpy())

    preds = np.concatenate(preds)
    trues = np.concatenate(trues)

    mae = mean_absolute_error(trues, preds)
    rmse = np.sqrt(mean_squared_error(trues, preds))

    return mae, rmse, preds, trues



# =========================================================
#  主 函 数（加载数据 → 训练 → 评估）
# =========================================================
def main():
    # -------------------------
    #  0. 固定随机种子（确保结果可复现）
    # -------------------------
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(SEED)
    
    # -------------------------
    #  1. 设置日志和保存路径
    # -------------------------
    # 创建带时间戳的文件夹
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join("ckpt", timestamp)
    os.makedirs(save_dir, exist_ok=True)
    
    # 配置日志：同时输出到控制台和文件
    log_file = os.path.join(save_dir, f"log_{timestamp}.txt")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()  # 输出到控制台
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info("=" * 50)
    logger.info("开始运行程序")
    logger.info(f"随机种子已固定: {SEED}")
    logger.info(f"保存目录: {save_dir}")
    
    # -------------------------
    #  模型配置选项
    # -------------------------
    # 选择LSTM层数：1 或 2
    NUM_LAYERS = 2  # 改为 1 使用单层LSTM，改为 2 使用双层LSTM
    
    # 选择是否使用注意力机制：True 或 False
    USE_ATTENTION = True  # True 使用注意力机制，False 不使用
    
    logger.info(f"\n模型配置：")
    logger.info(f"  LSTM层数: {NUM_LAYERS}")
    logger.info(f"  使用注意力机制: {USE_ATTENTION}")
    
    # -------------------------
    #  2. 设备设置
    # -------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"当前设备：{device}")

    # -------------------------
    #  3. 加载预处理后的数据
    # -------------------------
    seq_len = 12  # 序列长度
    X_train, y_train, X_test, y_test, scaler, OT_index = load_weather_dataset(
        path="./dataset/weather.csv",
        seq_len=seq_len,
        target_col="OT"
    )

    # 转为 Tensor
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test  = torch.tensor(X_test, dtype=torch.float32)
    y_test  = torch.tensor(y_test, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=64, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test),
                             batch_size=64, shuffle=False)

    # -------------------------
    #  4. 初始化模型
    # -------------------------
    feature_dim = X_train.shape[2]  # 应该是 21
    model = LSTMRegressor(
        input_dim=feature_dim, 
        seq_len=seq_len,
        num_layers=NUM_LAYERS,
        use_attention=USE_ATTENTION
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # -------------------------
    #  5. 开始训练
    # -------------------------
    epochs = 30
    logger.info("\n====== 开始训练模型 ======\n")

    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        logger.info(f"Epoch {epoch}/{epochs} - Train Loss: {loss:.6f}")

    # -------------------------
    #  6. 测试集评估（预测OT变化量，标准化后的值）
    # -------------------------
    mae, rmse, preds_delta_scaled, trues_delta_scaled = evaluate(model, test_loader, device)
    logger.info("\n====== 测试集结果（OT变化量预测，标准化后） ======")
    logger.info(f"MAE  = {mae:.4f}")
    logger.info(f"RMSE = {rmse:.4f}")

    # -------------------------
    #  7. 获取测试集中每个序列的当前OT值（用于计算下一时刻的OT）
    # -------------------------
    # X_test的最后一个时间步的OT值就是当前时刻的OT值
    X_test_np = X_test.numpy()
    current_ot_scaled = X_test_np[:, -1, OT_index]  # 每个序列最后一个时间步的OT值（标准化后）
    
    # -------------------------
    #  8. 将变化量和OT值还原到原始尺度
    # -------------------------
    # 直接从scaler中获取OT列的均值和标准差进行反标准化
    # 标准化公式：x_scaled = (x - mean) / std
    # 反标准化公式：x = x_scaled * std + mean
    # 对于变化量：delta = delta_scaled * std（因为变化量的均值接近0）
    
    ot_mean = scaler.mean_[OT_index]
    ot_std = scaler.scale_[OT_index]
    
    # 反标准化变化量：delta = delta_scaled * std
    preds_delta = preds_delta_scaled * ot_std
    trues_delta = trues_delta_scaled * ot_std
    
    # 反标准化当前OT值：OT = OT_scaled * std + mean
    current_ot = current_ot_scaled * ot_std + ot_mean
    
    # 计算下一时刻的OT值：OT[t+1] = OT[t] + delta
    preds_ot_next = current_ot + preds_delta  # 通过预测的变化量计算的下一时刻OT
    trues_ot_next = current_ot + trues_delta   # 真实的下一时刻OT
    
    # 计算原始尺度下的评估指标（基于变化量）
    mae_delta_original = mean_absolute_error(trues_delta, preds_delta)
    rmse_delta_original = np.sqrt(mean_squared_error(trues_delta, preds_delta))
    logger.info("\n====== 测试集结果（OT变化量预测，原始尺度） ======")
    logger.info(f"MAE  = {mae_delta_original:.4f}")
    logger.info(f"RMSE = {rmse_delta_original:.4f}")

    # -------------------------
    #  9. 保存模型
    # -------------------------
    model_path = os.path.join(save_dir, f"model_{timestamp}.pth")
    torch.save(model.state_dict(), model_path)
    logger.info(f"模型已保存到: {model_path}")

    # -------------------------
    #  10. 保存预测结果到CSV（使用原始尺度的值）
    # -------------------------
    pred_df = pd.DataFrame({
        'Current_OT': current_ot,                    # 当前时刻的OT值
        'True_Delta_OT': trues_delta,               # 真实的OT变化量
        'Predicted_Delta_OT': preds_delta,         # 预测的OT变化量
        'Delta_Error': trues_delta - preds_delta,   # 变化量的误差
        'Delta_Absolute_Error': np.abs(trues_delta - preds_delta),  # 变化量的绝对误差
        'True_OT_Next': trues_ot_next,              # 真实的下一时刻OT值
        'Predicted_OT_Next': preds_ot_next,         # 通过预测变化量计算的下一时刻OT值
        'OT_Error': trues_ot_next - preds_ot_next,  # OT值的误差
        'OT_Absolute_Error': np.abs(trues_ot_next - preds_ot_next)  # OT值的绝对误差
    })
    csv_path = os.path.join(save_dir, f"predictions_{timestamp}.csv")
    pred_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    logger.info(f"预测结果已保存到: {csv_path}")

    # -------------------------
    #  11. 绘制预测 vs 真实并保存（使用原始尺度的值）
    # -------------------------
    # 创建两个子图：一个显示变化量，一个显示OT值
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 10))
    
    # 子图1：OT变化量的预测
    ax1.plot(trues_delta[:500], label="True Delta OT", alpha=0.7)
    ax1.plot(preds_delta[:500], label="Predicted Delta OT", alpha=0.7)
    ax1.set_title("OT Change Prediction")
    ax1.set_xlabel("Time Index")
    ax1.set_ylabel("Delta Temperature")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 子图2：通过变化量计算的OT值
    ax2.plot(trues_ot_next[:500], label="True OT (Next)", alpha=0.7)
    ax2.plot(preds_ot_next[:500], label="Predicted OT (Next)", alpha=0.7)
    ax2.set_title("OT Value Prediction via Delta")
    ax2.set_xlabel("Time Index")
    ax2.set_ylabel("Temperature")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    fig_path = os.path.join(save_dir, f"visualization_{timestamp}.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    logger.info(f"可视化图已保存到: {fig_path}")
    plt.close()  # 关闭图形以释放内存
    
    logger.info("=" * 50)
    logger.info("程序运行完成！")
    logger.info(f"所有文件已保存到: {save_dir}")


# =========================================================
#  主 程 序 入口
# =========================================================
if __name__ == "__main__":
    main()
