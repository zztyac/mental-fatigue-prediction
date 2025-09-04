"""
金属疲劳寿命预测模型 (LSTM+FCNN)

该模型结合了LSTM网络处理时序数据和全连接神经网络处理结构特征，
用于预测金属材料的疲劳寿命。
"""

import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset

# 机器学习相关库
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score,
    r2_score, mean_absolute_error, mean_squared_error
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPRegressor


def load_and_preprocess_data(summary_path, csv_folder_path):
    """
    加载并预处理疲劳数据
    
    Args:
        summary_path: 汇总CSV文件路径
        csv_folder_path: 包含时序数据的文件夹路径
    
    Returns:
        处理后的特征、时序数据和标签
    """
    # 加载汇总数据
    pd_train = pd.read_csv(summary_path, nrows=100000)
    
    # 加载时序数据文件
    csv_files = pd_train['load'].values
    value_list = []
    
    # 遍历每个时序文件并加载
    for i in range(len(csv_files)):
        try:
            one_df = pd.read_csv(csv_folder_path + csv_files[i], header=None).iloc[:, :2]
            value_list.append(one_df.values)
            # print(one_df.shape)
        except:
            print(csv_folder_path + csv_files[i], 'no here ')
    
    # 转换为NumPy数组
    csv_value_array = np.array(value_list)
    
    # 预处理数值特征
    num_cols = pd_train.select_dtypes(exclude=['object']).columns.tolist()[:-1]
    
    # 创建预处理管道
    num_si_step = ('si', SimpleImputer(strategy='median'))  # 使用中位数填充缺失值
    num_ss_step = ('ss', StandardScaler())  # 标准化特征
    num_steps = [num_si_step, num_ss_step]

    num_pipe = Pipeline(num_steps)
    num_transformers = [('num', num_pipe, num_cols)]
    
    ct = ColumnTransformer(transformers=num_transformers)
    
    # 应用预处理
    x_all = ct.fit_transform(pd_train)
    y_all = pd_train['Nf(label)'].values
    
    return x_all, csv_value_array, y_all, csv_files


def split_dataset(x_all, csv_value_array, y_all, csv_files, test_size=0.2, random_state=42):
    """
    分割数据集为训练集和测试集
    
    Args:
        x_all: 特征矩阵
        csv_value_array: 时序数据数组
        y_all: 目标变量
        csv_files: 文件名列表
        test_size: 测试集比例
        random_state: 随机种子
    
    Returns:
        训练集和测试集数据
    """
    x_train, x_test, csv_value_train, csv_value_test, y_train, y_test, csv_files_train, csv_files_test = train_test_split(
        x_all, csv_value_array, y_all, csv_files, test_size=test_size, random_state=random_state
    )
    
    print("训练集大小：", x_train.shape[0])
    print("测试集大小：", x_test.shape[0])
    
    return x_train, x_test, csv_value_train, csv_value_test, y_train, y_test


def create_dataloaders(x_train, x_test, csv_value_train, csv_value_test, y_train, y_test, batch_size=20):
    """
    创建PyTorch数据加载器
    
    Args:
        x_train, x_test: 特征数据
        csv_value_train, csv_value_test: 时序数据
        y_train, y_test: 标签数据
        batch_size: 批次大小
    
    Returns:
        训练集和测试集的DataLoader
    """
    # 创建TensorDataset
    train_ds = TensorDataset(
        torch.tensor(x_train).float(),
        torch.tensor(csv_value_train).float(), 
        torch.tensor(y_train).float().unsqueeze(1)
    )
    
    test_ds = TensorDataset(
        torch.tensor(x_test).float(),
        torch.tensor(csv_value_test).float(), 
        torch.tensor(y_test).float().unsqueeze(1)
    )
    
    # 创建DataLoader
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    return train_dl, test_dl


class DNNModel(nn.Module):
    """
    结构特征处理的深度神经网络模型
    """
    def __init__(self, input_dim):
        super().__init__()
        self.nor = nn.BatchNorm1d(input_dim)  # 批归一化层
        self.lin1 = nn.Linear(input_dim, 200)  # 第一个全连接层
        self.lin2 = nn.Linear(200, 100)        # 第二个全连接层
        self.lin3 = nn.Linear(100, 100)        # 第三个全连接层
        self.flatten = nn.Flatten()            # 展平层

    def forward(self, x):
        """前向传播"""
        x = F.relu(self.lin1(x))  # 应用ReLU激活函数
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        return x


class LSTMModel(nn.Module):
    """
    时序数据处理的LSTM模型
    """
    def __init__(self, input_dim):
        super().__init__()
        self.nor = nn.BatchNorm1d(input_dim)   # 批归一化层
        self.lstm = nn.LSTM(2, 5)              # LSTM层，输入维度2，隐藏状态维度5
        self.flatten = nn.Flatten()            # 展平层
        self.lin1 = nn.Linear(1205, 100)       # 全连接层

    def forward(self, x):
        """前向传播"""
        x, _ = self.lstm(x)        # LSTM处理
        x = self.flatten(x)        # 展平输出
        x = self.lin1(x)           # 线性变换
        return x


class CombinedModel(nn.Module):
    """
    结合DNN和LSTM的混合模型
    """
    def __init__(self, input_dim):
        super().__init__()
        self.strct_block = DNNModel(input_dim)  # 结构特征处理模块
        self.lstm_block = LSTMModel(input_dim)  # 时序数据处理模块
        self.flatten = nn.Flatten()
        # 组合层
        self.lin1 = nn.Linear(200, 100)
        self.lin2 = nn.Linear(100, 100)
        self.out = nn.Linear(100, 1)           # 输出层

    def forward(self, b_x, b_x_csv):
        """
        前向传播
        
        Args:
            b_x: 结构特征输入
            b_x_csv: 时序数据输入
        """
        x_strct = self.strct_block(b_x)        # 处理结构特征
        x_lstm = self.lstm_block(b_x_csv)      # 处理时序数据
        
        # 组合两种特征
        x = torch.stack((x_strct, x_lstm), dim=1)
        
        x = self.flatten(x)                    # 展平组合特征
        
        x = F.relu(self.lin1(x))               # 应用全连接层
        x = F.relu(self.lin2(x))
        x = self.out(x)                        # 生成预测
        
        return x


def train_epoch(dataloader, model, loss_fn, optimizer, device):
    """
    训练一个epoch
    
    Args:
        dataloader: 数据加载器
        model: 模型
        loss_fn: 损失函数
        optimizer: 优化器
        device: 计算设备
    """
    size = len(dataloader.dataset)
    model.train()  # 设置为训练模式
    
    for batch, (X, x_csv, y) in enumerate(dataloader):
        # 将数据移至计算设备
        X, x_csv, y = X.to(device), x_csv.to(device), y.to(device)
        
        # 计算预测和损失
        pred = model(X, x_csv)
        loss = loss_fn(pred, y)
        
        # 反向传播
        optimizer.zero_grad()  # 清除梯度
        loss.backward()        # 计算梯度
        optimizer.step()       # 更新参数
        
        # 打印训练进度
        if batch % 5 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def evaluate(dataloader, model, loss_fn, device):
    """
    评估模型性能
    
    Args:
        dataloader: 数据加载器
        model: 模型
        loss_fn: 损失函数
        device: 计算设备
    
    Returns:
        平均损失
    """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()  # 设置为评估模式
    test_loss = 0
    
    # 禁用梯度计算
    with torch.no_grad():
        for (X, x_csv, y) in dataloader:
            X, x_csv, y = X.to(device), x_csv.to(device), y.to(device)
            pred = model(X, x_csv)
            test_loss += loss_fn(pred, y).item()
            
    test_loss /= num_batches
    print(f"Avg loss: {test_loss:>8f} \n")
    return test_loss


def train_model(model, train_dl, test_dl, loss_fn, optimizer, epochs, device, model_save_path):
    """
    训练模型
    
    Args:
        model: 模型
        train_dl: 训练数据加载器
        test_dl: 测试数据加载器
        loss_fn: 损失函数
        optimizer: 优化器
        epochs: 训练轮数
        device: 计算设备
        model_save_path: 模型保存路径
    
    Returns:
        训练和测试损失历史
    """
    
    # 初始化最佳测试损失
    best_test_loss = float('inf')
    train_loss_list = []
    test_loss_list = []
    
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        # 训练
        train_epoch(train_dl, model, loss_fn, optimizer, device)
        
        # 评估
        train_loss = evaluate(train_dl, model, loss_fn, device)
        test_loss = evaluate(test_dl, model, loss_fn, device)
        
        # 记录损失
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        
        # 保存最佳模型
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"Best model saved! loss is {best_test_loss}")
    
    print("Done!")
    return train_loss_list, test_loss_list


def plot_loss_curves(train_loss, test_loss, save_path="/home/zty/code/mental_fatigue/lstm/lossvalue"):
    """
    绘制损失曲线
    
    Args:
        train_loss: 训练损失历史
        test_loss: 测试损失历史
        save_path: 图像保存路径
    """
    plt.figure()
    plt.plot(train_loss, label='train_loss')
    plt.plot(test_loss, label='test_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.legend(loc='upper right')
    plt.savefig(save_path, dpi=1200)
    plt.show()


def make_predictions(model, dataloader, device):
    """
    使用模型进行预测
    
    Args:
        model: 训练好的模型
        dataloader: 数据加载器
        device: 计算设备
    
    Returns:
        预测值和实际值
    """
    predict_list = []
    label_list = []
    
    # 设置为评估模式
    model.eval()
    
    with torch.no_grad():
        for (X, x_csv, y) in dataloader:
            X, x_csv, y = X.to(device), x_csv.to(device), y.to(device)
            predict_score = model(X, x_csv)
            predict_list.append(predict_score.flatten().detach().cpu().numpy().flatten())
            label_list.append(y.flatten().detach().cpu().numpy())
    
    # 合并所有批次的预测结果
    predict_array = np.hstack(predict_list)
    label_array = np.hstack(label_list)
    
    return predict_array, label_array


def save_predictions_to_csv(predictions, labels, filename):
    """
    将预测结果保存到CSV文件
    
    Args:
        predictions: 预测值
        labels: 实际标签
        filename: 输出文件名
    """
    data_to_write = np.column_stack((predictions, labels))
    
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # 写入标题行
        writer.writerow(['Prediction', 'Label'])
        
        # 写入数据
        for row in data_to_write:
            writer.writerow(row)
    
    print(f"预测结果已保存到 {filename}")


def main():
    """主函数，执行整个训练和评估流程"""
    # 数据路径设置
    summary_path = "/home/zty/code/mental_fatigue/dataset/data_all_strain-controlled.csv"  
    csv_folder_path = "/home/zty/code/mental_fatigue/dataset/All data_Strain/"  
    
    # 加载和预处理数据
    x_all, csv_value_array, y_all, csv_files = load_and_preprocess_data(summary_path, csv_folder_path)
    
    # 分割数据集
    x_train, x_test, csv_value_train, csv_value_test, y_train, y_test = split_dataset(
        x_all, csv_value_array, y_all, csv_files
    )
    
    # 创建数据加载器
    batch_size = 20
    train_dl, test_dl = create_dataloaders(
        x_train, x_test, csv_value_train, csv_value_test, y_train, y_test, batch_size
    )
    
    # 打印数据集大小
    print(f"\n训练集大小: {len(train_dl.dataset)}")
    print(f"测试集大小: {len(test_dl.dataset)}\n")
    
    # 设置计算设备
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using {device} device")
    
    # 特征维度
    dim_num = x_all.shape[-1]
    
    # 初始化模型
    dnn_block = DNNModel(dim_num).to(device)
    lstm_block = LSTMModel(dim_num).to(device)
    combined_model = CombinedModel(dim_num).to(device)
    
    # 设置损失函数和优化器
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(combined_model.parameters(), lr=1e-3)
    
    # 训练模型
    epochs = 100
    model_save_path = '/home/zty/code/mental_fatigue/lstm/best_model_weights_LSTM.pth'
    train_loss, test_loss = train_model(
        combined_model, train_dl, test_dl, loss_fn, optimizer, epochs, device, model_save_path
    )
    
    # 绘制训练曲线
    plot_loss_curves(train_loss, test_loss)
    
    # 加载最佳模型
    best_model = combined_model
    best_model.load_state_dict(torch.load(model_save_path))
    
    # 在测试集上进行预测
    print("\n" + "="*50)
    print("在测试集上评估模型性能:")
    predictions, labels = make_predictions(best_model, test_dl, device)
    
    # 计算更多评估指标
    mse = mean_squared_error(labels, predictions)
    mae = mean_absolute_error(labels, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(labels, predictions)
    
    print(f"均方误差 (MSE): {mse:.6f}")
    print(f"平均绝对误差 (MAE): {mae:.6f}")
    print(f"均方根误差 (RMSE): {rmse:.6f}")
    print(f"R² Score: {r2:.6f}")
    print("="*50 + "\n")
    
    # 保存预测结果
    save_predictions_to_csv(predictions, labels, "/home/zty/code/mental_fatigue/lstm/predictions_and_labels_train-LSTM.csv")


if __name__ == "__main__":
    main()








