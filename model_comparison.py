"""
金属疲劳寿命预测模型对比脚本

该脚本训练并比较CNN、LSTM和Transformer模型在金属疲劳寿命预测任务上的表现。
包括训练损失、测试损失、R²分数和训练时间的对比。
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import sys
import json
from tqdm import tqdm
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 确保项目目录可以被导入
sys.path.append(os.path.abspath("./"))

# 导入自定义模型和数据加载器、训练器
# 导入模型
from cnn.cnn_fcnn import CombinedModel as CNNCombinedModel, load_and_preprocess_data
from lstm.lstm_fcnn import CombinedModel as LSTMCombinedModel

# 导入Transformer模型的数据加载器和训练器
from fatigue_prediction.data.data_loader import FatigueDataLoader
from fatigue_prediction.training.trainer import FatigueTrainer
from fatigue_prediction.models.transformer_model import FatigueTransformer
from fatigue_prediction.configs.config import DATA_CONFIG, MODEL_CONFIG, TRAIN_CONFIG


def prepare_data_for_cnn_lstm(x_train, x_test, csv_train, csv_test, y_train, y_test):
    """为CNN和LSTM模型准备特定格式的数据"""
    # 创建数据集
    train_ds = TensorDataset(
        torch.tensor(x_train).float(),      # 材料特征
        torch.tensor(csv_train).float(),    # 时序数据
        torch.tensor(y_train).float().unsqueeze(1)  # 目标值
    )
    
    test_ds = TensorDataset(
        torch.tensor(x_test).float(),
        torch.tensor(csv_test).float(),
        torch.tensor(y_test).float().unsqueeze(1)
    )
    
    return train_ds, test_ds

def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch（用于CNN和LSTM模型）"""
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        # 获取数据
        material_feature, time_series, target = [item.to(device) for item in batch]
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(material_feature, time_series)
        
        # 计算损失
        loss = criterion(outputs, target)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    """在测试集上评估模型（用于CNN和LSTM模型）"""
    model.eval()
    total_loss = 0
    all_targets = []
    all_predictions = []
    
    with torch.no_grad():
        for batch in dataloader:
            # 获取数据
            material_feature, time_series, target = [item.to(device) for item in batch]
            
            # 前向传播
            outputs = model(material_feature, time_series)
            
            # 计算损失
            loss = criterion(outputs, target)
            total_loss += loss.item()
            
            # 收集预测和目标
            all_targets.extend(target.cpu().numpy())
            all_predictions.extend(outputs.cpu().numpy())
    
    # 计算评估指标
    test_loss = total_loss / len(dataloader)
    all_targets = np.array(all_targets)
    all_predictions = np.array(all_predictions)
    r2 = r2_score(all_targets, all_predictions)
    
    return test_loss, r2

def train_and_evaluate(model, train_loader, test_loader, epochs, device, model_name):
    """训练并评估CNN/LSTM模型"""
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    # 记录指标
    train_losses = []
    test_losses = []
    r2_scores = []
    
    # 记录训练时间
    start_time = time.time()
    
    for epoch in range(epochs):
        # 训练
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        
        # 测试
        test_loss, r2 = validate(model, test_loader, criterion, device)
        test_losses.append(test_loss)
        r2_scores.append(r2)
        
        # 记录进度
        if (epoch + 1) % 10 == 0:
            logger.info(f"{model_name} - Epoch {epoch+1}/{epochs}: "
                        f"Train Loss = {train_loss:.6f}, "
                        f"Test Loss = {test_loss:.6f}, "
                        f"R² = {r2:.6f}")
    
    # 计算总训练时间
    training_time = time.time() - start_time
    
    return {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'r2_scores': r2_scores,
        'training_time': training_time
    }



def get_results_from_trainer_history(history, training_time):
    """从FatigueTrainer的历史记录中提取结果"""
    # 获取训练损失和测试损失（确保是列表）
    train_losses = history.get('train_loss', [])
    test_losses = history.get('test_loss', [])
    
    # 确保它们是列表类型
    if not isinstance(train_losses, list):
        train_losses = [train_losses]
    if not isinstance(test_losses, list):
        test_losses = [test_losses]
    
    # 从metrics中提取R2分数
    r2_scores = []
    
    # 检查metrics是否存在且是列表
    if 'test_metrics' in history:
        # 如果是单个测试指标
        if isinstance(history['test_metrics'], dict):
            r2_scores = [history['test_metrics'].get('r2', 0.0)]
        # 如果是多个epoch的测试指标
        elif isinstance(history['test_metrics'], list):
            r2_scores = [metric.get('r2', 0.0) for metric in history['test_metrics']]
    elif 'metrics' in history and isinstance(history['metrics'], list):
        r2_scores = [metric.get('r2', 0.0) for metric in history['metrics']]
    
    if not r2_scores:
        logger.warning("无法从历史记录中找到R2分数")
        r2_scores = [0.0]
    
    # 确保数组长度一致
    max_length = max(len(train_losses), len(test_losses), len(r2_scores))
    
    # 如果任一数组为空，则创建相同长度的零数组
    if len(train_losses) == 0:
        train_losses = [0.0] * max_length
        logger.warning("训练损失数组为空，使用零填充")
    if len(test_losses) == 0:
        test_losses = [0.0] * max_length
        logger.warning("测试损失数组为空，使用零填充")
    if len(r2_scores) == 0:
        r2_scores = [0.0] * max_length
        logger.warning("R2分数数组为空，使用零填充")
    
    # 如果数组长度不等，则填充或裁剪到最大长度
    if len(train_losses) < max_length:
        train_losses = train_losses + [train_losses[-1] if train_losses else 0.0] * (max_length - len(train_losses))
    elif len(train_losses) > max_length:
        train_losses = train_losses[:max_length]
    
    if len(test_losses) < max_length:
        test_losses = test_losses + [test_losses[-1] if test_losses else 0.0] * (max_length - len(test_losses))
    elif len(test_losses) > max_length:
        test_losses = test_losses[:max_length]
        
    if len(r2_scores) < max_length:
        r2_scores = r2_scores + [r2_scores[-1] if r2_scores else 0.0] * (max_length - len(r2_scores))
    elif len(r2_scores) > max_length:
        r2_scores = r2_scores[:max_length]
        
    # 确保所有值都是浮点数
    train_losses = [float(x) if x is not None else 0.0 for x in train_losses]
    test_losses = [float(x) if x is not None else 0.0 for x in test_losses]
    r2_scores = [float(x) if x is not None else 0.0 for x in r2_scores]
    
    # 包含所有必要的数据
    results = {
        'train_losses': train_losses,
        'test_losses': test_losses,  # 使用测试损失替代验证损失
        'r2_scores': r2_scores,
        'training_time': float(training_time)
    }
    
    # 确认数据长度一致
    logger.info(f"Transformer训练历史: 训练损失 {len(train_losses)} 项, "
                f"测试损失 {len(test_losses)} 项, R2指标 {len(r2_scores)} 项")
    
    return results

def plot_comparison(results, epochs):
    """绘制模型性能对比图"""
    models = list(results.keys())
    model_count = len(models)
    
    # 如果只有一个或两个模型，调整颜色
    if model_count == 1:
        colors = ['blue']
    elif model_count == 2:
        colors = ['blue', 'green']
    else:
        # 生成足够的颜色
        import matplotlib.cm as cm
        cmap = cm.get_cmap('tab10')
        colors = [cmap(i) for i in range(model_count)]
    
    # 创建一个2x2的子图布局
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 训练损失对比
    ax = axs[0, 0]
    for i, model_name in enumerate(models):
        ax.plot(range(1, epochs+1), results[model_name]['train_losses'], 
                color=colors[i], label=model_name)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Comparison')
    ax.legend()
    ax.grid(True)
    
    # 2. 测试损失对比
    ax = axs[0, 1]
    for i, model_name in enumerate(models):
        ax.plot(range(1, epochs+1), results[model_name]['test_losses'], 
                color=colors[i], label=model_name)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Test Loss Comparison')
    ax.legend()
    ax.grid(True)
    
    # 3. 测试R²分数对比 (折线图)
    ax = axs[1, 0]
    for i, model_name in enumerate(models):
        ax.plot(range(1, epochs+1), results[model_name]['r2_scores'], 
                color=colors[i], label=model_name)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('R² Score')
    ax.set_title('Test R² Score (Training Process)')
    ax.legend()
    ax.grid(True)
    
    # 4. 最终R²分数对比 (带数值标注的柱状图)
    ax = axs[1, 1]
    
    # 获取最终的R²分数 (每个模型最后一个epoch的值)
    final_r2_scores = [results[model]['r2_scores'][-1] for model in models]
    
    # 设置柱状图
    bars = ax.bar(range(model_count), final_r2_scores, color=colors, tick_label=models)
    ax.set_ylabel('R² Score')
    ax.set_title('Final R² Score Comparison')
    ax.set_ylim([0, max(final_r2_scores) * 1.2])  # 设置合适的Y轴范围，留出标签空间
    
    # 在柱状图上显示具体数值
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f"{final_r2_scores[i]:.4f}", ha='center', va='bottom', fontsize=10)
    
    # 添加网格线使柱状图更易读
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    # 设置标题
    fig.suptitle('Metal Fatigue Life Prediction Model Comparison', fontsize=16)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)  # 为总标题腾出空间
    plt.savefig('/data/coding/metal_fatigue/model_comparison_results.png', dpi=300, bbox_inches='tight')

def main():
    """主函数"""
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 初始化结果字典
    results = {}
    
    # 定义模型参数计数函数
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 设置训练轮数
    epochs = 100
    
    #==============================================
    # 1. CNN和LSTM模型使用其原有数据加载和训练方式
    #==============================================
    # 数据路径设置    
    summary_path = "/data/coding/metal_fatigue/dataset/data_all_strain-controlled.csv"
    csv_folder_path = "/data/coding/metal_fatigue/dataset/All data_Strain/"
    # 加载数据
    x_all, csv_value_array, y_all, csv_files = load_and_preprocess_data(summary_path, csv_folder_path)
    
    # 划分训练集和测试集
    x_train, x_test, csv_train, csv_test, y_train, y_test = train_test_split(
        x_all, csv_value_array, y_all, test_size=0.2, random_state=42
    )
    
    logger.info(f"CNN/LSTM训练集大小: {len(x_train)}")
    logger.info(f"CNN/LSTM测试集大小: {len(x_test)}")
    
    # 准备数据集
    train_ds, test_ds = prepare_data_for_cnn_lstm(x_train, x_test, csv_train, csv_test, y_train, y_test)
    
    # 创建数据加载器
    batch_size = 32
    cnn_lstm_train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    cnn_lstm_test_loader = DataLoader(test_ds, batch_size=batch_size)
    
    # 获取特征维度
    input_dim = x_all.shape[1]
    
    # 保存参数数量
    params_count = {}
    
    # 训练CNN模型
    logger.info("初始化CNN模型...")
    cnn_model = CNNCombinedModel(input_dim).to(device)
    params_count['CNN'] = f"{count_parameters(cnn_model):,}"
    logger.info(f"CNN模型参数数量: {params_count['CNN']}")
    
    logger.info("训练CNN模型...")
    results['CNN'] = train_and_evaluate(cnn_model, cnn_lstm_train_loader, cnn_lstm_test_loader, epochs, device, "CNN")
    
    # 训练LSTM模型
    logger.info("初始化LSTM模型...")
    lstm_model = LSTMCombinedModel(input_dim).to(device)
    params_count['LSTM'] = f"{count_parameters(lstm_model):,}"
    logger.info(f"LSTM模型参数数量: {params_count['LSTM']}")
    
    logger.info("训练LSTM模型...")
    results['LSTM'] = train_and_evaluate(lstm_model, cnn_lstm_train_loader, cnn_lstm_test_loader, epochs, device, "LSTM")
    
    #==============================================
    # 2. Transformer模型使用FatigueDataLoader和FatigueTrainer
    #==============================================

    logger.info("使用FatigueDataLoader加载Transformer模型的数据...")
    
    # 修改配置
    data_config = DATA_CONFIG.copy()
    model_config = MODEL_CONFIG.copy()
    train_config = TRAIN_CONFIG.copy()
    
    # 设置轮数
    train_config['epochs'] = epochs
    
    # 初始化数据加载器
    transformer_data_loader = FatigueDataLoader(data_config)
    
    # 加载应变控制数据
    transformer_train_loader, transformer_test_loader, _ = transformer_data_loader.prepare_dataloaders(data_type='strain')
    
    # 获取材料特征维度
    for batch in transformer_train_loader:
        material_feature_dim = batch['material_feature'].shape[1]
        break
    
    logger.info(f"Transformer模型材料特征维度: {material_feature_dim}")
    
    # 更新模型配置
    model_config['material_feature_dim'] = material_feature_dim
    
    # 初始化Transformer模型
    logger.info("初始化Transformer模型...")
    transformer_model = FatigueTransformer(model_config).to(device)
    params_count['Transformer'] = f"{count_parameters(transformer_model):,}"
    logger.info(f"Transformer模型参数数量: {params_count['Transformer']}")
    
    # 初始化训练器
    transformer_trainer = FatigueTrainer(transformer_model, train_config, device=device)
    
    # 训练Transformer模型
    logger.info("训练Transformer模型...")
    transformer_start_time = time.time()
    
    # 训练模型
    transformer_history = transformer_trainer.train(
        train_loader=transformer_train_loader,
        test_loader=transformer_test_loader
    )
    
    # 计算训练时间
    transformer_training_time = time.time() - transformer_start_time
    
    # 从训练历史中提取结果
    results['Transformer'] = get_results_from_trainer_history(
        transformer_history,
        transformer_training_time
    )

    
    # 确保所有模型的结果长度一致（用于绘图）
    target_length = epochs
    for model_name in results:
        for key in ['train_losses', 'test_losses', 'r2_scores']:
            if len(results[model_name][key]) > target_length:
                results[model_name][key] = results[model_name][key][:target_length]
            elif len(results[model_name][key]) < target_length:
                logger.warning(f"{model_name}的{key}长度不足，将进行填充")
                last_value = results[model_name][key][-1]
                padding = [last_value] * (target_length - len(results[model_name][key]))
                results[model_name][key] = results[model_name][key] + padding
    
    # 绘制对比图
    plot_comparison(results, epochs)
    
    # 保存详细结果用于前端绘图 - 包含每个epoch的数据
    detailed_results = {
        model_name: {
            'train_losses': [float(val) if val is not None else 0.0 for val in results[model_name]['train_losses']],
            'test_losses': [float(val) if val is not None else 0.0 for val in results[model_name]['test_losses']],
            'r2_scores': [float(val) if val is not None else 0.0 for val in results[model_name]['r2_scores']],
            'training_time': float(results[model_name]['training_time']),
            # 添加前端期望的字段名
            'train_loss': [float(val) if val is not None else 0.0 for val in results[model_name]['train_losses']],
            'test_loss': [float(val) if val is not None else 0.0 for val in results[model_name]['test_losses']],
            'r2': [float(val) if val is not None else 0.0 for val in results[model_name]['r2_scores']]
        }
        for model_name in results
    }
    
    # 保存最终结果
    final_results = {
        model_name: {
            'final_train_loss': float(results[model_name]['train_losses'][-1]),
            'final_test_loss': float(results[model_name]['test_losses'][-1]),
            'final_r2': float(results[model_name]['r2_scores'][-1]),
            'training_time': float(results[model_name]['training_time'])
        }
        for model_name in results
    }
    
    # 输出最终结果
    logger.info("\n最终结果对比:")
    for model_name, metrics in final_results.items():
        logger.info(f"{model_name}:")
        logger.info(f"  训练损失: {metrics['final_train_loss']:.6f}")
        logger.info(f"  测试损失: {metrics['final_test_loss']:.6f}")
        logger.info(f"  R²分数: {metrics['final_r2']:.6f}")
        logger.info(f"  训练时间: {metrics['training_time']:.2f}秒")
    
    # 创建包含所有结果的字典
    comparison_data = {
        'results': final_results,  # 简化结果，向后兼容
        'detailed_results': detailed_results,  # 详细结果，包含每个epoch的数据
        'params': params_count,
        'epochs': epochs,  # 添加epochs数量信息
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # 保存结果到JSON文件
    try:
        with open('/data/coding/metal_fatigue/model_comparison_results.json', 'w') as f:
            json.dump(comparison_data, f, indent=4)
        logger.info("模型对比结果已保存到JSON文件")
    except Exception as e:
        logger.error(f"保存结果到JSON文件时出错: {e}")
        
    # 在项目中保存一份副本
    try:
        results_path = os.path.join('/data/coding/metal_fatigue/fatigue_prediction', 'results', 'model_comparison_results.json')
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, 'w') as f:
            json.dump(comparison_data, f, indent=4)
        logger.info(f"模型对比结果副本已保存到: {results_path}")
    except Exception as e:
        logger.warning(f"无法保存结果副本: {e}")

if __name__ == "__main__":
    main() 