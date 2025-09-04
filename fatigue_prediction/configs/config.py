"""
全局配置文件 - 包含数据路径、模型参数和训练参数
"""

import os

# 获取当前项目根目录
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# 数据相关配置
DATA_CONFIG = {
    'dataset_dir': os.path.join(PROJECT_ROOT, 'dataset'),  # 数据集根目录
    'strain_controlled_dir': os.path.join(PROJECT_ROOT, 'dataset', 'All data_Strain'),  # 应变控制数据目录
    'stress_controlled_dir': os.path.join(PROJECT_ROOT, 'dataset', 'All data_Stress'),  # 应力控制数据目录
    'strain_summary_file': os.path.join(PROJECT_ROOT, 'dataset', 'data_all_strain-controlled.csv'),  # 应变控制汇总文件
    'stress_summary_file': os.path.join(PROJECT_ROOT, 'dataset', 'data_all_stress-controlled.csv'),  # 应力控制汇总文件
    'sequence_length': 241,  # 时序数据标准长度
    'test_size': 0.2,  # 测试集比例
    'random_seed': 42,  # 随机种子
    'batch_size': 32,  # 批次大小
    'num_workers': 4,  # 数据加载线程数
}

# Transformer模型配置
MODEL_CONFIG = {
    # 输入特征维度
    'time_series_dim': 2,  # 加载路径时序数据维度
    'material_feature_dim': 4,  # 材料特征维度 (弹性模量、抗拉强度、屈服强度、泊松比)
    
    # Transformer参数
    'd_model': 128,  # 模型维度
    'nhead': 4,  # 注意力头数
    'num_encoder_layers': 3,  # 编码器层数
    'dim_feedforward': 512,  # 前馈网络维度
    'dropout': 0.1,  # Dropout率
    'activation': 'gelu',  # 激活函数
    
    # 位置编码
    'max_seq_length': 241,  # 最大序列长度
    
    # 输出参数
    'output_dim': 1,  # 输出维度 (疲劳寿命)
}

# 训练配置
TRAIN_CONFIG = {
    'epochs': 100,  # 训练轮数
    'learning_rate': 5e-4,  # 初始学习率
    'weight_decay': 2e-5,  # 权重衰减
    'lr_scheduler_factor': 0.3,  # 学习率衰减因子
    'lr_scheduler_patience': 10,  # 学习率调整耐心值
    'early_stop_patience': 30,  # 早停耐心值
    'gradient_clip_val': 1.0,  # 梯度裁剪值
    'save_dir': os.path.join(PROJECT_ROOT, 'fatigue_prediction', 'checkpoints'),  # 模型保存目录
    'logging_dir': os.path.join(PROJECT_ROOT, 'fatigue_prediction', 'logs'),  # 日志目录
}

# 评估指标配置
METRICS_CONFIG = {
    'metrics': ['mse', 'mae', 'r2', 'rmse'],  # 评估指标列表
}

# 系统配置
SYSTEM_CONFIG = {
    'use_gpu': True,  # 是否使用GPU
    'precision': 32,  # 计算精度 (16, 32)
} 