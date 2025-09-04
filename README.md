# 金属多轴疲劳寿命预测系统

## 项目简介

本系统是一个基于深度学习的金属多轴疲劳寿命预测平台，集成了多种先进的深度学习模型（CNN、LSTM、Transformer），通过分析材料特征和时序数据，实现对金属材料疲劳寿命的精确预测。系统提供了友好的Web界面，支持数据上传、模型训练、预测和结果可视化等功能。

## 功能特点

- 多模型支持：集成CNN、LSTM和Transformer等多种深度学习模型
- 数据处理：支持应力数据和应变数据的预处理和特征工程
- 模型对比：提供不同模型性能的对比分析和可视化
- Web界面：直观的用户界面，支持交互式操作
- 结果可视化：详细的预测结果展示和性能指标分析
- 日志记录：完整的训练和预测过程日志

## 系统要求

### 硬件要求
- CPU: 推荐Intel i5/i7或同等性能处理器
- 内存: 最少8GB，推荐16GB或以上
- GPU: 推荐NVIDIA GPU（支持CUDA）用于模型训练
- 存储空间: 至少10GB可用空间

### 软件要求
- 操作系统: Linux（推荐Ubuntu 18.04或更高版本）
- Python: 3.8或更高版本
- CUDA: 11.0或更高版本（如使用GPU）

## 环境配置

1. 创建虚拟环境：
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
.\venv\Scripts\activate  # Windows
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 目录结构

```
metal_fatigue/
├── fatigue_prediction/    # 主预测模块
│   ├── web/              # Web应用
│   ├── models/           # 模型定义
│   ├── data/             # 数据处理
│   └── configs/          # 配置文件
├── cnn/                  # CNN模型实现
├── lstm/                 # LSTM模型实现
├── utils/                # 工具函数
├── dataset/              # 数据集目录
│   ├── All data_Strain/  # 应变数据
│   └── All data_Stress/  # 应力数据
├── results/              # 结果输出
├── uploads/              # 上传文件
└── start.py             # 启动脚本
```

## 使用说明

### 1. 启动系统

```bash
python start.py
```

默认配置：
- 主机: localhost
- 端口: 5000
- 调试模式: 开启

可选参数：
- `--host`: 指定主机地址
- `--port`: 指定端口号
- `--debug`: 启用调试模式
- `--log-level`: 设置日志级别（DEBUG/INFO/WARNING/ERROR/CRITICAL）

### 2. 数据准备

1. 数据格式要求：
   - 应力数据：CSV格式，包含时间序列应力数据
   - 应变数据：CSV格式，包含时间序列应变数据
   - 材料特征：CSV格式，包含材料基本特性参数

2. 数据目录结构：
   - 将应力数据放置在 `dataset/All data_Stress/` 目录
   - 将应变数据放置在 `dataset/All data_Strain/` 目录

### 3. 模型训练

1. 通过Web界面上传训练数据
2. 选择要使用的模型（CNN/LSTM/Transformer）
3. 设置训练参数（学习率、批次大小、训练轮数等）
4. 启动训练并监控进度

### 4. 预测使用

1. 上传待预测的数据
2. 选择已训练好的模型
3. 执行预测
4. 查看预测结果和可视化图表

### 5. 模型对比

使用模型对比工具：
```bash
python model_comparison.py
```

## API文档

系统提供以下主要API端点：

1. 数据管理
   - POST `/api/upload`: 上传数据文件
   - GET `/api/data/list`: 获取数据列表

2. 模型操作
   - POST `/api/model/train`: 启动模型训练
   - GET `/api/model/status`: 获取训练状态
   - POST `/api/model/predict`: 执行预测

3. 结果查询
   - GET `/api/results`: 获取预测结果
   - GET `/api/results/visualization`: 获取可视化数据

## 常见问题

1. **Q: 系统启动失败怎么办？**
   A: 检查以下几点：
   - 确保所有依赖都已正确安装
   - 检查端口是否被占用
   - 查看日志文件中的错误信息

2. **Q: 训练过程很慢怎么办？**
   A: 可以：
   - 使用GPU加速（如果可用）
   - 减小批次大小
   - 调整模型参数
   - 减少训练数据量

3. **Q: 预测结果不准确怎么办？**
   A: 尝试以下方法：
   - 增加训练数据量
   - 调整模型超参数
   - 尝试不同的模型架构
   - 检查数据预处理步骤

## 维护和更新

### 日志管理
- 日志文件位置：`fatigue_prediction/logs/`
- 定期检查日志文件大小
- 配置日志轮转策略

### 数据备份
- 定期备份训练数据
- 备份模型检查点
- 保存重要的预测结果