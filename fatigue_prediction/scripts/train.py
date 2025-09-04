"""
主训练脚本 - 用于训练模型并预测疲劳寿命
"""

import os
import sys
import logging
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
from datetime import datetime

# 添加项目根目录到系统路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.transformer_model import FatigueTransformer
from training.trainer import FatigueTrainer
from data.data_loader import FatigueDataLoader
from configs.config import DATA_CONFIG, MODEL_CONFIG, TRAIN_CONFIG

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练疲劳寿命预测模型')
    
    # 数据相关参数
    parser.add_argument('--data_type', type=str, default='strain', choices=['strain', 'stress'],
                        help='使用的数据类型: strain(应变控制), stress(应力控制)')
    parser.add_argument('--batch_size', type=int, default=None, help='批次大小')
    parser.add_argument('--num_workers', type=int, default=None, help='数据加载线程数')
    
    # 模型相关参数
    parser.add_argument('--d_model', type=int, default=None, help='模型维度')
    parser.add_argument('--nhead', type=int, default=None, help='注意力头数')
    parser.add_argument('--num_layers', type=int, default=None, help='编码器层数')
    
    # 训练相关参数
    parser.add_argument('--epochs', type=int, default=None, help='训练轮数')
    parser.add_argument('--lr', type=float, default=None, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=None, help='权重衰减')
    parser.add_argument('--resume', type=str, default=None, help='从检查点恢复训练')
    
    # 运行模式
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                        help='运行模式: train(训练), test(测试)')
    parser.add_argument('--model_path', type=str, default=None, help='测试或预测模式下的模型路径')

    parser.add_argument('--output_dir', type=str, default='/data/coding/metal_fatigue/results', help='结果保存目录')
    
    # GPU相关参数
    parser.add_argument('--use_gpu', action='store_true', default=None, help='是否使用GPU')
    
    return parser.parse_args()

def update_config(config, args, config_name):
    """使用命令行参数更新配置"""
    # 获取命令行参数字典
    args_dict = vars(args)
    
    # 更新配置
    for key, value in args_dict.items():
        if value is not None and key in config:
            logger.info(f"更新 {config_name} 配置: {key} = {value}")
            config[key] = value
    
    return config

def setup_output_dir(args):
    """设置输出目录"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f"{args.data_type}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建子目录
    checkpoints_dir = os.path.join(output_dir, 'checkpoints')
    logs_dir = os.path.join(output_dir, 'logs')
    plots_dir = os.path.join(output_dir, 'plots')
    
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # 更新训练配置
    TRAIN_CONFIG['save_dir'] = checkpoints_dir
    TRAIN_CONFIG['logging_dir'] = logs_dir
    
    return output_dir, plots_dir

def save_configs(output_dir, data_config, model_config, train_config):
    """保存配置到文件"""
    configs = {
        'data_config': data_config,
        'model_config': model_config,
        'train_config': train_config
    }
    
    config_path = os.path.join(output_dir, 'configs.json')
    with open(config_path, 'w') as f:
        json.dump(configs, f, indent=4)
    
    logger.info(f"配置保存到 {config_path}")

def train_model(args):
    """Train model"""
    # 更新配置
    data_config = update_config(DATA_CONFIG.copy(), args, 'DATA')
    model_config = update_config(MODEL_CONFIG.copy(), args, 'MODEL')
    train_config = update_config(TRAIN_CONFIG.copy(), args, 'TRAIN')
    
    # 设置输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f'{args.data_type}_train_{timestamp}')
    checkpoints_dir = os.path.join(output_dir, 'checkpoints')
    plots_dir = os.path.join(output_dir, 'plots')
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # 保存配置
    save_configs(output_dir, data_config, model_config, train_config)
    
    # 加载数据
    logger.info(f"正在加载 {args.data_type} 数据...")
    data_loader = FatigueDataLoader(data_config)
    
    train_loader, test_loader, file_names = data_loader.prepare_dataloaders(args.data_type)
    
    # 创建模型
    logger.info("正在创建模型...")
    model = FatigueTransformer(model_config)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型总参数量: {total_params}, 可训练参数量: {trainable_params}")
    
    # 初始化训练器
    logger.info("正在初始化训练器...")
    trainer = FatigueTrainer(model, train_config)
    
    # 训练模型
    logger.info("开始训练模型...")
    history = trainer.train(
        train_loader=train_loader,
        test_loader=test_loader,
        resume_from=args.resume,
        data_type=args.data_type
    )
    
    # 绘制训练历史
    logger.info("绘制训练历史...")
    plot_training_history(history, plots_dir)
    
    # 绘制预测结果
    logger.info("绘制预测结果...")
    if 'test_metrics' in history:
        plot_predictions(history['test_metrics'], plots_dir)
    
    # 保存训练结果和模型
    model_save_path = os.path.join(checkpoints_dir, f'model_{timestamp}.pth')
    logger.info(f"保存模型到: {model_save_path}")
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'history': history,
        'model_config': model_config,
        'timestamp': timestamp
    }, model_save_path)
    
    logger.info(f"训练完成！结果保存在 {output_dir}")
    
    return model, history, output_dir

def test_model(args):
    """测试模型"""
    # 加载数据
    logger.info(f"正在加载 {args.data_type} 数据...")
    data_loader = FatigueDataLoader(DATA_CONFIG)
    
    _, _, test_loader, file_names = data_loader.prepare_dataloaders(args.data_type)
    
    # 加载模型
    logger.info(f"正在加载模型 {args.model_path}...")
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # 创建模型
    model = FatigueTransformer(MODEL_CONFIG)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # 创建训练器
    trainer = FatigueTrainer(model, TRAIN_CONFIG, device=device)
    
    # 测试模型
    logger.info("开始测试模型...")
    test_loss, test_metrics = trainer.test(test_loader)
    
    # 设置输出目录
    output_dir = os.path.join(args.output_dir, 'test_results')
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # 绘制预测结果
    logger.info("绘制预测结果...")
    plot_predictions(test_metrics, plots_dir)
    
    # 保存测试结果
    result_path = os.path.join(output_dir, 'test_results.json')
    with open(result_path, 'w') as f:
        # 去掉NumPy数组等无法序列化的对象
        serializable_metrics = {k: v for k, v in test_metrics.items() 
                               if k not in ['predictions', 'targets']}
        json.dump(serializable_metrics, f, indent=4)
    
    logger.info(f"测试完成！结果保存在 {output_dir}")
    
    return test_metrics

def plot_training_history(history, output_dir):
    """Plot training history"""
    # Set up font support
    import matplotlib.pyplot as plt
    import matplotlib as mpl
        
    # Plot loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Training Loss')
    if 'test_loss' in history:
        plt.plot(history['test_loss'], label='Test Loss')
    plt.title('Training and Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot evaluation metrics
    if 'test_metrics' in history:
        metrics_list = history['test_metrics']
        if len(metrics_list) > 0:
            metrics = ['mse', 'mae', 'r2', 'rmse']
            
            for metric in metrics:
                if metric in metrics_list[0]:
                    values = [m[metric] for m in metrics_list]
                    
                    plt.figure(figsize=(10, 6))
                    plt.plot(values)
                    plt.title(f'Test {metric.upper()}')
                    plt.xlabel('Epoch')
                    plt.ylabel(metric.upper())
                    plt.grid(True)
                    plt.savefig(os.path.join(output_dir, f'{metric}_curve.png'), dpi=300, bbox_inches='tight')
                    plt.close()

def plot_predictions(metrics, output_dir):
    """Plot prediction results"""
    # Set up font support
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    
    if 'predictions' in metrics and 'targets' in metrics:
        predictions = metrics['predictions']
        targets = metrics['targets']
        
        # Create predictions vs actual scatter plot
        plt.figure(figsize=(10, 8))
        plt.scatter(targets, predictions, alpha=0.6)
        
        # Add perfect prediction line (diagonal)
        min_val = min(np.min(targets), np.min(predictions))
        max_val = max(np.max(targets), np.max(predictions))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        plt.title('Predicted vs Actual Values')
        plt.xlabel('Actual Fatigue Life (log)')
        plt.ylabel('Predicted Fatigue Life (log)')
        plt.grid(True)
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'predictions_vs_targets.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create residuals plot
        residuals = predictions - targets
        plt.figure(figsize=(10, 6))
        plt.scatter(targets, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title('Prediction Residuals')
        plt.xlabel('Actual Fatigue Life (log)')
        plt.ylabel('Residuals (Predicted - Actual)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'residuals.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create residuals histogram
        plt.figure(figsize=(10, 6))
        plt.hist(residuals, bins=20, alpha=0.6)
        plt.title('Residuals Distribution')
        plt.xlabel('Residuals (Predicted - Actual)')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'residuals_histogram.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save prediction results to CSV
        results_df = pd.DataFrame({
            'Actual': targets.flatten(),
            'Predicted': predictions.flatten(),
            'Residual': residuals.flatten()
        })
        
        results_df.to_csv(os.path.join(output_dir, 'prediction_results.csv'), index=False)

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 根据运行模式执行相应操作
    if args.mode == 'train':
        train_model(args)
    elif args.mode == 'test':
        if args.model_path is None:
            logger.error("测试模式需要指定模型路径 (--model_path)")
            return
        test_model(args)

if __name__ == "__main__":
    main() 