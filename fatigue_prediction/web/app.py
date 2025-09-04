"""
金属多轴疲劳寿命预测系统 - Flask Web应用
"""

import os
import sys
import logging
import json
import time
import threading
import numpy as np
import pandas as pd
import torch
from torch.serialization import add_safe_globals
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash, send_from_directory
from werkzeug.utils import secure_filename
import zipfile

# 添加项目根目录到系统路径，使得可以导入其他模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入项目模块
from models.transformer_model import FatigueTransformer
from data.data_loader import FatigueDataLoader
from training.trainer import FatigueTrainer
from configs.config import DATA_CONFIG, MODEL_CONFIG, TRAIN_CONFIG, PROJECT_ROOT

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 当前文件路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 初始化Flask应用
app = Flask(__name__, 
           static_folder=os.path.join(current_dir, 'static'),
           template_folder=os.path.join(current_dir, 'templates'))

           
app.secret_key = 'fatigue_prediction_secret_key'

# 配置文件上传
app.config['UPLOAD_FOLDER'] = os.path.join(PROJECT_ROOT, 'uploads')
app.config['DATA_FOLDER'] = os.path.join(PROJECT_ROOT, 'dataset')
app.config['RESULTS_FOLDER'] = os.path.join(PROJECT_ROOT, 'results')
app.config['CHECKPOINT_FOLDER'] = os.path.join(PROJECT_ROOT, 'fatigue_prediction', 'checkpoints')
app.config['ALLOWED_EXTENSIONS'] = {'csv', 'zip'}
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB限制
app.config['STATIC_FOLDER'] = app.static_folder  # 添加静态文件夹路径到配置中

# 确保目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
os.makedirs(app.config['CHECKPOINT_FOLDER'], exist_ok=True)

# 日志记录静态文件路径配置
logger.info(f"静态文件路径: {app.static_folder}")
logger.info(f"模板文件路径: {app.template_folder}")

# 全局变量
training_stats = {
    'is_training': False,
    'current_epoch': 0,
    'total_epochs': 0,
    'train_loss': [],
    'test_loss': [],  
    'current_batch': 0,
    'total_batches': 0,
    'status': '就绪',
    'material_count': 0,
    'sample_count': 0,
    'start_time': None,
    'elapsed_time': 0
}

prediction_stats = {
    'is_predicting': False,
    'status': '就绪',
    'results': None,
    'material_stats': {}
}

# 辅助函数
def allowed_file(filename):
    """检查文件是否允许上传"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def start_training_thread(data_type, custom_params):
    """在后台线程中启动模型训练"""
    global training_stats
    
    try:
        # 初始化训练状态
        training_stats['is_training'] = True
        training_stats['status'] = '数据加载中...'
        training_stats['start_time'] = time.time()
        
        # 更新配置
        data_config = DATA_CONFIG.copy()
        model_config = MODEL_CONFIG.copy()
        train_config = TRAIN_CONFIG.copy()
        
        # 添加调试模式
        data_config['debug_mode'] = True
        
        # 应用自定义参数
        if custom_params:
            # 更新数据配置
            if 'batch_size' in custom_params:
                data_config['batch_size'] = int(custom_params['batch_size'])
            
            # 更新模型配置
            if 'd_model' in custom_params:
                model_config['d_model'] = int(custom_params['d_model'])
            if 'nhead' in custom_params:
                model_config['nhead'] = int(custom_params['nhead'])
            if 'num_encoder_layers' in custom_params:
                model_config['num_encoder_layers'] = int(custom_params['num_encoder_layers'])
            
            # 更新训练配置
            if 'epochs' in custom_params:
                train_config['epochs'] = int(custom_params['epochs'])
            if 'learning_rate' in custom_params:
                train_config['learning_rate'] = float(custom_params['learning_rate'])
        
        # 设置训练轮数
        training_stats['total_epochs'] = train_config['epochs']
        
        # 加载数据
        logger.info(f"开始加载 {data_type} 类型的数据")
        
        try:
            data_loader = FatigueDataLoader(data_config)
        except Exception as e:
            logger.error(f"初始化数据加载器失败: {e}", exc_info=True)
            raise ValueError(f"初始化数据加载器失败: {str(e)}")
        
        # 创建数据加载器
        try:
            train_loader, test_loader, file_names = data_loader.prepare_dataloaders(data_type)
                
            logger.info(f"数据加载完成，训练集: {len(train_loader.dataset)} 样本，测试集: {len(test_loader.dataset)} 样本")
        except Exception as e:
            logger.error(f"准备数据加载器失败: {e}", exc_info=True)
            raise ValueError(f"准备数据加载器失败: {str(e)}")
        
        # 更新样本计数
        training_stats['sample_count'] = len(train_loader.dataset) + len(test_loader.dataset)
        
        # 材料统计
        try:
            # 尝试不同的编码读取汇总文件
            encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin1', 'iso-8859-1', 'cp1252']
            df = None
            
            for encoding in encodings:
                try:
                    if data_type == 'strain':
                        df = pd.read_csv(data_config['strain_summary_file'], encoding=encoding)
                    elif data_type == 'stress':
                        df = pd.read_csv(data_config['stress_summary_file'], encoding=encoding)
                    logger.info(f"成功使用 {encoding} 编码读取汇总文件")
                    break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    logger.warning(f"使用 {encoding} 编码读取汇总文件失败: {e}")
                    continue
            
            if df is None:
                raise ValueError(f"无法使用任何编码读取汇总文件")
                
            training_stats['material_count'] = len(df['E(Gpa)'].unique())
            logger.info(f"识别到 {training_stats['material_count']} 种不同的材料")
        except Exception as e:
            logger.error(f"读取汇总文件失败: {e}", exc_info=True)
            training_stats['material_count'] = 0
        
        # 创建模型
        training_stats['status'] = '模型初始化中...'
        
        try:
            logger.info("创建模型...")
            model = FatigueTransformer(model_config)
            logger.info("模型创建成功")
        except Exception as e:
            logger.error(f"创建模型失败: {e}", exc_info=True)
            raise ValueError(f"创建模型失败: {str(e)}")
        
        # 创建训练器
        try:
            logger.info("创建训练器...")
            trainer = FatigueTrainer(model, train_config)
            logger.info("训练器创建成功")
        except Exception as e:
            logger.error(f"创建训练器失败: {e}", exc_info=True)
            raise ValueError(f"创建训练器失败: {str(e)}")
        
        # 自定义回调函数，用于更新训练状态
        def update_training_stats(epoch, batch, batch_count, loss):
            training_stats['current_epoch'] = epoch + 1
            training_stats['current_batch'] = batch + 1
            training_stats['total_batches'] = batch_count
            training_stats['elapsed_time'] = time.time() - training_stats['start_time']
            if batch == batch_count - 1:  # 仅在批次结束时更新损失
                training_stats['train_loss'].append(loss)
                training_stats['status'] = f'训练中 - 第 {epoch+1}/{train_config["epochs"]} 轮'
        
        def update_test_stats(epoch, loss, mse, mae, r2, rmse):
            training_stats['test_loss'].append(loss)
        
        # 设置回调函数
        trainer.set_training_callbacks(
            on_batch_end=update_training_stats,
            on_test_end=update_test_stats
        )
        
        # 创建输出目录
        try:
            # 训练模型
            training_stats['status'] = '训练开始...'
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(app.config['RESULTS_FOLDER'], f'{data_type}_train_{timestamp}')
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"创建输出目录: {output_dir}")
            
            # 保存配置到输出目录
            with open(os.path.join(output_dir, 'config.json'), 'w') as f:
                config = {
                    'data_config': data_config,
                    'model_config': model_config,
                    'train_config': train_config
                }
                # 使用make_json_serializable函数处理配置，确保所有值都可序列化
                json.dump(make_json_serializable(config), f, indent=4)
            logger.info("配置已保存")
        except Exception as e:
            logger.error(f"创建输出目录或保存配置失败: {e}", exc_info=True)
            raise ValueError(f"创建输出目录或保存配置失败: {str(e)}")
        
        # 执行训练
        try:
            logger.info("开始训练...")
            history = trainer.train(
                train_loader=train_loader,
                test_loader=test_loader
            )
            logger.info("训练完成")
        except Exception as e:
            logger.error(f"训练过程中发生错误: {e}", exc_info=True)
            raise ValueError(f"训练过程中发生错误: {str(e)}")
        
        # 保存训练结果和模型
        try:
            model_save_path = os.path.join(app.config['CHECKPOINT_FOLDER'], f'model_{timestamp}.pth')
            logger.info(f"保存模型到: {model_save_path}")
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'history': history,
                'model_config': model_config,
                'timestamp': timestamp
            }, model_save_path)
            
            # 更新训练状态
            training_stats['status'] = '训练完成'
            training_stats['is_training'] = False
            
            # 保存训练历史到JSON文件
            history_path = os.path.join(output_dir, 'history.json')
            logger.info(f"保存训练历史到: {history_path}")
            
            with open(history_path, 'w') as f:
                # 使用make_json_serializable函数处理所有数据，确保可序列化
                serialized_history = make_json_serializable(history)
                json.dump(serialized_history, f, indent=4)
            
            logger.info(f"训练完成，模型保存至: {model_save_path}")
        except Exception as e:
            logger.error(f"保存模型或训练历史失败: {e}", exc_info=True)
            raise ValueError(f"保存模型或训练历史失败: {str(e)}")
        
    except Exception as e:
        logger.error(f"训练过程中发生错误: {str(e)}", exc_info=True)
        training_stats['status'] = f'错误: {str(e)}'
        training_stats['is_training'] = False

def start_prediction_thread(data_type, model_path):
    """在后台线程中启动模型预测"""
    global prediction_stats
    
    try:
        # 初始化预测状态
        prediction_stats['is_predicting'] = True
        prediction_stats['status'] = '数据加载中...'
        prediction_stats['start_time'] = time.time()
        
        # 加载数据配置
        data_config = DATA_CONFIG.copy()
        
        # 配置数据加载器
        try:
            data_loader = FatigueDataLoader(data_config)
            logger.info("数据加载器初始化成功")
        except Exception as e:
            logger.error(f"初始化数据加载器失败: {e}", exc_info=True)
            raise ValueError(f"初始化数据加载器失败: {str(e)}")
        
        # 加载测试数据
        try:
            prediction_stats['status'] = '加载预测数据...'
            train_loader, test_loader, file_names = data_loader.prepare_dataloaders(data_type)
            
            if len(test_loader.dataset) == 0:
                raise ValueError(f"未找到任何测试数据。请确保数据文件存在且格式正确。")
            
            # 检查测试数据的结构
            logger.info(f"成功加载测试数据: {len(test_loader.dataset)}个样本")
            
            # 尝试检查DataLoader返回的数据结构
            try:
                sample_iter = iter(test_loader)
                sample_batch = next(sample_iter)
                logger.info(f"数据批次类型: {type(sample_batch)}")
                
                if isinstance(sample_batch, dict):
                    logger.info(f"批次数据是字典，包含键: {list(sample_batch.keys())}")
                elif isinstance(sample_batch, (list, tuple)):
                    logger.info(f"批次数据是{type(sample_batch).__name__}，长度为: {len(sample_batch)}")
                    for i, item in enumerate(sample_batch):
                        if isinstance(item, torch.Tensor):
                            logger.info(f"  项目 {i} 是张量，形状: {item.shape}, 类型: {item.dtype}")
                        else:
                            logger.info(f"  项目 {i} 类型: {type(item)}")
                else:
                    logger.info(f"批次数据是其他类型: {type(sample_batch)}")
            except StopIteration:
                logger.warning("无法提取样本批次用于检查")
            except Exception as e:
                logger.warning(f"检查数据批次结构时出错: {e}")
                
        except Exception as e:
            logger.error(f"加载测试数据失败: {e}", exc_info=True)
            raise ValueError(f"加载测试数据失败: {str(e)}")
        
        # 添加需要的NumPy类型到安全全局列表
        try:
            # 添加numpy.core.multiarray.scalar
            from numpy.core.multiarray import scalar
            add_safe_globals([scalar])
            
            # 添加numpy.dtype
            from numpy import dtype
            add_safe_globals([dtype])
            
            # 添加其他可能需要的NumPy类型
            import numpy as np
            add_safe_globals([np.ndarray, np.generic])
            
            logger.info("已添加NumPy类型到安全全局列表用于模型预测")
        except (ImportError, ModuleNotFoundError) as e:
            logger.warning(f"无法导入NumPy类型进行安全加载: {e}")
        
        # 加载模型
        prediction_stats['status'] = '加载模型中...'
        try:
            # 先尝试使用weights_only=False加载
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            logger.info(f"使用weights_only=False成功加载模型: {model_path}")
        except Exception as e1:
            logger.warning(f"使用weights_only=False加载模型失败: {e1}")
            try:
                # 如果失败，尝试使用默认设置加载
                checkpoint = torch.load(model_path, map_location='cpu')
                logger.info(f"使用默认设置成功加载模型: {model_path}")
            except Exception as e2:
                logger.error(f"无法加载模型: {e2}")
                raise ValueError(f"无法加载模型: {str(e2)}")
        
        # 检查模型文件的格式        
        logger.info(f"加载的模型检查点包含以下键: {list(checkpoint.keys())}")
        
        # 获取model_config，如果不存在则使用默认配置
        if 'model_config' not in checkpoint:
            logger.warning("模型文件中缺少'model_config'，使用默认配置")
            model_config = MODEL_CONFIG.copy()
            # 设置基本的输入输出维度
            model_config['time_series_dim'] = 2  # 默认时序维度
            model_config['material_feature_dim'] = 4  # 默认材料特征维度
            model_config['output_dim'] = 1  # 默认输出维度
        else:
            model_config = checkpoint['model_config']
            
        # 创建模型
        try:
            model = FatigueTransformer(model_config)
            
            # 检查state_dict键
            if 'model_state_dict' not in checkpoint:
                if 'state_dict' in checkpoint:
                    logger.warning("使用'state_dict'替代'model_state_dict'")
                    model.load_state_dict(checkpoint['state_dict'])
                else:
                    # 如果既没有model_state_dict也没有state_dict，检查checkpoint本身是否是state_dict
                    if any(k.endswith('.weight') or k.endswith('.bias') for k in checkpoint.keys()):
                        logger.warning("检测到checkpoint本身似乎是state_dict，直接加载")
                        model.load_state_dict(checkpoint)
                    else:
                        raise ValueError("无法在模型文件中找到权重信息")
            else:
                model.load_state_dict(checkpoint['model_state_dict'])
                
            model.eval()  # 切换到评估模式
        except Exception as e:
            logger.error(f"创建或加载模型权重失败: {e}", exc_info=True)
            raise ValueError(f"创建或加载模型权重失败: {str(e)}")
        
        # 选择设备并将模型移动到设备上
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"使用设备: {device}")
            model = model.to(device)
        except Exception as e:
            logger.error(f"将模型移动到设备时出错: {e}", exc_info=True)
            # 尝试继续使用CPU
            device = torch.device('cpu')
            logger.info("回退到CPU设备")
            model = model.to(device)
        
        # 统计材料信息
        if data_type == 'strain':
            material_df = pd.read_csv(data_config['strain_summary_file'])
        elif data_type == 'stress':
            material_df = pd.read_csv(data_config['stress_summary_file'])
        # 开始预测
        prediction_stats['status'] = '预测中...'
        
        try:
            predictions = []
            targets = []
            material_names = []
            
            # 不使用梯度计算
            with torch.no_grad():
                for batch_idx, batch in enumerate(test_loader):
                    # 支持不同的批次数据格式
                    if isinstance(batch, dict):
                        # 字典格式 - 将所有张量移动到同一设备
                        time_series = batch['time_series'].to(device)
                        material_features = batch['material_feature'].to(device)
                        # 确保target也在正确的设备上
                        target = batch['target'].to(device) if isinstance(batch['target'], torch.Tensor) else batch['target']
                        # 如果存在attention_mask，确保它也在同一设备上
                        attention_mask = batch.get('attention_mask')
                        if attention_mask is not None:
                            attention_mask = attention_mask.to(device)
                        # 获取当前批次对应的文件名
                        current_file_names = file_names[1][batch_idx*test_loader.batch_size:(batch_idx+1)*test_loader.batch_size]
                    elif isinstance(batch, (list, tuple)):
                        if len(batch) == 3:
                            # 元组格式，包含3个元素（时序数据，材料特征，目标值）
                            time_series, material_features, target = batch
                            # 将所有张量移动到同一设备
                            time_series = time_series.to(device)
                            material_features = material_features.to(device)
                            # 确保target也在正确的设备上
                            target = target.to(device) if isinstance(target, torch.Tensor) else target
                            attention_mask = None
                            # 获取当前批次对应的文件名
                            current_file_names = file_names[1][batch_idx*test_loader.batch_size:(batch_idx+1)*test_loader.batch_size]
                        elif len(batch) == 4:
                            # 元组格式，包含4个元素（时序数据，材料特征，目标值，注意力掩码）
                            time_series, material_features, target, attention_mask = batch
                            # 将所有张量移动到同一设备
                            time_series = time_series.to(device)
                            material_features = material_features.to(device)
                            # 确保target也在正确的设备上
                            target = target.to(device) if isinstance(target, torch.Tensor) else target
                            # 确保attention_mask在同一设备上
                            if attention_mask is not None:
                                attention_mask = attention_mask.to(device)
                            # 获取当前批次对应的文件名
                            current_file_names = file_names[1][batch_idx*test_loader.batch_size:(batch_idx+1)*test_loader.batch_size]
                        else:
                            logger.warning(f"批次数据包含意外数量的元素：{len(batch)}，尝试使用前三个元素")
                            # 尝试使用前三个元素，并确保所有张量在同一设备上
                            time_series = batch[0].to(device)
                            material_features = batch[1].to(device)
                            # 确保target也在正确的设备上
                            target = batch[2].to(device) if isinstance(batch[2], torch.Tensor) else batch[2]
                            # 如果有第四个元素且是attention_mask，确保它也在同一设备上
                            attention_mask = None if len(batch) <= 3 else batch[3].to(device)
                            # 获取当前批次对应的文件名（如果可能）
                            try:
                                current_file_names = file_names[1][batch_idx*test_loader.batch_size:(batch_idx+1)*test_loader.batch_size]
                            except:
                                current_file_names = ["unknown"] * len(time_series)
                    else:
                        logger.error(f"不支持的批次数据类型: {type(batch)}")
                        raise ValueError(f"不支持的批次数据类型: {type(batch)}")
                    
                    # 前向传播（根据模型是否需要attention_mask调整调用）
                    try:
                        # 记录设备信息以进行调试
                        logger.info(f"时序数据设备: {time_series.device}")
                        logger.info(f"材料特征设备: {material_features.device}")
                        if attention_mask is not None:
                            logger.info(f"注意力掩码设备: {attention_mask.device}")
                        
                        # 尝试调用需要attention_mask的版本
                        if attention_mask is not None and hasattr(model, 'forward') and 'attention_mask' in model.forward.__code__.co_varnames:
                            # 确保模型和所有输入都在同一设备上
                            output = model(time_series, material_features, attention_mask=attention_mask)
                        else:
                            # 如果不需要attention_mask或它不存在
                            output = model(time_series, material_features)
                    except TypeError as e:
                        logger.warning(f"模型前向传播出错: {e}，尝试不同的参数组合")
                        # 尝试不同的参数组合
                        try:
                            output = model(time_series, material_features)
                        except Exception as e2:
                            logger.error(f"所有尝试都失败: {e2}")
                            raise ValueError(f"模型前向传播失败: {str(e2)}")
                    
                    # 收集预测结果
                    # 处理不同的输出格式
                    if isinstance(output, tuple):
                        # 如果输出是元组（例如，包含预测值和注意力权重）
                        pred_tensor = output[0].detach().cpu() 
                        predictions.append(pred_tensor.numpy())
                    else:
                        # 如果输出只是预测值
                        pred_tensor = output.detach().cpu()
                        predictions.append(pred_tensor.numpy())
                    
                    # 确保target在CPU上进行numpy转换
                    if isinstance(target, torch.Tensor):
                        target_cpu = target.detach().cpu().numpy()
                    else:
                        target_cpu = target
                    targets.append(target_cpu)
                    
                    # 使用文件名前缀作为材料标识符
                    for file_name in current_file_names:
                        try:
                            # 定义已知材料列表
                            known_materials = [
                                "1Cr18Ni9T", "16MnR", "45mild", "304SS", "410", "1045HR", 
                                "2024-T3", "6061-T6", "7075-T651", "AISI 316L", "Al5083", 
                                "AZ31B", "AZ61A", "BT9", "CA", "CP-Ti", "CS", "CuZn37", 
                                "E235", "E355", "GH4169", "Haynes188", "HRB335", "inconel718", 
                                "mild", "PA38", "pureTi", "q235b", "S45C", "S347", "S460N", 
                                "SNCM630", "TC4", "X5CrNi", "ZK60", "5% chrome work roll steel",
                                "30CrMnSiA", "2024-STSA", "2024-T3", "2198-T8", "6082-T6",
                                "7075-T651", "LY12CZ", "SM45C"
                            ]
                            
                            # 检查文件名是否以任何已知材料名称开头
                            material_name = None
                            for material in known_materials:
                                if file_name.startswith(material):
                                    material_name = material
                                    break
                            
                            # 如果没有匹配到任何已知材料，尝试提取前缀作为材料名
                            if material_name is None:
                                # 提取第一个连字符之前的部分作为材料名
                                material_name = file_name.split('-')[0]
                            
                            material_names.append(material_name)
                            logger.info(f"从文件名 {file_name} 提取材料名: {material_name}")
                        except Exception as e:
                            logger.warning(f"从文件名 {file_name} 提取材料名失败: {e}")
                            material_names.append("Unknown-Material")
            
            # 将所有批次的预测结果连接起来
            predictions = np.vstack(predictions)
            targets = np.vstack(targets)
            
            # 计算指标
            mse = np.mean((predictions - targets) ** 2)
            mae = np.mean(np.abs(predictions - targets))
            rmse = np.sqrt(mse)
            
            # 计算R2分数
            if np.var(targets) == 0:
                r2 = 0.0  # 避免除以零
                logger.warning("目标值的方差为零，无法计算R2分数")
            else:
                r2 = 1 - (np.sum((predictions - targets) ** 2) / np.sum((targets - np.mean(targets)) ** 2))
            
            # 收集材料级别的统计信息
            unique_materials = list(set(material_names))
            material_stats = {}
            
            for material in unique_materials:
                indices = [i for i, name in enumerate(material_names) if name == material]
                material_predictions = predictions[indices]
                material_targets = targets[indices]
                
                # 计算该材料的指标
                material_mse = np.mean((material_predictions - material_targets) ** 2)
                material_mae = np.mean(np.abs(material_predictions - material_targets))
                
                material_stats[material] = {
                    'mse': float(material_mse),
                    'mae': float(material_mae),
                    'sample_count': len(indices),
                    'targets': material_targets.flatten().tolist(),  # 添加targets数组
                    'predictions': material_predictions.flatten().tolist()  # 添加predictions数组
                }
            
            # 更新预测状态
            prediction_stats['results'] = {
                'mse': float(mse),
                'mae': float(mae),
                'r2': float(r2),
                'rmse': float(rmse),
                'targets': targets.flatten().tolist(),  # 添加targets数组
                'predictions': predictions.flatten().tolist(),  # 添加predictions数组
                'file_names': file_names[1]  # 添加文件名列表
            }
            
            # 添加日志，显示识别到的所有材料
            logger.info(f"识别到的材料种类: {len(unique_materials)}")
            for material in unique_materials:
                logger.info(f"  - {material}")
            
            prediction_stats['material_stats'] = material_stats
            prediction_stats['status'] = '预测完成'
            prediction_stats['is_predicting'] = False
            prediction_stats['elapsed_time'] = time.time() - prediction_stats['start_time']
            
            # 保存结果
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(app.config['RESULTS_FOLDER'], f'predict_{timestamp}')
            os.makedirs(output_dir, exist_ok=True)
            
            # 保存预测结果为CSV
            results_df = pd.DataFrame({
                'Material': material_names,
                'Predicted': predictions.flatten(),
                'Actual': targets.flatten(),
                'Error': predictions.flatten() - targets.flatten()
            })
            results_df.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)
            
            # 保存汇总指标
            with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
                metrics = {
                    'mse': float(mse),
                    'mae': float(mae),
                    'r2': float(r2),
                    'rmse': float(rmse),
                    'sample_count': len(predictions),
                    'material_count': len(unique_materials)
                }
                # 使用make_json_serializable确保所有值都可序列化
                json.dump(make_json_serializable(metrics), f, indent=4)
            
            logger.info(f"预测完成，结果保存至: {output_dir}")
        except Exception as e:
            logger.error(f"预测过程中发生错误: {e}", exc_info=True)
            raise ValueError(f"预测过程中发生错误: {str(e)}")
            
    except Exception as e:
        logger.error(f"预测过程中发生错误: {str(e)}", exc_info=True)
        prediction_stats['status'] = f'错误: {str(e)}'
        prediction_stats['is_predicting'] = False

def get_available_models():
    """获取可用的已训练模型列表"""
    models = []
    checkpoint_dir = app.config['CHECKPOINT_FOLDER']
    
    # 添加需要的NumPy类型到安全全局列表
    try:
        # 添加numpy.core.multiarray.scalar
        from numpy.core.multiarray import scalar
        add_safe_globals([scalar])
        
        # 添加numpy.dtype
        from numpy import dtype
        add_safe_globals([dtype])
        
        # 添加其他可能需要的NumPy类型
        import numpy as np
        add_safe_globals([np.ndarray, np.generic])
        
        logger.info("已添加NumPy类型到安全全局列表")
    except (ImportError, ModuleNotFoundError) as e:
        logger.warning(f"无法导入NumPy类型进行安全加载: {e}")
    
    if os.path.exists(checkpoint_dir):
        for filename in os.listdir(checkpoint_dir):
            if filename.endswith('.pth'):
                model_path = os.path.join(checkpoint_dir, filename)
                try:
                    # 使用weights_only=False（较不安全但兼容性好）
                    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                    # 提取模型信息
                    timestamp = checkpoint.get('timestamp', filename.replace('model_', '').replace('.pth', ''))
                    models.append({
                        'filename': filename,
                        'path': model_path,
                        'timestamp': timestamp,
                        'display_name': f"模型 {timestamp}"
                    })
                except Exception as e:
                    logger.warning(f"无法加载模型 {filename}: {str(e)}", exc_info=True)
                    continue
    
    return sorted(models, key=lambda x: x['timestamp'], reverse=True)

# 添加一个函数来使NumPy数组和其他类型可JSON序列化
def make_json_serializable(obj):
    """递归地将NumPy数组和其他不可序列化对象转换为JSON可序列化的类型"""
    if isinstance(obj, (np.ndarray, np.number)):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif hasattr(obj, 'isoformat'):  # 处理datetime对象
        return obj.isoformat()
    elif hasattr(obj, '__dict__'):  # 处理自定义类
        return make_json_serializable(obj.__dict__)
    else:
        # 对于无法转换的对象，尝试转为字符串
        try:
            return str(obj)
        except:
            return None

# 路由
@app.route('/')
def index():
    """首页 - 系统概览"""
    return render_template('index.html')

@app.route('/train')
def train():
    """训练页面路由"""
    # 先将DATA_CONFIG中的训练参数渲染到页面
    return render_template('train.html', 
                           training_stats=training_stats,
                           default_params={
                               'batch_size': DATA_CONFIG['batch_size'],
                               'd_model': MODEL_CONFIG['d_model'],
                               'nhead': MODEL_CONFIG['nhead'],
                               'num_encoder_layers': MODEL_CONFIG['num_encoder_layers'],
                               'epochs': TRAIN_CONFIG['epochs'],
                               'learning_rate': TRAIN_CONFIG['learning_rate']
                           })

@app.route('/start_training', methods=['GET', 'POST'])
def start_training():
    """启动模型训练（API路由）"""
    if training_stats['is_training']:
        return jsonify({'status': 'error', 'message': '训练已在进行中'})
    
    # 获取表单数据（训练的参数）
    if request.method == 'POST':
        # 数据通过 request.form 获取，这是表单数据
        data_type = request.form.get('data_type', 'strain')
        custom_params = {
            'batch_size': request.form.get('batch_size'),
            'd_model': request.form.get('d_model'),
            'nhead': request.form.get('nhead'),
            'num_encoder_layers': request.form.get('num_encoder_layers'),
            'epochs': request.form.get('epochs'),
            'learning_rate': request.form.get('learning_rate')
        }
    else:  # GET方法
        # 数据通过 request.args 获取，这是 URL 查询参数
        data_type = request.args.get('data_type', 'strain')
        custom_params = {
            'batch_size': request.args.get('batch_size'),
            'd_model': request.args.get('d_model'),
            'nhead': request.args.get('nhead'),
            'num_encoder_layers': request.args.get('num_encoder_layers'),
            'epochs': request.args.get('epochs'),
            'learning_rate': request.args.get('learning_rate')
        }
    
    # 清除训练状态
    training_stats['train_loss'] = []
    training_stats['test_loss'] = []
    training_stats['current_epoch'] = 0
    training_stats['current_batch'] = 0
    
    # 启动训练线程
    threading.Thread(target=start_training_thread, args=(data_type, custom_params)).start()
    
    return jsonify({'status': 'success', 'message': '训练已启动'})

@app.route('/training_status')
def training_status():
    """获取训练状态"""
    return jsonify(training_stats)

@app.route('/predict')
def predict():
    """预测页面"""
    # 获取可用模型
    models = get_available_models()
    
    return render_template('predict.html', 
                          prediction_stats=prediction_stats,
                          models=models)

@app.route('/start_prediction', methods=['GET', 'POST'])
def start_prediction():
    """启动模型预测"""
    if prediction_stats['is_predicting']:
        return jsonify({'status': 'error', 'message': '预测已在进行中'})
    
    # 获取表单数据
    if request.method == 'POST':
        data_type = request.form.get('data_type', 'strain')
        model_path = request.form.get('model_path')
    else:  # GET方法
        data_type = request.args.get('data_type', 'strain')
        model_path = request.args.get('model_path')
    
    if not model_path or not os.path.exists(model_path):
        return jsonify({'status': 'error', 'message': '请选择有效的模型'})
    
    # 清除预测状态
    prediction_stats['results'] = None
    prediction_stats['material_stats'] = {}
    
    # 启动预测线程
    threading.Thread(target=start_prediction_thread, args=(data_type, model_path)).start()
    
    return jsonify({'status': 'success', 'message': '预测已启动'})

@app.route('/prediction_status')
def prediction_status():
    """获取预测状态"""
    return jsonify(prediction_stats)

@app.route('/dashboard')
def dashboard():
    """仪表板页面"""
    now = datetime.now()
    
    # 确保prediction_stats中包含必要的字段，避免模板中的错误
    if 'material_stats' not in prediction_stats:
        prediction_stats['material_stats'] = {}
        
    if 'results' not in prediction_stats:
        prediction_stats['results'] = {
            'mse': 0.0,
            'mae': 0.0,
            'r2': 0.0,
            'rmse': 0.0,
            'sample_count': 0,
            'targets': [],
            'predictions': []
        }
    
    # 确保每个材料的统计信息中包含targets和predictions
    for material, stats in prediction_stats['material_stats'].items():
        if 'targets' not in stats:
            stats['targets'] = []
        if 'predictions' not in stats:
            stats['predictions'] = []
        # 确保有sample_count字段
        if 'sample_count' not in stats:
            stats['sample_count'] = 0
    
    # 计算总样本数
    total_samples = 0
    if prediction_stats['results'] and 'targets' in prediction_stats['results']:
        total_samples = len(prediction_stats['results']['targets'])
    else:
        # 如果results中没有targets，从材料统计中计算
        for material, stats in prediction_stats['material_stats'].items():
            total_samples += stats.get('sample_count', 0)
    
    # 记录样本数量日志
    logger.info(f"仪表板页面加载 - 总样本数: {total_samples}")
    
    return render_template('dashboard.html', prediction_stats=prediction_stats, now=now, total_samples=total_samples)

@app.route('/model_comparison')
def model_comparison():
    """模型对比页面"""
    # 读取模型对比结果文件
    comparison_results_path = os.path.join(app.config['RESULTS_FOLDER'], 'model_comparison_results.json')
    print(comparison_results_path)
    comparison_results = {}
    comparison_params = {}
    detailed_results = {}
    epochs = 0
    
    if os.path.exists(comparison_results_path):
        try:
            with open(comparison_results_path, 'r') as f:
                comparison_data = json.load(f)
                comparison_results = comparison_data.get('results', {})
                comparison_params = comparison_data.get('params', {})
                detailed_results = comparison_data.get('detailed_results', {})
                epochs = comparison_data.get('epochs', 0)
                
                # 确保结果数据格式正确，修复字段名称不匹配问题
                for model_name, model_data in detailed_results.items():
                    # 检查并重命名字段，确保前端期望的字段名存在
                    if 'train_losses' in model_data and 'train_loss' not in model_data:
                        model_data['train_loss'] = model_data['train_losses']
                    if 'test_losses' in model_data and 'test_loss' not in model_data:
                        model_data['test_loss'] = model_data['test_losses']
                    if 'r2_scores' in model_data and 'r2' not in model_data:
                        model_data['r2'] = model_data['r2_scores']
                        
                logger.info(f"成功加载模型对比数据: {len(comparison_results)} 个模型, {epochs} 轮")
        except Exception as e:
            logger.error(f"读取模型对比结果失败: {e}")
    
    return render_template('model_comparison.html', 
                          comparison_results=comparison_results,
                          comparison_params=comparison_params,
                          detailed_results=detailed_results,
                          epochs=epochs)

@app.route('/generate_model_comparison', methods=['POST'])
def generate_model_comparison():
    """生成模型对比"""
    # 检查是否已经有对比生成任务在执行
    if getattr(app, 'is_generating_comparison', False):
        return jsonify({'success': False, 'message': '已有模型对比正在生成中'})
    
    # 设置生成标志
    app.is_generating_comparison = True
    
    try:
        # 在子进程中执行模型对比脚本
        import subprocess
        
        # 获取model_comparison.py的路径
        script_path = os.path.join(os.path.dirname(os.path.dirname(app.root_path)), 'model_comparison.py')
        
        if not os.path.exists(script_path):
            logger.error(f"模型对比脚本不存在: {script_path}")
            return jsonify({'success': False, 'message': '模型对比脚本不存在'})
        
        # 执行脚本
        logger.info(f"开始执行模型对比脚本: {script_path}")
        result = subprocess.run(['python', script_path], 
                               capture_output=True, 
                               text=True)
        
        # 打印脚本输出以便调试
        logger.info("模型对比脚本标准输出:")
        logger.info(result.stdout)
        logger.info("模型对比脚本标准错误:")
        logger.info(result.stderr)
        
        if result.returncode != 0:
            logger.error(f"模型对比脚本执行失败: {result.stderr}")
            return jsonify({'success': False, 'message': f'模型对比脚本执行失败: {result.stderr}'})
        
        # 读取JSON结果文件
        json_file_paths = [
            'model_comparison_results.json',  # 项目根目录
            os.path.join(os.path.dirname(os.path.dirname(app.root_path)), 'model_comparison_results.json'),  # 与脚本同目录
            os.path.join(app.config['RESULTS_FOLDER'], 'model_comparison_results.json')  # 指定的结果目录
        ]
        
        comparison_data = None
        for file_path in json_file_paths:
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r') as f:
                        comparison_data = json.load(f)
                    logger.info(f"成功从 {file_path} 读取比较结果")
                    break
                except Exception as e:
                    logger.error(f"读取JSON文件 {file_path} 失败: {e}")
        
        if comparison_data is None:
            logger.error("无法找到或读取模型比较结果JSON文件")
            # 尝试解析脚本输出作为备选方案
            import re
            final_results = {}
            params_count = {}
            
            # 查找各模型参数数量
            params_info = re.findall(r'(CNN|LSTM|Transformer)模型参数数量[:：]?\s*([\d,]+)', result.stdout)
            for model, params in params_info:
                params_count[model] = params
                logger.info(f"从输出中提取到{model}参数数量: {params}")
            
            # 查找各模型性能指标
            for model in ['CNN', 'LSTM', 'Transformer']:
                pattern = rf'{model}:.*?训练损失[:：]?\s*([\d\.]+).*?测试损失[:：]?\s*([\d\.]+).*?R²分数[:：]?\s*([\d\.]+).*?训练时间[:：]?\s*([\d\.]+)'
                match = re.search(pattern, result.stdout, re.DOTALL)
                if match:
                    final_results[model] = {
                        'final_train_loss': float(match.group(1)),
                        'final_test_loss': float(match.group(2)),
                        'final_r2': float(match.group(3)),
                        'training_time': float(match.group(4))
                    }
                    logger.info(f"从输出中提取到{model}性能指标: {final_results[model]}")
            
            # 当前时间戳
            current_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # 构造基本结果数据
            comparison_data = {
                'results': final_results,
                'params': params_count,
                'timestamp': current_timestamp
            }

        else:
            # 确保比较数据包含所有必要的字段
            if 'detailed_results' in comparison_data:
                for model_name, model_data in comparison_data['detailed_results'].items():
                    # 添加前端期望的字段名（如果不存在）
                    if 'train_losses' in model_data and 'train_loss' not in model_data:
                        model_data['train_loss'] = model_data['train_losses']
                    if 'test_losses' in model_data and 'test_loss' not in model_data:
                        model_data['test_loss'] = model_data['test_losses']
                    if 'r2_scores' in model_data and 'r2' not in model_data:
                        model_data['r2'] = model_data['r2_scores']
            
            # 确保数据包含epochs字段
            if 'epochs' not in comparison_data and 'detailed_results' in comparison_data:
                # 从详细结果中推断epochs数
                for model_name, model_data in comparison_data['detailed_results'].items():
                    if 'train_losses' in model_data:
                        comparison_data['epochs'] = len(model_data['train_losses'])
                        break
        
        # 确保结果目录存在
        os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
        
        # 保存结果副本到指定目录
        results_path = os.path.join(app.config['RESULTS_FOLDER'], 'model_comparison_results.json')
        with open(results_path, 'w') as f:
            json.dump(comparison_data, f, indent=4)
        
        logger.info(f"模型对比结果已保存到: {results_path}")
        
        # 提取要返回的数据
        final_results = comparison_data.get('results', {})
        detailed_results = comparison_data.get('detailed_results', {})
        epochs = comparison_data.get('epochs', 0)
        params_count = comparison_data.get('params', {})
        current_timestamp = comparison_data.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        # 返回给前端的响应包含数据
        return jsonify({
            'success': True, 
            'message': '模型对比已成功生成',
            'results': final_results,
            'detailed_results': detailed_results,
            'epochs': epochs,
            'params': params_count,
            'timestamp': current_timestamp
        })
        
    except Exception as e:
        logger.error(f"生成模型对比时发生错误: {e}", exc_info=True)
        return jsonify({'success': False, 'message': f'生成模型对比失败: {str(e)}'})
    finally:
        # 重置生成标志
        app.is_generating_comparison = False

@app.route('/upload_dataset', methods=['GET', 'POST'])
def upload_dataset():
    """上传数据集"""
    if request.method == 'GET':
        return jsonify({
            'status': 'info', 
            'message': '请使用POST方法上传数据集文件',
            'required_files': ['strain_data', 'strain_summary'],
            'allowed_extensions': ['csv', 'zip']
        })
    
    # 检查是否有文件上传
    if 'strain_data' not in request.files and 'strain_summary' not in request.files:
        flash('没有选择文件', 'error')
        return redirect(request.url)
        
    strain_data = request.files.get('strain_data')
    strain_summary = request.files.get('strain_summary')
    
    # 检查文件是否为空
    if (strain_data and strain_data.filename == '') or (strain_summary and strain_summary.filename == ''):
        flash('没有选择文件', 'error')
        return redirect(request.url)
    
    # 处理ZIP文件
    if strain_data and allowed_file(strain_data.filename) and strain_data.filename.endswith('.zip'):
        try:
            # 保存ZIP文件到临时位置
            zip_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(strain_data.filename))
            strain_data.save(zip_path)
            logger.info(f"ZIP文件已保存到: {zip_path}")
            
            # 创建目标目录
            target_dir = os.path.join(app.config['DATA_FOLDER'], 'All data_Strain')
            os.makedirs(target_dir, exist_ok=True)
            
            # 解压文件
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(target_dir)
            logger.info(f"ZIP文件已解压到: {target_dir}")
            
            # 清理ZIP文件
            os.remove(zip_path)
            logger.info(f"临时ZIP文件已删除: {zip_path}")
            
        except Exception as e:
            logger.error(f"解压ZIP文件失败: {str(e)}")
            flash(f'解压文件失败: {str(e)}', 'error')
            return redirect(request.url)
    
    if strain_summary and allowed_file(strain_summary.filename):
        filename = secure_filename(strain_summary.filename)
        summary_path = os.path.join(app.config['DATA_FOLDER'], filename)
        strain_summary.save(summary_path)
        logger.info(f"保存汇总文件到 {summary_path}")
        
        # 如果文件名不是预期的，则重命名
        expected_name = 'data_all_strain-controlled.csv'
        if filename != expected_name:
            os.rename(summary_path, os.path.join(app.config['DATA_FOLDER'], expected_name))
            logger.info(f"重命名汇总文件为 {expected_name}")
    
    flash('数据集上传成功', 'success')
    return redirect(url_for('train'))

# 启动应用
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 