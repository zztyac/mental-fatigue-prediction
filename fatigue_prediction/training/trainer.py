"""
训练器模块 - 负责模型训练、验证和测试
"""

import os
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FatigueTrainer:
    """
    疲劳寿命预测模型训练器，处理训练、验证和测试过程
    """
    
    def __init__(self, model, train_config, device=None):
        """
        初始化训练器
        
        参数:
            model: 待训练的模型
            train_config: 训练配置字典
            device: 训练设备（CPU或GPU）
        """
        self.model = model
        self.config = train_config
        
        # 设置设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() and self.config.get('use_gpu', True) else 'cpu')
        else:
            self.device = device
            
        # 将模型移动到指定设备
        self.model.to(self.device)
        
        # 设置损失函数
        self.criterion = nn.MSELoss()
        
        # 设置优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # 设置学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=self.config['lr_scheduler_factor'],
            patience=self.config['lr_scheduler_patience']
        )
        
        # 设置早停
        self.early_stop_patience = self.config['early_stop_patience']
        self.best_val_loss = float('inf')
        self.best_epoch = -1
        self.epochs_without_improvement = 0
        
        # 设置日志目录
        if 'logging_dir' in self.config:
            os.makedirs(self.config['logging_dir'], exist_ok=True)
            self.writer = SummaryWriter(log_dir=self.config['logging_dir'])
        else:
            self.writer = None
        
        # 回调函数
        self.on_batch_end_callback = None
        self.on_epoch_end_callback = None
        self.on_test_end_callback = None
        self.on_training_end_callback = None
        
        # 日志
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"训练器初始化完成，设备: {self.device}")
        
        # 数据类型 
        self.data_type = None
        
    def set_training_callbacks(self, on_batch_end=None, on_epoch_end=None, on_test_end=None, on_training_end=None):
        """
        设置训练过程的回调函数
        
        参数:
            on_batch_end: 每个批次结束时的回调
            on_epoch_end: 每个轮次结束时的回调
            on_test_end: 测试结束时的回调
            on_training_end: 训练结束时的回调
        """
        self.on_batch_end_callback = on_batch_end
        self.on_epoch_end_callback = on_epoch_end
        self.on_test_end_callback = on_test_end
        self.on_training_end_callback = on_training_end
        
    def train_epoch(self, train_loader, epoch):
        """
        训练一个epoch
        
        参数:
            train_loader: 训练数据加载器
            epoch: 当前epoch
            
        返回:
            平均训练损失
        """
        self.model.train()
        total_loss = 0.0
        
        # 进度记录
        start_time = time.time()
        batch_count = len(train_loader)
        log_interval = max(batch_count // 10, 1)  # 每10%的批次记录一次
        
        # 遍历批次
        for i, batch in enumerate(train_loader):
            # 获取数据
            time_series = batch['time_series'].to(self.device)
            material_feature = batch['material_feature'].to(self.device)
            target = batch['target'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            predicted, _ = self.model(time_series, material_feature, attention_mask)
            
            # 计算损失
            loss = self.criterion(predicted, target)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            if self.config.get('gradient_clip_val', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['gradient_clip_val']
                )
            
            # 更新参数
            self.optimizer.step()
            
            # 累加损失
            total_loss += loss.item()
            
            # 记录进度
            if (i + 1) % log_interval == 0:
                elapsed = time.time() - start_time
                self.logger.info(
                    f"Epoch {epoch+1}, Batch {i+1}/{batch_count}, "
                    f"Loss: {loss.item():.6f}, "
                    f"Time: {elapsed:.2f}s"
                )
            
            # 执行批次结束回调
            if self.on_batch_end_callback:
                self.on_batch_end_callback(epoch, i, batch_count, loss.item())
        
        # 计算平均损失
        avg_loss = total_loss / batch_count
        self.logger.info(f"Epoch {epoch+1} 完成，平均训练损失: {avg_loss:.6f}")
        
        # 执行轮次结束回调
        if self.on_epoch_end_callback:
            self.on_epoch_end_callback(epoch, avg_loss)
        
        return avg_loss
    
    def evaluate(self, test_loader, epoch):
        """
        在测试集上评估模型
        
        参数:
            test_loader: 测试数据加载器
            epoch: 当前epoch
            
        返回:
            平均测试损失和评估指标
        """
        self.model.eval()
        total_loss = 0.0
        all_targets = []
        all_preds = []
        
        with torch.no_grad():
            for batch in test_loader:
                # 获取数据
                time_series = batch['time_series'].to(self.device)
                material_feature = batch['material_feature'].to(self.device)
                target = batch['target'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # 前向传播
                predicted, _ = self.model(time_series, material_feature, attention_mask)
                
                # 计算损失
                loss = self.criterion(predicted, target)
                total_loss += loss.item()
                
                # 收集预测和目标值
                all_targets.extend(target.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
        
        # 计算平均损失和评估指标
        batch_count = len(test_loader)
        avg_loss = total_loss / batch_count
        
        # 转换为NumPy数组
        all_targets = np.array(all_targets)
        all_preds = np.array(all_preds)
        
        # 计算评估指标
        mse = mean_squared_error(all_targets, all_preds)
        mae = mean_absolute_error(all_targets, all_preds)
        r2 = r2_score(all_targets, all_preds)
        rmse = np.sqrt(mse)
        
        # 记录日志
        self.logger.info(
            f"测试结果 - Epoch {epoch+1}, Loss: {avg_loss:.6f}, MSE: {mse:.6f}, "
            f"MAE: {mae:.6f}, R2: {r2:.6f}, RMSE: {rmse:.6f}"
        )
        
        # 记录到TensorBoard
        if self.writer:
            self.writer.add_scalar('Loss/Test', avg_loss, epoch)
            self.writer.add_scalar('Metrics/MSE', mse, epoch)
            self.writer.add_scalar('Metrics/MAE', mae, epoch)
            self.writer.add_scalar('Metrics/R2', r2, epoch)
            self.writer.add_scalar('Metrics/RMSE', rmse, epoch)
        
        # 执行测试结束回调
        if self.on_test_end_callback:
            self.on_test_end_callback(epoch, avg_loss, mse, mae, r2, rmse)
        
        # 返回测试结果
        metrics = {
            'loss': avg_loss,
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'rmse': rmse,
            'predictions': all_preds,
            'targets': all_targets
        }
        
        return avg_loss, metrics
    
    def save_checkpoint(self, epoch, val_loss, metrics, is_best):
        """
        保存检查点
        
        参数:
            epoch: 当前epoch
            val_loss: 验证损失
            metrics: 验证指标
            is_best: 是否是目前最佳模型
        """
        # 创建检查点字典
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'metrics': metrics,
            'config': self.config,
            'data_type': self.data_type  # 在检查点中保存数据类型
        }
        
        # 获取数据类型前缀
        data_type_str = f"_{self.data_type}" if self.data_type else ""
        
        # 保存检查点
        try:
            # 保存最新检查点
            checkpoint_path = os.path.join(self.config['save_dir'], f'checkpoint_epoch_{epoch+1}{data_type_str}.pth')
            torch.save(checkpoint, checkpoint_path)
            self.logger.info(f"保存检查点到 {checkpoint_path}")
            
            # 如果是最佳模型则保存
            if is_best:
                best_model_path = os.path.join(self.config['save_dir'], f'best_model{data_type_str}.pth')
                torch.save(checkpoint, best_model_path)
                self.logger.info(f"保存最佳模型到 {best_model_path}")
                
        except Exception as e:
            self.logger.error(f"保存检查点时出错: {e}")
            raise
    
    def load_checkpoint(self, checkpoint_path):
        """
        加载检查点
        
        参数:
            checkpoint_path: 检查点路径
            
        返回:
            加载的epoch
        """
        self.logger.info(f"从 {checkpoint_path} 加载检查点")
        
        # 定义加载方法列表，按优先级尝试
        load_methods = [
            # 方法1: 标准加载 PyTorch 2.0+
            lambda: torch.load(checkpoint_path, map_location=self.device, weights_only=False),
            # 方法2: 标准加载 PyTorch 1.x
            lambda: torch.load(checkpoint_path, map_location=self.device),
            # 方法3: 使用pickle
            lambda: torch.load(checkpoint_path, map_location=self.device, pickle_module=__import__('pickle')),
            # 方法4: 使用safe_globals
            lambda: self._load_with_safe_globals(checkpoint_path)
        ]
        
        # 尝试所有加载方法
        checkpoint = None
        exception_messages = []
        
        for method_idx, load_method in enumerate(load_methods):
            try:
                checkpoint = load_method()
                self.logger.info(f"使用方法{method_idx+1}成功加载检查点")
                break
            except Exception as e:
                exception_messages.append(f"方法{method_idx+1}失败: {e}")
                continue
        
        if checkpoint is None:
            error_msg = "所有加载方法都失败: " + "; ".join(exception_messages)
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        # 检查检查点数据结构
        required_keys = ['model_state_dict', 'optimizer_state_dict']
        for key in required_keys:
            if key not in checkpoint:
                self.logger.warning(f"检查点中缺少关键字 '{key}'，尝试推断内容...")
                
                # 如果检查点直接是模型状态字典
                if key == 'model_state_dict' and all(k in checkpoint for k in next(self.model.parameters()).keys()):
                    self.logger.info("检查点似乎直接是模型状态字典")
                    checkpoint = {'model_state_dict': checkpoint, 'optimizer_state_dict': self.optimizer.state_dict(), 
                                 'scheduler_state_dict': self.scheduler.state_dict(), 'val_loss': float('inf'), 'epoch': -1}
                    break
        
        # 加载模型状态
        try:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        except Exception as e:
            self.logger.warning(f"加载模型状态失败: {e}，尝试部分加载...")
            # 尝试部分加载
            model_dict = self.model.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)
            self.logger.info(f"成功加载 {len(pretrained_dict)}/{len(model_dict)} 层")
        
        # 加载优化器状态
        if 'optimizer_state_dict' in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception as e:
                self.logger.warning(f"加载优化器状态失败: {e}，使用当前优化器状态")
        
        # 加载调度器状态
        if 'scheduler_state_dict' in checkpoint:
            try:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except Exception as e:
                self.logger.warning(f"加载调度器状态失败: {e}，使用当前调度器状态")
        
        # 恢复其他状态
        if 'val_loss' in checkpoint:
            self.best_val_loss = checkpoint['val_loss']
        
        return checkpoint.get('epoch', -1)
    
    def _load_with_safe_globals(self, checkpoint_path):
        """使用safe_globals加载检查点"""
        import numpy as np
        from torch.serialization import safe_globals
        with safe_globals([np.core.multiarray.scalar]):
            return torch.load(checkpoint_path, map_location=self.device)
    
    def train(self, train_loader, test_loader=None, resume_from=None, data_type=None):
        """
        训练模型
        
        参数:
            train_loader: 训练数据加载器
            test_loader: 测试数据加载器（用于评估）
            resume_from: 从检查点恢复训练（可选）
            data_type: 训练数据类型 ('strain' 或 'stress')
            
        返回:
            训练历史记录
        """
        # 设置数据类型
        self.data_type = data_type
        
        # 记录历史
        history = {
            'train_loss': [],
            'test_loss': [],
            'test_metrics': [],
            'data_type': data_type  # 在历史记录中添加数据类型
        }
        
        # 设置起始epoch
        start_epoch = 0
        
        # 从检查点恢复训练
        if resume_from and os.path.exists(resume_from):
            start_epoch = self.load_checkpoint(resume_from) + 1
            self.logger.info(f"从epoch {start_epoch} 恢复训练")
        
        # 训练循环
        num_epochs = self.config['epochs']
        self.logger.info(f"开始训练，共 {num_epochs} 轮")
        
        for epoch in range(start_epoch, num_epochs):
            # 训练一个epoch
            train_loss = self.train_epoch(train_loader, epoch)
            history['train_loss'].append(train_loss)
            
            # 记录到tensorboard
            if self.writer:
                self.writer.add_scalar('Loss/train', train_loss, epoch)
            
            # 在测试集上评估
            if test_loader is not None:
                test_loss, test_metrics = self.evaluate(test_loader, epoch)
                history['test_loss'].append(test_loss)
                history['test_metrics'].append(test_metrics)
            
                # 更新学习率
                self.scheduler.step(test_loss)
                
                # 检查是否需要保存模型
                if test_loss < self.best_val_loss:
                    self.best_val_loss = test_loss
                    self.best_epoch = epoch
                    self.epochs_without_improvement = 0
                    
                    # 保存模型
                    if 'save_dir' in self.config:
                        os.makedirs(self.config['save_dir'], exist_ok=True)
                        checkpoint_path = os.path.join(self.config['save_dir'], f'best_model{self.data_type}.pth')
                        self.save_checkpoint(epoch, test_loss, test_metrics, True)
                        self.logger.info(f"保存最佳模型至 {checkpoint_path}")
                else:
                    self.epochs_without_improvement += 1
                    self.logger.info(f"测试损失未改善，已经 {self.epochs_without_improvement} 轮")
                    
                    # 早停
                    if self.epochs_without_improvement >= self.early_stop_patience:
                        self.logger.info(f"早停触发，停止训练")
                        break
            
                # 检查是否为最佳模型
                is_best = test_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = test_loss
                    self.best_epoch = epoch
                    self.epochs_without_improvement = 0
                
                # 修改为每10个epoch保存一次检查点或是最佳模型时保存
                if (epoch + 1) % 10 == 0 or is_best:
                    self.save_checkpoint(epoch, test_loss, test_metrics, is_best)
                    if (epoch + 1) % 10 == 0:
                        self.logger.info(f"每10个epoch定期保存检查点，当前epoch: {epoch+1}")
        
        # 加载最佳模型
        best_model_path = os.path.join(self.config['save_dir'], f'best_model{self.data_type}.pth')
        if os.path.exists(best_model_path):
            self.load_checkpoint(best_model_path)
            self.logger.info("加载最佳模型进行最终测试")
        
        # 在测试集上进行最终评估
        if test_loader is not None:
            test_loss, test_metrics = self.evaluate(test_loader, epoch)
            history['final_test_loss'] = test_loss
            history['final_test_metrics'] = test_metrics
            self.logger.info(f"最终测试完成，损失: {test_loss:.6f}")
        
        # 清理
        if self.writer:
            self.writer.close()
        
        # 执行训练结束回调
        if self.on_training_end_callback:
            self.on_training_end_callback(history)
        
        return history

