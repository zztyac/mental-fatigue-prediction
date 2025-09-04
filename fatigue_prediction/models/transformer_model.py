"""
Transformer模型模块 - 实现基于Transformer架构的疲劳寿命预测模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import logging

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PositionalEncoding(nn.Module):
    """
    位置编码模块，为时序数据添加位置信息
    
    实现了标准的Transformer位置编码，使用正弦和余弦函数
    """
    
    def __init__(self, d_model, max_seq_length=241, dropout=0.1):
        """
        初始化位置编码
        
        参数:
            d_model: 模型维度
            max_seq_length: 最大序列长度
            dropout: Dropout率
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        # 使用正弦和余弦函数
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 添加批次维度
        pe = pe.unsqueeze(0)
        
        # 注册为缓冲区（不作为模型参数）
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        前向传播，添加位置编码
        
        参数:
            x: 输入张量，形状 [batch_size, seq_length, d_model]
            
        返回:
            带有位置编码的张量
        """
        # 添加位置编码
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TimeSeriesEncoder(nn.Module):
    """
    时序编码器，处理加载路径时序数据
    
    使用Transformer编码器处理时序数据
    """
    
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward, dropout):
        """
        初始化时序编码器
        
        参数:
            input_dim: 输入特征维度
            d_model: 模型维度
            nhead: 注意力头数
            num_layers: 编码器层数
            dim_feedforward: 前馈网络维度
            dropout: Dropout率
        """
        super().__init__()
        
        # 特征映射层，将输入维度映射到模型维度  2->128
        self.feature_projection = nn.Linear(input_dim, d_model)
        
        # 位置编码
        self.positional_encoding = PositionalEncoding(
            d_model=d_model,
            dropout=dropout
        )
        
        # Transformer编码器层 3层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        # Transformer编码器
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )
        
    def forward(self, x, mask=None):
        """
        前向传播
        
        参数:
            x: 输入时序数据，形状 [batch_size, seq_length, input_dim]
            mask: 注意力掩码，形状 [batch_size, seq_length]
            
        返回:
            编码后的时序特征
        """
        # 特征映射
        x = self.feature_projection(x)
        
        # 添加位置编码
        x = self.positional_encoding(x)
        
        # 创建注意力掩码（如果提供）
        if mask is not None:
            # 转换为Transformer所需的掩码格式
            # key_padding_mask应该是形状为[batch_size, seq_length]的二维张量
            # 其中True表示需要掩码的位置
            transformer_mask = mask.eq(0)
        else:
            transformer_mask = None
        
        # Transformer编码
        encoded = self.transformer_encoder(x, src_key_padding_mask=transformer_mask)
        
        return encoded


class MaterialFeatureEncoder(nn.Module):
    """
    材料特征编码器，处理材料属性数据
    """
    
    def __init__(self, input_dim, d_model, dropout=0.1):
        """
        初始化材料特征编码器
        
        参数:
            input_dim: 输入特征维度（材料属性数量）
            d_model: 模型维度
            dropout: Dropout率
        """
        super().__init__()
        
        # 使用多层感知机
        self.fc1 = nn.Linear(input_dim, d_model * 2)  # 4 --> 256
        self.fc2 = nn.Linear(d_model * 2, d_model)  # 256 --> 128
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入材料特征，形状 [batch_size, input_dim]
            
        返回:
            编码后的材料特征
        """
        # 第一层全连接
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        
        # 第二层全连接
        x = self.fc2(x)
        x = self.norm(x)
        
        return x


class CrossAttention(nn.Module):
    """
    交叉注意力模块，实现时序特征与材料特征的融合
    """
    
    def __init__(self, d_model, nhead, dropout=0.1):
        """
        初始化交叉注意力模块
        
        参数:
            d_model: 模型维度
            nhead: 注意力头数
            dropout: Dropout率
        """
        super().__init__()
        
        # 多头注意力模块
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        
        # 残差连接后的层归一化
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, time_series_features, material_features, mask=None):
        """
        前向传播
        
        参数:
            time_series_features: 时序特征，形状 [batch_size, seq_length, d_model]
            material_features: 材料特征，形状 [batch_size, d_model]
            mask: 注意力掩码，形状 [batch_size, seq_length]
            
        返回:
            融合后的特征
        """
        # 将材料特征扩展为序列形式，匹配时序特征的序列长度
        batch_size, seq_length, d_model = time_series_features.shape
        
        # 扩展材料特征，形状: [batch_size, 1, d_model] -> [batch_size, seq_length, d_model]
        expanded_material_features = material_features.unsqueeze(1).expand(-1, seq_length, -1)
        
        # 创建注意力掩码（如果提供）
        attention_mask = None
        if mask is not None:
            # 转换为MultiheadAttention所需的掩码格式
            # key_padding_mask应该是形状为[batch_size, seq_length]的二维张量
            # 其中True表示需要掩码的位置
            attention_mask = mask.eq(0)
        
        # 交叉注意力：使用材料特征作为查询，时序特征作为键和值
        attn_output, attn_weights = self.multihead_attn(
            query=expanded_material_features,
            key=time_series_features,
            value=time_series_features,
            key_padding_mask=attention_mask
        )
        
        # 残差连接
        attn_output = expanded_material_features + self.dropout(attn_output)
        attn_output = self.norm(attn_output)
        
        return attn_output, attn_weights


class FatigueTransformer(nn.Module):
    """
    疲劳寿命预测Transformer模型
    
    实现基于Transformer架构的疲劳寿命预测模型，整合时序特征和材料特征
    """
    
    def __init__(self, config):
        """
        初始化模型
        
        参数:
            config: 模型配置字典
        """
        super().__init__()
        
        # 从配置中提取参数
        self.time_series_dim = config['time_series_dim']
        self.material_feature_dim = config['material_feature_dim']
        self.d_model = config['d_model']  # 模型维度 128
        self.nhead = config['nhead']  # 注意力头数 4
        self.num_encoder_layers = config['num_encoder_layers']  # 编码器层数 3
        self.dim_feedforward = config['dim_feedforward']  # 前馈网络维度 512
        self.dropout = config['dropout']
        self.output_dim = config['output_dim']  # 1
        
        # 时序编码器
        self.time_series_encoder = TimeSeriesEncoder(
            input_dim=self.time_series_dim,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_encoder_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout
        )
        
        # 材料特征编码器
        self.material_encoder = MaterialFeatureEncoder(
            input_dim=self.material_feature_dim,
            d_model=self.d_model,
            dropout=self.dropout
        )
        
        # 交叉注意力模块
        self.cross_attention = CrossAttention(
            d_model=self.d_model,
            nhead=self.nhead,
            dropout=self.dropout
        )
        
        # 预测头（回归器）
        self.regression_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2), # 128 --> 64
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model // 2, self.output_dim)  # 64 --> 1
        )
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """初始化模型权重"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
    def forward(self, time_series, material_feature, attention_mask=None):
        """
        前向传播
        
        参数:
            time_series: 时序数据，形状 [batch_size, seq_length, time_series_dim]
            material_feature: 材料特征，形状 [batch_size, material_feature_dim]
            attention_mask: 注意力掩码，形状 [batch_size, seq_length]
            
        返回:
            predicted_life: 预测的疲劳寿命
            attention_weights: 注意力权重
        """
        # 编码时序数据
        time_series_encoded = self.time_series_encoder(time_series, attention_mask)
        
        # 编码材料特征
        material_encoded = self.material_encoder(material_feature)
        
        # 交叉注意力融合
        fused_features, attention_weights = self.cross_attention(
            time_series_encoded, material_encoded, attention_mask
        )
        
        # 全局池化：取序列平均值
        # 如果有掩码，只考虑有效部分
        if attention_mask is not None:
            # 创建扩展的掩码，形状为[batch_size, seq_length, 1]
            expanded_mask = attention_mask.unsqueeze(-1)
            # 应用掩码并计算均值
            masked_features = fused_features * expanded_mask
            # 计算有效元素数量（掩码中1的数量）
            seq_lengths = attention_mask.sum(dim=1, keepdim=True)
            # 防止除以零
            seq_lengths = torch.clamp(seq_lengths, min=1.0)
            # 计算均值
            pooled_features = masked_features.sum(dim=1) / seq_lengths
        else:
            # 如果没有掩码，直接计算均值
            pooled_features = torch.mean(fused_features, dim=1)
        
        # 预测疲劳寿命
        predicted_life = self.regression_head(pooled_features)
        
        return predicted_life, attention_weights
    
    def get_attention_maps(self, time_series, material_feature, attention_mask=None):
        """
        获取注意力权重图，用于可视化分析
        
        参数:
            time_series: 时序数据
            material_feature: 材料特征
            attention_mask: 注意力掩码
            
        返回:
            注意力权重图
        """
        self.eval()  # 设置为评估模式
        with torch.no_grad():
            _, attention_weights = self.forward(time_series, material_feature, attention_mask)
        return attention_weights


def test_transformer_model(config):
    """测试Transformer模型"""
    import os
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from data.data_loader import FatigueDataLoader
    from configs.config import DATA_CONFIG
    
    # 创建模型
    model = FatigueTransformer(config)

    # 加载真实数据
    data_type = 'strain'  # 使用应变控制数据
    
    # 初始化数据加载器
    data_loader = FatigueDataLoader(DATA_CONFIG)
    
    # 加载数据
    _, _, test_loader, _ = data_loader.prepare_dataloaders(data_type)
    
    # 获取一个小批次数据进行测试
    for batch in test_loader:
        time_series = batch['time_series']
        material_feature = batch['material_feature']
        target = batch['target']
        attention_mask = batch['attention_mask']

        
        logger.info(f"加载的测试数据形状: 时间序列 {time_series.shape}, 材料特征 {material_feature.shape}, 目标 {target.shape}")
        print(time_series)
        print(material_feature)
        print(target)


        
        # # 前向传播测试
        # predicted_life, attention_weights = model(time_series, material_feature, attention_mask)
        
        # logger.info(f"预测结果形状: {predicted_life.shape}")
        
        # # 打印几个样本的实际值和预测值进行比较
        # for i in range(min(5, len(target))):
        #     logger.info(f"样本 {i+1} - 实际值: {target[i].item():.4f}, 预测值: {predicted_life[i].item():.4f}")
        
        # # 只测试一个批次
        break
    
    logger.info("Transformer模型测试完成")
    return model


if __name__ == "__main__":
    # 添加项目根目录到系统路径
    import os
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from configs.config import MODEL_CONFIG
    test_transformer_model(MODEL_CONFIG)
