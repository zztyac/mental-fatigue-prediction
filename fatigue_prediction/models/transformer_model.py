"""
Transformer Model Module - Implements fatigue life prediction model based on Transformer architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PositionalEncoding(nn.Module):
    """
    Positional encoding module that adds position information to time series data
    
    Implements standard Transformer positional encoding using sine and cosine functions
    """
    
    def __init__(self, d_model, max_seq_length=241, dropout=0.1):
        """
        Initialize positional encoding
        
        Args:
            d_model: Model dimension
            max_seq_length: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        # Use sine and cosine functions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension
        pe = pe.unsqueeze(0)
        
        # Register as buffer (not a model parameter)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Forward propagation, add positional encoding
        
        Args:
            x: Input tensor, shape [batch_size, seq_length, d_model]
            
        Returns:
            Tensor with positional encoding
        """
        # Add positional encoding
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TimeSeriesEncoder(nn.Module):
    """
    Time series encoder that processes loading path time series data
    
    Uses Transformer encoder to process time series data
    """
    
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward, dropout):
        """
        Initialize time series encoder
        
        Args:
            input_dim: Input feature dimension
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of encoder layers
            dim_feedforward: Feedforward network dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        # Feature projection layer, maps input dimension to model dimension  2->128
        self.feature_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            d_model=d_model,
            dropout=dropout
        )
        
        # Transformer encoder layers (3 layers)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        # Transformer encoder
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )
        
    def forward(self, x, mask=None):
        """
        Forward propagation
        
        Args:
            x: Input time series data, shape [batch_size, seq_length, input_dim]
            mask: Attention mask, shape [batch_size, seq_length]
            
        Returns:
            Encoded time series features
        """
        # Feature projection
        x = self.feature_projection(x)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Create attention mask (if provided)
        if mask is not None:
            # Convert to mask format required by Transformer
            # key_padding_mask should be a 2D tensor of shape [batch_size, seq_length]
            # where True indicates positions to be masked
            transformer_mask = mask.eq(0)
        else:
            transformer_mask = None
        
        # Transformer encoding
        encoded = self.transformer_encoder(x, src_key_padding_mask=transformer_mask)
        
        return encoded


class MaterialFeatureEncoder(nn.Module):
    """
    Material feature encoder that processes material property data
    """
    
    def __init__(self, input_dim, d_model, dropout=0.1):
        """
        Initialize material feature encoder
        
        Args:
            input_dim: Input feature dimension (number of material properties)
            d_model: Model dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        # Use multi-layer perceptron
        self.fc1 = nn.Linear(input_dim, d_model * 2)  # 4 --> 256
        self.fc2 = nn.Linear(d_model * 2, d_model)  # 256 --> 128
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        """
        Forward propagation
        
        Args:
            x: Input material features, shape [batch_size, input_dim]
            
        Returns:
            Encoded material features
        """
        # First fully connected layer
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        
        # Second fully connected layer
        x = self.fc2(x)
        x = self.norm(x)
        
        return x


class CrossAttention(nn.Module):
    """
    Cross-attention module that fuses time series features with material features
    """
    
    def __init__(self, d_model, nhead, dropout=0.1):
        """
        Initialize cross-attention module
        
        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        # Multi-head attention module
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization after residual connection
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, time_series_features, material_features, mask=None):
        """
        Forward propagation
        
        Args:
            time_series_features: Time series features, shape [batch_size, seq_length, d_model]
            material_features: Material features, shape [batch_size, d_model]
            mask: Attention mask, shape [batch_size, seq_length]
            
        Returns:
            Fused features
        """
        # Expand material features to sequence form to match time series feature sequence length
        batch_size, seq_length, d_model = time_series_features.shape
        
        # Expand material features, shape: [batch_size, 1, d_model] -> [batch_size, seq_length, d_model]
        expanded_material_features = material_features.unsqueeze(1).expand(-1, seq_length, -1)
        
        # Create attention mask (if provided)
        attention_mask = None
        if mask is not None:
            # Convert to mask format required by MultiheadAttention
            # key_padding_mask should be a 2D tensor of shape [batch_size, seq_length]
            # where True indicates positions to be masked
            attention_mask = mask.eq(0)
        
        # Cross-attention: use material features as query, time series features as key and value
        attn_output, attn_weights = self.multihead_attn(
            query=expanded_material_features,
            key=time_series_features,
            value=time_series_features,
            key_padding_mask=attention_mask
        )
        
        # Residual connection
        attn_output = expanded_material_features + self.dropout(attn_output)
        attn_output = self.norm(attn_output)
        
        return attn_output, attn_weights


class FatigueTransformer(nn.Module):
    """
    Fatigue life prediction Transformer model
    
    Implements fatigue life prediction model based on Transformer architecture, 
    integrating time series features and material features
    """
    
    def __init__(self, config):
        """
        Initialize model
        
        Args:
            config: Model configuration dictionary
        """
        super().__init__()
        
        # Extract parameters from config
        self.time_series_dim = config['time_series_dim']
        self.material_feature_dim = config['material_feature_dim']
        self.d_model = config['d_model']  # Model dimension 128
        self.nhead = config['nhead']  # Number of attention heads 4
        self.num_encoder_layers = config['num_encoder_layers']  # Number of encoder layers 3
        self.dim_feedforward = config['dim_feedforward']  # Feedforward network dimension 512
        self.dropout = config['dropout']
        self.output_dim = config['output_dim']  # 1
        
        # Time series encoder
        self.time_series_encoder = TimeSeriesEncoder(
            input_dim=self.time_series_dim,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_encoder_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout
        )
        
        # Material feature encoder
        self.material_encoder = MaterialFeatureEncoder(
            input_dim=self.material_feature_dim,
            d_model=self.d_model,
            dropout=self.dropout
        )
        
        # Cross-attention module
        self.cross_attention = CrossAttention(
            d_model=self.d_model,
            nhead=self.nhead,
            dropout=self.dropout
        )
        
        # Prediction head (regressor)
        self.regression_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2), # 128 --> 64
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model // 2, self.output_dim)  # 64 --> 1
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
    def forward(self, time_series, material_feature, attention_mask=None):
        """
        Forward propagation
        
        Args:
            time_series: Time series data, shape [batch_size, seq_length, time_series_dim]
            material_feature: Material features, shape [batch_size, material_feature_dim]
            attention_mask: Attention mask, shape [batch_size, seq_length]
            
        Returns:
            predicted_life: Predicted fatigue life
            attention_weights: Attention weights
        """
        # Encode time series data
        time_series_encoded = self.time_series_encoder(time_series, attention_mask)
        
        # Encode material features
        material_encoded = self.material_encoder(material_feature)
        
        # Cross-attention fusion
        fused_features, attention_weights = self.cross_attention(
            time_series_encoded, material_encoded, attention_mask
        )
        
        # Global pooling: take sequence average
        # If mask exists, only consider valid parts
        if attention_mask is not None:
            # Create expanded mask, shape [batch_size, seq_length, 1]
            expanded_mask = attention_mask.unsqueeze(-1)
            # Apply mask and compute mean
            masked_features = fused_features * expanded_mask
            # Calculate number of valid elements (number of 1s in mask)
            seq_lengths = attention_mask.sum(dim=1, keepdim=True)
            # Prevent division by zero
            seq_lengths = torch.clamp(seq_lengths, min=1.0)
            # Compute mean
            pooled_features = masked_features.sum(dim=1) / seq_lengths
        else:
            # If no mask, directly compute mean
            pooled_features = torch.mean(fused_features, dim=1)
        
        # Predict fatigue life
        predicted_life = self.regression_head(pooled_features)
        
        return predicted_life, attention_weights
    
    def get_attention_maps(self, time_series, material_feature, attention_mask=None):
        """
        Get attention weight maps for visualization analysis
        
        Args:
            time_series: Time series data
            material_feature: Material features
            attention_mask: Attention mask
            
        Returns:
            Attention weight maps
        """
        self.eval()  # Set to evaluation mode
        with torch.no_grad():
            _, attention_weights = self.forward(time_series, material_feature, attention_mask)
        return attention_weights


def test_transformer_model(config):
    """Test Transformer model"""
    import os
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from data.data_loader import FatigueDataLoader
    from configs.config import DATA_CONFIG
    
    # Create model
    model = FatigueTransformer(config)

    # Load real data
    data_type = 'strain'  # Use strain-controlled data
    
    # Initialize data loader
    data_loader = FatigueDataLoader(DATA_CONFIG)
    
    # Load data
    _, _, test_loader, _ = data_loader.prepare_dataloaders(data_type)
    
    # Get a small batch of data for testing
    for batch in test_loader:
        time_series = batch['time_series']
        material_feature = batch['material_feature']
        target = batch['target']
        attention_mask = batch['attention_mask']

        
        logger.info(f"Loaded test data shapes: time series {time_series.shape}, material features {material_feature.shape}, target {target.shape}")
        print(time_series)
        print(material_feature)
        print(target)


        
        # # Forward propagation test
        # predicted_life, attention_weights = model(time_series, material_feature, attention_mask)
        
        # logger.info(f"Prediction result shape: {predicted_life.shape}")
        
        # # Print actual and predicted values for a few samples for comparison
        # for i in range(min(5, len(target))):
        #     logger.info(f"Sample {i+1} - Actual: {target[i].item():.4f}, Predicted: {predicted_life[i].item():.4f}")
        
        # # Only test one batch
        break
    
    logger.info("Transformer model testing completed")
    return model


if __name__ == "__main__":
    # Add project root directory to system path
    import os
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from configs.config import MODEL_CONFIG
    test_transformer_model(MODEL_CONFIG)
