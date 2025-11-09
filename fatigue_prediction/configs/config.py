"""
Global Configuration File - Contains data paths, model parameters and training parameters
"""

import os

# Get current project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Data-related configuration
DATA_CONFIG = {
    'dataset_dir': os.path.join(PROJECT_ROOT, 'dataset'),  # Dataset root directory
    'strain_controlled_dir': os.path.join(PROJECT_ROOT, 'dataset', 'All data_Strain'),  # Strain-controlled data directory
    'stress_controlled_dir': os.path.join(PROJECT_ROOT, 'dataset', 'All data_Stress'),  # Stress-controlled data directory
    'strain_summary_file': os.path.join(PROJECT_ROOT, 'dataset', 'data_all_strain-controlled.csv'),  # Strain-controlled summary file
    'stress_summary_file': os.path.join(PROJECT_ROOT, 'dataset', 'data_all_stress-controlled.csv'),  # Stress-controlled summary file
    'sequence_length': 241,  # Standard time series data length
    'test_size': 0.2,  # Test set ratio
    'random_seed': 42,  # Random seed
    'batch_size': 32,  # Batch size
    'num_workers': 4,  # Number of data loading threads
}

# Transformer model configuration
MODEL_CONFIG = {
    # Input feature dimensions
    'time_series_dim': 2,  # Loading path time series data dimension
    'material_feature_dim': 4,  # Material feature dimension (Elastic modulus, Tensile strength, Yield strength, Poisson's ratio)
    
    # Transformer parameters
    'd_model': 128,  # Model dimension
    'nhead': 4,  # Number of attention heads
    'num_encoder_layers': 3,  # Number of encoder layers
    'dim_feedforward': 512,  # Feedforward network dimension
    'dropout': 0.1,  # Dropout rate
    'activation': 'gelu',  # Activation function
    
    # Positional encoding
    'max_seq_length': 241,  # Maximum sequence length
    
    # Output parameters
    'output_dim': 1,  # Output dimension (fatigue life)
}

# Training configuration
TRAIN_CONFIG = {
    'epochs': 100,  # Number of training epochs
    'learning_rate': 5e-4,  # Initial learning rate
    'weight_decay': 2e-5,  # Weight decay
    'lr_scheduler_factor': 0.3,  # Learning rate decay factor
    'lr_scheduler_patience': 10,  # Learning rate adjustment patience
    'early_stop_patience': 30,  # Early stopping patience
    'gradient_clip_val': 1.0,  # Gradient clipping value
    'save_dir': os.path.join(PROJECT_ROOT, 'fatigue_prediction', 'checkpoints'),  # Model save directory
    'logging_dir': os.path.join(PROJECT_ROOT, 'fatigue_prediction', 'logs'),  # Logging directory
}

# Evaluation metrics configuration
METRICS_CONFIG = {
    'metrics': ['mse', 'mae', 'r2', 'rmse'],  # List of evaluation metrics
}

# System configuration
SYSTEM_CONFIG = {
    'use_gpu': True,  # Whether to use GPU
    'precision': 32,  # Computation precision (16, 32)
} 