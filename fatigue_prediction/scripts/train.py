"""
Main Training Script - For training models and predicting fatigue life
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

# Add project root directory to system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.transformer_model import FatigueTransformer
from training.trainer import FatigueTrainer
from data.data_loader import FatigueDataLoader
from configs.config import DATA_CONFIG, MODEL_CONFIG, TRAIN_CONFIG

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train fatigue life prediction model')
    
    # Data-related parameters
    parser.add_argument('--data_type', type=str, default='strain', choices=['strain', 'stress'],
                        help='Data type to use: strain (strain-controlled), stress (stress-controlled)')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=None, help='Number of data loading threads')
    
    # Model-related parameters
    parser.add_argument('--d_model', type=int, default=None, help='Model dimension')
    parser.add_argument('--nhead', type=int, default=None, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=None, help='Number of encoder layers')
    
    # Training-related parameters
    parser.add_argument('--epochs', type=int, default=None, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=None, help='Weight decay')
    parser.add_argument('--resume', type=str, default=None, help='Resume training from checkpoint')
    
    # Running mode
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                        help='Running mode: train (training), test (testing)')
    parser.add_argument('--model_path', type=str, default=None, help='Model path for test or prediction mode')

    parser.add_argument('--output_dir', type=str, default='/data/coding/metal_fatigue/results', help='Result save directory')
    
    # GPU-related parameters
    parser.add_argument('--use_gpu', action='store_true', default=None, help='Whether to use GPU')
    
    return parser.parse_args()

def update_config(config, args, config_name):
    """Update configuration using command line arguments"""
    # Get command line arguments dictionary
    args_dict = vars(args)
    
    # Update configuration
    for key, value in args_dict.items():
        if value is not None and key in config:
            logger.info(f"Updating {config_name} configuration: {key} = {value}")
            config[key] = value
    
    return config

def setup_output_dir(args):
    """Setup output directory"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f"{args.data_type}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirectories
    checkpoints_dir = os.path.join(output_dir, 'checkpoints')
    logs_dir = os.path.join(output_dir, 'logs')
    plots_dir = os.path.join(output_dir, 'plots')
    
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # Update training configuration
    TRAIN_CONFIG['save_dir'] = checkpoints_dir
    TRAIN_CONFIG['logging_dir'] = logs_dir
    
    return output_dir, plots_dir

def save_configs(output_dir, data_config, model_config, train_config):
    """Save configurations to file"""
    configs = {
        'data_config': data_config,
        'model_config': model_config,
        'train_config': train_config
    }
    
    config_path = os.path.join(output_dir, 'configs.json')
    with open(config_path, 'w') as f:
        json.dump(configs, f, indent=4)
    
    logger.info(f"Configurations saved to {config_path}")

def train_model(args):
    """Train model"""
    # Update configurations
    data_config = update_config(DATA_CONFIG.copy(), args, 'DATA')
    model_config = update_config(MODEL_CONFIG.copy(), args, 'MODEL')
    train_config = update_config(TRAIN_CONFIG.copy(), args, 'TRAIN')
    
    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f'{args.data_type}_train_{timestamp}')
    checkpoints_dir = os.path.join(output_dir, 'checkpoints')
    plots_dir = os.path.join(output_dir, 'plots')
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # Save configurations
    save_configs(output_dir, data_config, model_config, train_config)
    
    # Load data
    logger.info(f"Loading {args.data_type} data...")
    data_loader = FatigueDataLoader(data_config)
    
    train_loader, test_loader, file_names = data_loader.prepare_dataloaders(args.data_type)
    
    # Create model
    logger.info("Creating model...")
    model = FatigueTransformer(model_config)
    
    # Print model information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total model parameters: {total_params}, trainable parameters: {trainable_params}")
    
    # Initialize trainer
    logger.info("Initializing trainer...")
    trainer = FatigueTrainer(model, train_config)
    
    # Train model
    logger.info("Starting model training...")
    history = trainer.train(
        train_loader=train_loader,
        test_loader=test_loader,
        resume_from=args.resume,
        data_type=args.data_type
    )
    
    # Plot training history
    logger.info("Plotting training history...")
    plot_training_history(history, plots_dir)
    
    # Plot prediction results
    logger.info("Plotting prediction results...")
    if 'test_metrics' in history:
        plot_predictions(history['test_metrics'], plots_dir)
    
    # Save training results and model
    model_save_path = os.path.join(checkpoints_dir, f'model_{timestamp}.pth')
    logger.info(f"Saving model to: {model_save_path}")
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'history': history,
        'model_config': model_config,
        'timestamp': timestamp
    }, model_save_path)
    
    logger.info(f"Training completed! Results saved in {output_dir}")
    
    return model, history, output_dir

def test_model(args):
    """Test model"""
    # Load data
    logger.info(f"Loading {args.data_type} data...")
    data_loader = FatigueDataLoader(DATA_CONFIG)
    
    _, _, test_loader, file_names = data_loader.prepare_dataloaders(args.data_type)
    
    # Load model
    logger.info(f"Loading model {args.model_path}...")
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Create model
    model = FatigueTransformer(MODEL_CONFIG)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Create trainer
    trainer = FatigueTrainer(model, TRAIN_CONFIG, device=device)
    
    # Test model
    logger.info("Starting model testing...")
    test_loss, test_metrics = trainer.test(test_loader)
    
    # Setup output directory
    output_dir = os.path.join(args.output_dir, 'test_results')
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot prediction results
    logger.info("Plotting prediction results...")
    plot_predictions(test_metrics, plots_dir)
    
    # Save test results
    result_path = os.path.join(output_dir, 'test_results.json')
    with open(result_path, 'w') as f:
        # Remove NumPy arrays and other non-serializable objects
        serializable_metrics = {k: v for k, v in test_metrics.items() 
                               if k not in ['predictions', 'targets']}
        json.dump(serializable_metrics, f, indent=4)
    
    logger.info(f"Testing completed! Results saved in {output_dir}")
    
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
    """Main function"""
    # Parse command line arguments
    args = parse_args()
    
    # Execute corresponding operations based on running mode
    if args.mode == 'train':
        train_model(args)
    elif args.mode == 'test':
        if args.model_path is None:
            logger.error("Test mode requires model path to be specified (--model_path)")
            return
        test_model(args)

if __name__ == "__main__":
    main() 