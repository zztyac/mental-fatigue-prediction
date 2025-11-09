"""
Trainer Module - Responsible for model training, validation and testing
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

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FatigueTrainer:
    """
    Fatigue life prediction model trainer that handles training, validation and testing processes
    """
    
    def __init__(self, model, train_config, device=None):
        """
        Initialize trainer
        
        Args:
            model: Model to be trained
            train_config: Training configuration dictionary
            device: Training device (CPU or GPU)
        """
        self.model = model
        self.config = train_config
        
        # Setup device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() and self.config.get('use_gpu', True) else 'cpu')
        else:
            self.device = device
            
        # Move model to specified device
        self.model.to(self.device)
        
        # Setup loss function
        self.criterion = nn.MSELoss()
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # Setup learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=self.config['lr_scheduler_factor'],
            patience=self.config['lr_scheduler_patience']
        )
        
        # Setup early stopping
        self.early_stop_patience = self.config['early_stop_patience']
        self.best_val_loss = float('inf')
        self.best_epoch = -1
        self.epochs_without_improvement = 0
        
        # Setup logging directory
        if 'logging_dir' in self.config:
            os.makedirs(self.config['logging_dir'], exist_ok=True)
            self.writer = SummaryWriter(log_dir=self.config['logging_dir'])
        else:
            self.writer = None
        
        # Callback functions
        self.on_batch_end_callback = None
        self.on_epoch_end_callback = None
        self.on_test_end_callback = None
        self.on_training_end_callback = None
        
        # Logger
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Trainer initialized, device: {self.device}")
        
        # Data type 
        self.data_type = None
        
    def set_training_callbacks(self, on_batch_end=None, on_epoch_end=None, on_test_end=None, on_training_end=None):
        """
        Set callback functions for training process
        
        Args:
            on_batch_end: Callback at the end of each batch
            on_epoch_end: Callback at the end of each epoch
            on_test_end: Callback at the end of testing
            on_training_end: Callback at the end of training
        """
        self.on_batch_end_callback = on_batch_end
        self.on_epoch_end_callback = on_epoch_end
        self.on_test_end_callback = on_test_end
        self.on_training_end_callback = on_training_end
        
    def train_epoch(self, train_loader, epoch):
        """
        Train for one epoch
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        
        # Progress recording
        start_time = time.time()
        batch_count = len(train_loader)
        log_interval = max(batch_count // 10, 1)  # Log every 10% of batches
        
        # Iterate through batches
        for i, batch in enumerate(train_loader):
            # Get data
            time_series = batch['time_series'].to(self.device)
            material_feature = batch['material_feature'].to(self.device)
            target = batch['target'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            # Forward propagation
            self.optimizer.zero_grad()
            predicted, _ = self.model(time_series, material_feature, attention_mask)
            
            # Compute loss
            loss = self.criterion(predicted, target)
            
            # Backward propagation
            loss.backward()
            
            # Gradient clipping
            if self.config.get('gradient_clip_val', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['gradient_clip_val']
                )
            
            # Update parameters
            self.optimizer.step()
            
            # Accumulate loss
            total_loss += loss.item()
            
            # Record progress
            if (i + 1) % log_interval == 0:
                elapsed = time.time() - start_time
                self.logger.info(
                    f"Epoch {epoch+1}, Batch {i+1}/{batch_count}, "
                    f"Loss: {loss.item():.6f}, "
                    f"Time: {elapsed:.2f}s"
                )
            
            # Execute batch end callback
            if self.on_batch_end_callback:
                self.on_batch_end_callback(epoch, i, batch_count, loss.item())
        
        # Calculate average loss
        avg_loss = total_loss / batch_count
        self.logger.info(f"Epoch {epoch+1} completed, average training loss: {avg_loss:.6f}")
        
        # Execute epoch end callback
        if self.on_epoch_end_callback:
            self.on_epoch_end_callback(epoch, avg_loss)
        
        return avg_loss
    
    def evaluate(self, test_loader, epoch):
        """
        Evaluate model on test set
        
        Args:
            test_loader: Test data loader
            epoch: Current epoch
            
        Returns:
            Average test loss and evaluation metrics
        """
        self.model.eval()
        total_loss = 0.0
        all_targets = []
        all_preds = []
        
        with torch.no_grad():
            for batch in test_loader:
                # Get data
                time_series = batch['time_series'].to(self.device)
                material_feature = batch['material_feature'].to(self.device)
                target = batch['target'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Forward propagation
                predicted, _ = self.model(time_series, material_feature, attention_mask)
                
                # Compute loss
                loss = self.criterion(predicted, target)
                total_loss += loss.item()
                
                # Collect predictions and target values
                all_targets.extend(target.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
        
        # Calculate average loss and evaluation metrics
        batch_count = len(test_loader)
        avg_loss = total_loss / batch_count
        
        # Convert to NumPy arrays
        all_targets = np.array(all_targets)
        all_preds = np.array(all_preds)
        
        # Calculate evaluation metrics
        mse = mean_squared_error(all_targets, all_preds)
        mae = mean_absolute_error(all_targets, all_preds)
        r2 = r2_score(all_targets, all_preds)
        rmse = np.sqrt(mse)
        
        # Log results
        self.logger.info(
            f"Test results - Epoch {epoch+1}, Loss: {avg_loss:.6f}, MSE: {mse:.6f}, "
            f"MAE: {mae:.6f}, R2: {r2:.6f}, RMSE: {rmse:.6f}"
        )
        
        # Record to TensorBoard
        if self.writer:
            self.writer.add_scalar('Loss/Test', avg_loss, epoch)
            self.writer.add_scalar('Metrics/MSE', mse, epoch)
            self.writer.add_scalar('Metrics/MAE', mae, epoch)
            self.writer.add_scalar('Metrics/R2', r2, epoch)
            self.writer.add_scalar('Metrics/RMSE', rmse, epoch)
        
        # Execute test end callback
        if self.on_test_end_callback:
            self.on_test_end_callback(epoch, avg_loss, mse, mae, r2, rmse)
        
        # Return test results
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
        Save checkpoint
        
        Args:
            epoch: Current epoch
            val_loss: Validation loss
            metrics: Validation metrics
            is_best: Whether this is the best model so far
        """
        # Create checkpoint dictionary
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'metrics': metrics,
            'config': self.config,
            'data_type': self.data_type  # Save data type in checkpoint
        }
        
        # Get data type prefix
        data_type_str = f"_{self.data_type}" if self.data_type else ""
        
        # Save checkpoint
        try:
            # Save latest checkpoint
            checkpoint_path = os.path.join(self.config['save_dir'], f'checkpoint_epoch_{epoch+1}{data_type_str}.pth')
            torch.save(checkpoint, checkpoint_path)
            self.logger.info(f"Saving checkpoint to {checkpoint_path}")
            
            # Save if it's the best model
            if is_best:
                best_model_path = os.path.join(self.config['save_dir'], f'best_model{data_type_str}.pth')
                torch.save(checkpoint, best_model_path)
                self.logger.info(f"Saving best model to {best_model_path}")
                
        except Exception as e:
            self.logger.error(f"Error saving checkpoint: {e}")
            raise
    
    def load_checkpoint(self, checkpoint_path):
        """
        Load checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint
            
        Returns:
            Loaded epoch
        """
        self.logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        # Define list of loading methods, try in priority order
        load_methods = [
            # Method 1: Standard loading PyTorch 2.0+
            lambda: torch.load(checkpoint_path, map_location=self.device, weights_only=False),
            # Method 2: Standard loading PyTorch 1.x
            lambda: torch.load(checkpoint_path, map_location=self.device),
            # Method 3: Using pickle
            lambda: torch.load(checkpoint_path, map_location=self.device, pickle_module=__import__('pickle')),
            # Method 4: Using safe_globals
            lambda: self._load_with_safe_globals(checkpoint_path)
        ]
        
        # Try all loading methods
        checkpoint = None
        exception_messages = []
        
        for method_idx, load_method in enumerate(load_methods):
            try:
                checkpoint = load_method()
                self.logger.info(f"Successfully loaded checkpoint using method {method_idx+1}")
                break
            except Exception as e:
                exception_messages.append(f"Method {method_idx+1} failed: {e}")
                continue
        
        if checkpoint is None:
            error_msg = "All loading methods failed: " + "; ".join(exception_messages)
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        # Check checkpoint data structure
        required_keys = ['model_state_dict', 'optimizer_state_dict']
        for key in required_keys:
            if key not in checkpoint:
                self.logger.warning(f"Checkpoint missing key '{key}', attempting to infer content...")
                
                # If checkpoint is directly a model state dictionary
                if key == 'model_state_dict' and all(k in checkpoint for k in next(self.model.parameters()).keys()):
                    self.logger.info("Checkpoint appears to be directly a model state dictionary")
                    checkpoint = {'model_state_dict': checkpoint, 'optimizer_state_dict': self.optimizer.state_dict(), 
                                 'scheduler_state_dict': self.scheduler.state_dict(), 'val_loss': float('inf'), 'epoch': -1}
                    break
        
        # Load model state
        try:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        except Exception as e:
            self.logger.warning(f"Failed to load model state: {e}, attempting partial loading...")
            # Try partial loading
            model_dict = self.model.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)
            self.logger.info(f"Successfully loaded {len(pretrained_dict)}/{len(model_dict)} layers")
        
        # Load optimizer state
        if 'optimizer_state_dict' in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception as e:
                self.logger.warning(f"Failed to load optimizer state: {e}, using current optimizer state")
        
        # Load scheduler state
        if 'scheduler_state_dict' in checkpoint:
            try:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except Exception as e:
                self.logger.warning(f"Failed to load scheduler state: {e}, using current scheduler state")
        
        # Restore other states
        if 'val_loss' in checkpoint:
            self.best_val_loss = checkpoint['val_loss']
        
        return checkpoint.get('epoch', -1)
    
    def _load_with_safe_globals(self, checkpoint_path):
        """Load checkpoint using safe_globals"""
        import numpy as np
        from torch.serialization import safe_globals
        with safe_globals([np.core.multiarray.scalar]):
            return torch.load(checkpoint_path, map_location=self.device)
    
    def train(self, train_loader, test_loader=None, resume_from=None, data_type=None):
        """
        Train model
        
        Args:
            train_loader: Training data loader
            test_loader: Test data loader (for evaluation)
            resume_from: Resume training from checkpoint (optional)
            data_type: Training data type ('strain' or 'stress')
            
        Returns:
            Training history
        """
        # Set data type
        self.data_type = data_type
        
        # Record history
        history = {
            'train_loss': [],
            'test_loss': [],
            'test_metrics': [],
            'data_type': data_type  # Add data type to history
        }
        
        # Set starting epoch
        start_epoch = 0
        
        # Resume training from checkpoint
        if resume_from and os.path.exists(resume_from):
            start_epoch = self.load_checkpoint(resume_from) + 1
            self.logger.info(f"Resuming training from epoch {start_epoch}")
        
        # Training loop
        num_epochs = self.config['epochs']
        self.logger.info(f"Starting training, total {num_epochs} epochs")
        
        for epoch in range(start_epoch, num_epochs):
            # Train for one epoch
            train_loss = self.train_epoch(train_loader, epoch)
            history['train_loss'].append(train_loss)
            
            # Record to TensorBoard
            if self.writer:
                self.writer.add_scalar('Loss/train', train_loss, epoch)
            
            # Evaluate on test set
            if test_loader is not None:
                test_loss, test_metrics = self.evaluate(test_loader, epoch)
                history['test_loss'].append(test_loss)
                history['test_metrics'].append(test_metrics)
            
                # Update learning rate
                self.scheduler.step(test_loss)
                
                # Check if model needs to be saved
                if test_loss < self.best_val_loss:
                    self.best_val_loss = test_loss
                    self.best_epoch = epoch
                    self.epochs_without_improvement = 0
                    
                    # Save model
                    if 'save_dir' in self.config:
                        os.makedirs(self.config['save_dir'], exist_ok=True)
                        checkpoint_path = os.path.join(self.config['save_dir'], f'best_model{self.data_type}.pth')
                        self.save_checkpoint(epoch, test_loss, test_metrics, True)
                        self.logger.info(f"Saving best model to {checkpoint_path}")
                else:
                    self.epochs_without_improvement += 1
                    self.logger.info(f"Test loss not improved, {self.epochs_without_improvement} epochs without improvement")
                    
                    # Early stopping
                    if self.epochs_without_improvement >= self.early_stop_patience:
                        self.logger.info(f"Early stopping triggered, stopping training")
                        break
            
                # Check if it's the best model
                is_best = test_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = test_loss
                    self.best_epoch = epoch
                    self.epochs_without_improvement = 0
                
                # Save checkpoint every 10 epochs or when it's the best model
                if (epoch + 1) % 10 == 0 or is_best:
                    self.save_checkpoint(epoch, test_loss, test_metrics, is_best)
                    if (epoch + 1) % 10 == 0:
                        self.logger.info(f"Periodic checkpoint save every 10 epochs, current epoch: {epoch+1}")
        
        # Load best model
        best_model_path = os.path.join(self.config['save_dir'], f'best_model{self.data_type}.pth')
        if os.path.exists(best_model_path):
            self.load_checkpoint(best_model_path)
            self.logger.info("Loading best model for final testing")
        
        # Final evaluation on test set
        if test_loader is not None:
            test_loss, test_metrics = self.evaluate(test_loader, epoch)
            history['final_test_loss'] = test_loss
            history['final_test_metrics'] = test_metrics
            self.logger.info(f"Final testing completed, loss: {test_loss:.6f}")
        
        # Cleanup
        if self.writer:
            self.writer.close()
        
        # Execute training end callback
        if self.on_training_end_callback:
            self.on_training_end_callback(history)
        
        return history

