"""
Metal Fatigue Life Prediction Model (LSTM+FCNN)

This model combines LSTM network for processing time series data and fully connected neural network 
for processing structural features to predict the fatigue life of metal materials.
"""

import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset

# Machine learning related libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score,
    r2_score, mean_absolute_error, mean_squared_error
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPRegressor


def load_and_preprocess_data(summary_path, csv_folder_path):
    """
    Load and preprocess fatigue data
    
    Args:
        summary_path: Path to summary CSV file
        csv_folder_path: Path to folder containing time series data
    
    Returns:
        Processed features, time series data and labels
    """
    # Load summary data
    pd_train = pd.read_csv(summary_path, nrows=100000)
    
    # Load time series data files
    csv_files = pd_train['load'].values
    value_list = []
    
    # Iterate through each time series file and load
    for i in range(len(csv_files)):
        try:
            one_df = pd.read_csv(csv_folder_path + csv_files[i], header=None).iloc[:, :2]
            value_list.append(one_df.values)
            # print(one_df.shape)
        except:
            print(csv_folder_path + csv_files[i], 'no here ')
    
    # Convert to NumPy array
    csv_value_array = np.array(value_list)
    
    # Preprocess numerical features
    num_cols = pd_train.select_dtypes(exclude=['object']).columns.tolist()[:-1]
    
    # Create preprocessing pipeline
    num_si_step = ('si', SimpleImputer(strategy='median'))  # Fill missing values with median
    num_ss_step = ('ss', StandardScaler())  # Standardize features
    num_steps = [num_si_step, num_ss_step]

    num_pipe = Pipeline(num_steps)
    num_transformers = [('num', num_pipe, num_cols)]
    
    ct = ColumnTransformer(transformers=num_transformers)
    
    # Apply preprocessing
    x_all = ct.fit_transform(pd_train)
    y_all = pd_train['Nf(label)'].values
    
    return x_all, csv_value_array, y_all, csv_files


def split_dataset(x_all, csv_value_array, y_all, csv_files, test_size=0.2, random_state=42):
    """
    Split dataset into training and test sets
    
    Args:
        x_all: Feature matrix
        csv_value_array: Time series data array
        y_all: Target variable
        csv_files: List of file names
        test_size: Test set ratio
        random_state: Random seed
    
    Returns:
        Training and test set data
    """
    x_train, x_test, csv_value_train, csv_value_test, y_train, y_test, csv_files_train, csv_files_test = train_test_split(
        x_all, csv_value_array, y_all, csv_files, test_size=test_size, random_state=random_state
    )
    
    print("Training set size:", x_train.shape[0])
    print("Test set size:", x_test.shape[0])
    
    return x_train, x_test, csv_value_train, csv_value_test, y_train, y_test


def create_dataloaders(x_train, x_test, csv_value_train, csv_value_test, y_train, y_test, batch_size=20):
    """
    Create PyTorch data loaders
    
    Args:
        x_train, x_test: Feature data
        csv_value_train, csv_value_test: Time series data
        y_train, y_test: Label data
        batch_size: Batch size
    
    Returns:
        Training and test DataLoaders
    """
    # Create TensorDataset
    train_ds = TensorDataset(
        torch.tensor(x_train).float(),
        torch.tensor(csv_value_train).float(), 
        torch.tensor(y_train).float().unsqueeze(1)
    )
    
    test_ds = TensorDataset(
        torch.tensor(x_test).float(),
        torch.tensor(csv_value_test).float(), 
        torch.tensor(y_test).float().unsqueeze(1)
    )
    
    # Create DataLoader
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    return train_dl, test_dl


class DNNModel(nn.Module):
    """
    Deep neural network model for processing structural features
    """
    def __init__(self, input_dim):
        super().__init__()
        self.nor = nn.BatchNorm1d(input_dim)  # Batch normalization layer
        self.lin1 = nn.Linear(input_dim, 200)  # First fully connected layer
        self.lin2 = nn.Linear(200, 100)        # Second fully connected layer
        self.lin3 = nn.Linear(100, 100)        # Third fully connected layer
        self.flatten = nn.Flatten()            # Flatten layer

    def forward(self, x):
        """Forward propagation"""
        x = F.relu(self.lin1(x))  # Apply ReLU activation function
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        return x


class LSTMModel(nn.Module):
    """
    LSTM model for processing time series data
    """
    def __init__(self, input_dim):
        super().__init__()
        self.nor = nn.BatchNorm1d(input_dim)   # Batch normalization layer
        self.lstm = nn.LSTM(2, 5)              # LSTM layer, input dimension 2, hidden state dimension 5
        self.flatten = nn.Flatten()            # Flatten layer
        self.lin1 = nn.Linear(1205, 100)       # Fully connected layer

    def forward(self, x):
        """Forward propagation"""
        x, _ = self.lstm(x)        # LSTM processing
        x = self.flatten(x)        # Flatten output
        x = self.lin1(x)           # Linear transformation
        return x


class CombinedModel(nn.Module):
    """
    Hybrid model combining DNN and LSTM
    """
    def __init__(self, input_dim):
        super().__init__()
        self.strct_block = DNNModel(input_dim)  # Structural feature processing module
        self.lstm_block = LSTMModel(input_dim)  # Time series data processing module
        self.flatten = nn.Flatten()
        # Combination layers
        self.lin1 = nn.Linear(200, 100)
        self.lin2 = nn.Linear(100, 100)
        self.out = nn.Linear(100, 1)           # Output layer

    def forward(self, b_x, b_x_csv):
        """
        Forward propagation
        
        Args:
            b_x: Structural feature input
            b_x_csv: Time series data input
        """
        x_strct = self.strct_block(b_x)        # Process structural features
        x_lstm = self.lstm_block(b_x_csv)      # Process time series data
        
        # Combine two types of features
        x = torch.stack((x_strct, x_lstm), dim=1)
        
        x = self.flatten(x)                    # Flatten combined features
        
        x = F.relu(self.lin1(x))               # Apply fully connected layer
        x = F.relu(self.lin2(x))
        x = self.out(x)                        # Generate prediction
        
        return x


def train_epoch(dataloader, model, loss_fn, optimizer, device):
    """
    Train for one epoch
    
    Args:
        dataloader: Data loader
        model: Model
        loss_fn: Loss function
        optimizer: Optimizer
        device: Computing device
    """
    size = len(dataloader.dataset)
    model.train()  # Set to training mode
    
    for batch, (X, x_csv, y) in enumerate(dataloader):
        # Move data to computing device
        X, x_csv, y = X.to(device), x_csv.to(device), y.to(device)
        
        # Compute prediction and loss
        pred = model(X, x_csv)
        loss = loss_fn(pred, y)
        
        # Backward propagation
        optimizer.zero_grad()  # Clear gradients
        loss.backward()        # Compute gradients
        optimizer.step()       # Update parameters
        
        # Print training progress
        if batch % 5 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def evaluate(dataloader, model, loss_fn, device):
    """
    Evaluate model performance
    
    Args:
        dataloader: Data loader
        model: Model
        loss_fn: Loss function
        device: Computing device
    
    Returns:
        Average loss
    """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()  # Set to evaluation mode
    test_loss = 0
    
    # Disable gradient computation
    with torch.no_grad():
        for (X, x_csv, y) in dataloader:
            X, x_csv, y = X.to(device), x_csv.to(device), y.to(device)
            pred = model(X, x_csv)
            test_loss += loss_fn(pred, y).item()
            
    test_loss /= num_batches
    print(f"Avg loss: {test_loss:>8f} \n")
    return test_loss


def train_model(model, train_dl, test_dl, loss_fn, optimizer, epochs, device, model_save_path):
    """
    Train model
    
    Args:
        model: Model
        train_dl: Training data loader
        test_dl: Test data loader
        loss_fn: Loss function
        optimizer: Optimizer
        epochs: Number of training epochs
        device: Computing device
        model_save_path: Model save path
    
    Returns:
        Training and test loss history
    """
    
    # Initialize best test loss
    best_test_loss = float('inf')
    train_loss_list = []
    test_loss_list = []
    
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        # Train
        train_epoch(train_dl, model, loss_fn, optimizer, device)
        
        # Evaluate
        train_loss = evaluate(train_dl, model, loss_fn, device)
        test_loss = evaluate(test_dl, model, loss_fn, device)
        
        # Record loss
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        
        # Save best model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"Best model saved! loss is {best_test_loss}")
    
    print("Done!")
    return train_loss_list, test_loss_list


def plot_loss_curves(train_loss, test_loss, save_path="/home/zty/code/mental_fatigue/lstm/lossvalue"):
    """
    Plot loss curves
    
    Args:
        train_loss: Training loss history
        test_loss: Test loss history
        save_path: Image save path
    """
    plt.figure()
    plt.plot(train_loss, label='train_loss')
    plt.plot(test_loss, label='test_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.legend(loc='upper right')
    plt.savefig(save_path, dpi=1200)
    plt.show()


def make_predictions(model, dataloader, device):
    """
    Make predictions using the model
    
    Args:
        model: Trained model
        dataloader: Data loader
        device: Computing device
    
    Returns:
        Predictions and actual values
    """
    predict_list = []
    label_list = []
    
    # Set to evaluation mode
    model.eval()
    
    with torch.no_grad():
        for (X, x_csv, y) in dataloader:
            X, x_csv, y = X.to(device), x_csv.to(device), y.to(device)
            predict_score = model(X, x_csv)
            predict_list.append(predict_score.flatten().detach().cpu().numpy().flatten())
            label_list.append(y.flatten().detach().cpu().numpy())
    
    # Merge predictions from all batches
    predict_array = np.hstack(predict_list)
    label_array = np.hstack(label_list)
    
    return predict_array, label_array


def save_predictions_to_csv(predictions, labels, filename):
    """
    Save prediction results to CSV file
    
    Args:
        predictions: Predicted values
        labels: Actual labels
        filename: Output filename
    """
    data_to_write = np.column_stack((predictions, labels))
    
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header row
        writer.writerow(['Prediction', 'Label'])
        
        # Write data
        for row in data_to_write:
            writer.writerow(row)
    
    print(f"Prediction results saved to {filename}")


def main():
    """Main function to execute the entire training and evaluation process"""
    # Data path settings
    summary_path = "/home/zty/code/mental_fatigue/dataset/data_all_strain-controlled.csv"  
    csv_folder_path = "/home/zty/code/mental_fatigue/dataset/All data_Strain/"  
    
    # Load and preprocess data
    x_all, csv_value_array, y_all, csv_files = load_and_preprocess_data(summary_path, csv_folder_path)
    
    # Split dataset
    x_train, x_test, csv_value_train, csv_value_test, y_train, y_test = split_dataset(
        x_all, csv_value_array, y_all, csv_files
    )
    
    # Create data loaders
    batch_size = 20
    train_dl, test_dl = create_dataloaders(
        x_train, x_test, csv_value_train, csv_value_test, y_train, y_test, batch_size
    )
    
    # Print dataset size
    print(f"\nTraining set size: {len(train_dl.dataset)}")
    print(f"Test set size: {len(test_dl.dataset)}\n")
    
    # Set computing device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using {device} device")
    
    # Feature dimension
    dim_num = x_all.shape[-1]
    
    # Initialize model
    dnn_block = DNNModel(dim_num).to(device)
    lstm_block = LSTMModel(dim_num).to(device)
    combined_model = CombinedModel(dim_num).to(device)
    
    # Set loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(combined_model.parameters(), lr=1e-3)
    
    # Train model
    epochs = 100
    model_save_path = '/home/zty/code/mental_fatigue/lstm/best_model_weights_LSTM.pth'
    train_loss, test_loss = train_model(
        combined_model, train_dl, test_dl, loss_fn, optimizer, epochs, device, model_save_path
    )
    
    # Plot training curves
    plot_loss_curves(train_loss, test_loss)
    
    # Load best model
    best_model = combined_model
    best_model.load_state_dict(torch.load(model_save_path))
    
    # Make predictions on test set
    print("\n" + "="*50)
    print("Evaluating model performance on test set:")
    predictions, labels = make_predictions(best_model, test_dl, device)
    
    # Calculate more evaluation metrics
    mse = mean_squared_error(labels, predictions)
    mae = mean_absolute_error(labels, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(labels, predictions)
    
    print(f"Mean Squared Error (MSE): {mse:.6f}")
    print(f"Mean Absolute Error (MAE): {mae:.6f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")
    print(f"R² Score: {r2:.6f}")
    print("="*50 + "\n")
    
    # Save prediction results
    save_predictions_to_csv(predictions, labels, "/home/zty/code/mental_fatigue/lstm/predictions_and_labels_train-LSTM.csv")


if __name__ == "__main__":
    main()








