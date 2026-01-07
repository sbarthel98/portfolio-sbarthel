"""
MLflow Hyperparameter Tuning with CNN, Dropout, and Normalization
This script implements advanced hyperparameter tuning with MLflow tracking,
including convolutional layers, dropout, and batch normalization.
"""

import os
import ssl
import warnings

# SSL fix for corporate networks
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['PYTHONHTTPSVERIFY'] = '0'
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
warnings.filterwarnings('ignore')
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Monkey-patch requests
import requests
_original_requests_get = requests.get
def patched_get(url, **kwargs):
    kwargs['verify'] = False
    return _original_requests_get(url, **kwargs)
requests.get = patched_get

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from mads_datasets import DatasetFactoryProvider, DatasetType
import mlflow
import mlflow.pytorch
from pathlib import Path
from typing import List, Dict, Any
import itertools
from tqdm import tqdm


class FlexibleCNN(nn.Module):
    """
    Flexible CNN architecture with configurable:
    - Convolutional layers (with optional pooling)
    - Dropout
    - Batch normalization
    - Fully connected layers
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # Extract configuration
        num_conv_layers = config.get('num_conv_layers', 2)
        conv_channels = config.get('conv_channels', [32, 64])
        kernel_size = config.get('kernel_size', 3)
        use_pooling = config.get('use_pooling', True)
        use_batch_norm = config.get('use_batch_norm', True)
        dropout_rate = config.get('dropout_rate', 0.5)
        fc_units = config.get('fc_units', [128])
        
        # Build convolutional layers using ModuleList
        self.conv_layers = nn.ModuleList()
        in_channels = 1  # Fashion MNIST is grayscale
        
        for i in range(num_conv_layers):
            out_channels = conv_channels[i] if i < len(conv_channels) else conv_channels[-1]
            
            # Conv block: Conv2d -> BatchNorm (optional) -> ReLU -> MaxPool (optional)
            block = nn.ModuleList()
            block.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding=1))
            
            if use_batch_norm:
                block.append(nn.BatchNorm2d(out_channels))
            
            block.append(nn.ReLU())
            
            if use_pooling:
                block.append(nn.MaxPool2d(2))
            
            self.conv_layers.append(block)
            in_channels = out_channels
        
        # Calculate the flattened size after conv layers
        # Fashion MNIST is 28x28
        spatial_size = 28
        for _ in range(num_conv_layers):
            if use_pooling:
                spatial_size = spatial_size // 2
        
        flattened_size = conv_channels[min(num_conv_layers-1, len(conv_channels)-1)] * spatial_size * spatial_size
        
        # Build fully connected layers using ModuleList
        self.fc_layers = nn.ModuleList()
        in_features = flattened_size
        
        for fc_size in fc_units:
            self.fc_layers.append(nn.Linear(in_features, fc_size))
            self.fc_layers.append(nn.ReLU())
            if dropout_rate > 0:
                self.fc_layers.append(nn.Dropout(dropout_rate))
            in_features = fc_size
        
        # Output layer
        self.output = nn.Linear(in_features, 10)  # 10 classes for Fashion MNIST
    
    def forward(self, x):
        # Convolutional layers
        for block in self.conv_layers:
            for layer in block:
                x = layer(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        for layer in self.fc_layers:
            x = layer(x)
        
        # Output
        x = self.output(x)
        return x


def setup_data(batchsize: int):
    """Load Fashion MNIST dataset"""
    # Get Fashion MNIST
    fashionfactory = DatasetFactoryProvider.create_factory(DatasetType.FASHION)
    
    # Create simple datastreamers without preprocessor
    streamers = fashionfactory.create_datastreamer(batchsize=batchsize)
    
    train = streamers["train"]
    valid = streamers["valid"]
    
    trainstreamer = train.stream()
    validstreamer = valid.stream()
    
    # len(train) returns number of batches already prepared, not dataset size
    # So we return it directly as the number of batches per epoch
    return trainstreamer, validstreamer, len(train), len(valid)


def train_epoch(model, dataloader, criterion, optimizer, device, max_batches=None):
    """Train for one epoch with progress bar"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Use max_batches as the progress bar total if provided
    total_batches = max_batches if max_batches else None
    
    pbar = tqdm(enumerate(dataloader), total=total_batches, desc="Training", leave=False)
    
    for batch_idx, batch in pbar:
        # Stop early if max_batches specified
        if max_batches and batch_idx >= max_batches:
            break
            
        # Unpack batch: first element is tuple of images, second is tuple of labels
        inputs_tuple, labels_tuple = batch
        # Stack the tuple of tensors into a single batch tensor
        inputs = torch.stack(inputs_tuple)
        labels = torch.tensor(labels_tuple)
        
        # Reshape for CNN (add channel dimension if needed)
        if len(inputs.shape) == 3:
            inputs = inputs.unsqueeze(1)  # Add channel dimension
        
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar with current metrics
        current_loss = running_loss / (batch_idx + 1)
        current_acc = 100 * correct / total
        pbar.set_postfix({'loss': f'{current_loss:.4f}', 'acc': f'{current_acc:.2f}%'})
    
    # Use batch_idx + 1 as number of batches processed
    num_batches = batch_idx + 1
    epoch_loss = running_loss / num_batches
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device, max_batches=None):
    """Validate model with progress bar"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Use max_batches as the progress bar total if provided
    total_batches = max_batches if max_batches else None
    
    pbar = tqdm(enumerate(dataloader), total=total_batches, desc="Validating", leave=False)
    
    with torch.no_grad():
        for batch_idx, batch in pbar:
            # Stop early if max_batches specified
            if max_batches and batch_idx >= max_batches:
                break
                
            # Unpack batch: first element is tuple of images, second is tuple of labels
            inputs_tuple, labels_tuple = batch
            # Stack the tuple of tensors into a single batch tensor
            inputs = torch.stack(inputs_tuple)
            labels = torch.tensor(labels_tuple)
            
            # Reshape for CNN
            if len(inputs.shape) == 3:
                inputs = inputs.unsqueeze(1)
            
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar with current metrics
            current_loss = running_loss / (batch_idx + 1)
            current_acc = 100 * correct / total
            pbar.set_postfix({'loss': f'{current_loss:.4f}', 'acc': f'{current_acc:.2f}%'})
    
    # Use batch_idx + 1 as number of batches processed
    num_batches = batch_idx + 1
    val_loss = running_loss / num_batches
    val_acc = 100 * correct / total
    return val_loss, val_acc


def run_experiment(config: Dict[str, Any], experiment_name: str):
    """Run a single experiment with MLflow tracking"""
    
    print(f"\n{'='*70}")
    print(f"Running experiment: {experiment_name}")
    print(f"Config: {config}")
    print(f"{'='*70}\n")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Start MLflow run
    with mlflow.start_run(run_name=experiment_name):
        # Log all hyperparameters
        mlflow.log_params(config)
        
        # Setup data
        batchsize = config.get('batchsize', 64)
        trainloader, validloader, train_len, valid_len = setup_data(batchsize)
        
        # train_len and valid_len are already the number of batches (not dataset size)
        # Since streamers are infinite, we need to limit iterations
        train_batches_per_epoch = train_len
        valid_batches_per_epoch = valid_len
        
        # Allow override for quick testing
        max_train_batches = config.get('max_batches', train_batches_per_epoch)
        max_valid_batches = config.get('max_batches', valid_batches_per_epoch)
        
        print(f"Training batches per epoch: {max_train_batches}")
        print(f"Validation batches per epoch: {max_valid_batches}")
        
        # Create model
        model = FlexibleCNN(config).to(device)
        
        # Log model summary
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        mlflow.log_param("total_params", total_params)
        mlflow.log_param("trainable_params", trainable_params)
        
        print(f"Model has {total_params:,} total parameters ({trainable_params:,} trainable)")
        
        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer_class = getattr(optim, config.get('optimizer', 'Adam'))
        optimizer = optimizer_class(
            model.parameters(), 
            lr=config.get('learning_rate', 0.001)
        )
        
        # Training loop
        epochs = config.get('epochs', 10)
        best_val_acc = 0.0
        
        # Create progress bar for epochs
        epoch_pbar = tqdm(range(epochs), desc=f"Epochs ({experiment_name})", position=0)
        
        for epoch in epoch_pbar:
            train_loss, train_acc = train_epoch(model, trainloader, criterion, optimizer, device, max_train_batches)
            val_loss, val_acc = validate(model, validloader, criterion, device, max_valid_batches)
            
            # Update epoch progress bar with metrics
            epoch_pbar.set_postfix({
                'train_loss': f'{train_loss:.4f}',
                'train_acc': f'{train_acc:.2f}%',
                'val_loss': f'{val_loss:.4f}',
                'val_acc': f'{val_acc:.2f}%'
            })
            
            # Log metrics to MLflow
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_accuracy", train_acc, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
            
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Log final metrics
        mlflow.log_metric("final_train_loss", train_loss)
        mlflow.log_metric("final_train_accuracy", train_acc)
        mlflow.log_metric("final_val_loss", val_loss)
        mlflow.log_metric("final_val_accuracy", val_acc)
        mlflow.log_metric("best_val_accuracy", best_val_acc)
        
        # Save model
        mlflow.pytorch.log_model(model, "model")
        
        print(f"\nCompleted! Best validation accuracy: {best_val_acc:.2f}%\n")
        
        return {
            'final_val_acc': val_acc,
            'final_val_loss': val_loss,
            'best_val_acc': best_val_acc
        }


# ============================================================================
# EXPERIMENT GROUPS
# ============================================================================

def experiment_1_dropout():
    """Experiment 1: Impact of Dropout Rate"""
    print("\n" + "="*70)
    print("EXPERIMENT 1: Impact of Dropout Rate")
    print("="*70)
    
    dropout_rates = [0.0, 0.2, 0.3, 0.5, 0.7]
    base_config = {
        'epochs': 10,
        'batchsize': 64,
        'num_conv_layers': 2,
        'conv_channels': [32, 64],
        'kernel_size': 3,
        'use_pooling': True,
        'use_batch_norm': True,
        'fc_units': [128],
        'learning_rate': 0.001,
        'optimizer': 'Adam'
    }
    
    print(f"📊 Running {len(dropout_rates)} experiments with different dropout rates\n")
    
    for idx, dropout in enumerate(dropout_rates, 1):
        print(f"\n🔬 Experiment {idx}/{len(dropout_rates)}: Testing dropout={dropout}")
        config = base_config.copy()
        config['dropout_rate'] = dropout
        run_experiment(config, f"dropout_{dropout}")
        print(f"✅ Completed experiment {idx}/{len(dropout_rates)}\n")


def experiment_2_batch_norm():
    """Experiment 2: Batch Normalization vs No Normalization"""
    print("\n" + "="*70)
    print("EXPERIMENT 2: Impact of Batch Normalization")
    print("="*70)
    
    base_config = {
        'epochs': 10,
        'batchsize': 64,
        'num_conv_layers': 2,
        'conv_channels': [32, 64],
        'kernel_size': 3,
        'use_pooling': True,
        'dropout_rate': 0.5,
        'fc_units': [128],
        'learning_rate': 0.001,
        'optimizer': 'Adam'
    }
    
    for use_bn in [False, True]:
        config = base_config.copy()
        config['use_batch_norm'] = use_bn
        run_experiment(config, f"batchnorm_{use_bn}")


def experiment_3_conv_depth():
    """Experiment 3: Number of Convolutional Layers"""
    print("\n" + "="*70)
    print("EXPERIMENT 3: Impact of Convolutional Layer Depth")
    print("="*70)
    
    base_config = {
        'epochs': 10,
        'batchsize': 64,
        'kernel_size': 3,
        'use_pooling': True,
        'use_batch_norm': True,
        'dropout_rate': 0.5,
        'fc_units': [128],
        'learning_rate': 0.001,
        'optimizer': 'Adam'
    }
    
    configs = [
        {'num_conv_layers': 1, 'conv_channels': [32]},
        {'num_conv_layers': 2, 'conv_channels': [32, 64]},
        {'num_conv_layers': 3, 'conv_channels': [32, 64, 128]},
    ]
    
    for conv_config in configs:
        config = base_config.copy()
        config.update(conv_config)
        num_layers = conv_config['num_conv_layers']
        run_experiment(config, f"conv_depth_{num_layers}")


def experiment_4_pooling():
    """Experiment 4: Pooling vs No Pooling"""
    print("\n" + "="*70)
    print("EXPERIMENT 4: Impact of MaxPooling")
    print("="*70)
    
    base_config = {
        'epochs': 10,
        'batchsize': 64,
        'num_conv_layers': 2,
        'conv_channels': [32, 64],
        'kernel_size': 3,
        'use_batch_norm': True,
        'dropout_rate': 0.5,
        'fc_units': [128],
        'learning_rate': 0.001,
        'optimizer': 'Adam'
    }
    
    for use_pool in [False, True]:
        config = base_config.copy()
        config['use_pooling'] = use_pool
        run_experiment(config, f"pooling_{use_pool}")


def experiment_5_interactions():
    """Experiment 5: Interactions between Dropout and Batch Normalization"""
    print("\n" + "="*70)
    print("EXPERIMENT 5: Dropout x Batch Normalization Interactions")
    print("="*70)
    
    base_config = {
        'epochs': 10,
        'batchsize': 64,
        'num_conv_layers': 2,
        'conv_channels': [32, 64],
        'kernel_size': 3,
        'use_pooling': True,
        'fc_units': [128],
        'learning_rate': 0.001,
        'optimizer': 'Adam'
    }
    
    dropout_rates = [0.0, 0.5]
    batch_norms = [False, True]
    
    for dropout, use_bn in itertools.product(dropout_rates, batch_norms):
        config = base_config.copy()
        config['dropout_rate'] = dropout
        config['use_batch_norm'] = use_bn
        run_experiment(config, f"interaction_dropout{dropout}_bn{use_bn}")


def experiment_6_combined_optimal():
    """Experiment 6: Combined Optimal Configuration"""
    print("\n" + "="*70)
    print("EXPERIMENT 6: Combined Optimal Configurations")
    print("="*70)
    
    # Based on findings from previous experiments, test promising combinations
    configs = [
        {
            'name': 'optimal_shallow',
            'epochs': 15,
            'batchsize': 64,
            'num_conv_layers': 2,
            'conv_channels': [32, 64],
            'kernel_size': 3,
            'use_pooling': True,
            'use_batch_norm': True,
            'dropout_rate': 0.3,
            'fc_units': [256, 128],
            'learning_rate': 0.001,
            'optimizer': 'Adam'
        },
        {
            'name': 'optimal_deep',
            'epochs': 15,
            'batchsize': 64,
            'num_conv_layers': 3,
            'conv_channels': [32, 64, 128],
            'kernel_size': 3,
            'use_pooling': True,
            'use_batch_norm': True,
            'dropout_rate': 0.5,
            'fc_units': [256],
            'learning_rate': 0.001,
            'optimizer': 'Adam'
        },
        {
            'name': 'optimal_aggressive_dropout',
            'epochs': 15,
            'batchsize': 128,
            'num_conv_layers': 2,
            'conv_channels': [64, 128],
            'kernel_size': 3,
            'use_pooling': True,
            'use_batch_norm': True,
            'dropout_rate': 0.7,
            'fc_units': [512, 256],
            'learning_rate': 0.0005,
            'optimizer': 'AdamW'
        }
    ]
    
    for config in configs:
        name = config.pop('name')
        run_experiment(config, f"combined_{name}")


def main():
    """Main execution function"""
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("fashion_mnist_cnn_tuning")
    
    print("="*70)
    print("MLflow CNN HYPERPARAMETER TUNING - Fashion MNIST")
    print("="*70)
    print("\nThis script will run experiments testing:")
    print("1. Dropout rates (0.0 to 0.7)")
    print("2. Batch normalization (with/without)")
    print("3. Convolutional depth (1-3 layers)")
    print("4. Pooling (with/without)")
    print("5. Interactions (dropout x batch norm)")
    print("6. Combined optimal configurations")
    print("\nResults will be tracked in MLflow.")
    print("="*70)
    
    # Run all experiments
    experiment_1_dropout()
    experiment_2_batch_norm()
    experiment_3_conv_depth()
    experiment_4_pooling()
    experiment_5_interactions()
    experiment_6_combined_optimal()
    
    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETE!")
    print("="*70)
    print("\nView results with: mlflow ui")
    print("Then open: http://localhost:5000")
    print("="*70)


if __name__ == "__main__":
    main()
