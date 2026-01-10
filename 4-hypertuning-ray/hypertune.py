
import os
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Dict
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from filelock import FileLock
from loguru import logger
from mads_datasets import DatasetFactoryProvider, DatasetType

# --------------------------------------------------------------------------------
# Configurable Model
# --------------------------------------------------------------------------------
class TunableCNN(nn.Module):
    def __init__(self, config: Dict, num_classes: int = 5, input_shape: tuple = (3, 224, 224)):
        super(TunableCNN, self).__init__()
        
        layers = []
        in_channels = input_shape[0]
        
        # Dynamic Convolutional Layers
        # We'll rely on global "filters" config or per-layer config
        # Simplified: Double filters every layer, starting from config["start_filters"]
        start_filters = config.get("start_filters", 32)
        num_conv = config.get("num_conv_layers", 3)
        dropout_p = config.get("dropout", 0.2)
        
        current_filters = start_filters
        current_size = input_shape[1]
        
        for i in range(num_conv):
            layers.append(nn.Conv2d(in_channels, current_filters, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(current_filters))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2)) # Reduces size by 2
            
            in_channels = current_filters
            current_filters *= 2
            current_size //= 2
            
        self.features = nn.Sequential(*layers)
        
        # Fully Connected Layers
        # Calculate flattened size
        self.flat_size = in_channels * current_size * current_size
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flat_size, config.get("fc_units", 128)),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(config.get("fc_units", 128), num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# --------------------------------------------------------------------------------
# Training Loop
# --------------------------------------------------------------------------------
from torch.utils.data import DataLoader

def train_experiment(config: Dict, data_dir=None):
    # Setup Data
    if data_dir is None:
        data_dir = Path.home() / ".cache/mads_datasets"
    else:
        data_dir = Path(data_dir)
    
    with FileLock(data_dir / ".lock"):
        factory = DatasetFactoryProvider.create_factory(DatasetType.FLOWERS)
        streamers = factory.create_datastreamer(batchsize=config.get("batch_size", 32))
        
        # Create DataLoaders directly
        train_loader = DataLoader(streamers["train"].dataset, batch_size=config.get("batch_size", 32), shuffle=True)
        valid_loader = DataLoader(streamers["valid"].dataset, batch_size=config.get("batch_size", 32), shuffle=False)
        
        train_len = len(train_loader)
        valid_len = len(valid_loader)

    # Use GPU if available (important for faster training)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on device: {device}")
    
    model = TunableCNN(config, num_classes=5).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.get("lr", 1e-3))
    
    for epoch in range(10): # Max 10 epochs per trial controlled by scheduler
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Training loop
        for i, batch in enumerate(train_loader):
            inputs, labels = batch[0].to(device), batch[1].to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
                
        # Validation loop
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in valid_loader:
                inputs, labels = batch[0].to(device), batch[1].to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        # Calculate metrics
        avg_train_loss = train_loss / train_total if train_total > 0 else 0.0
        avg_val_loss = val_loss / val_total if val_total > 0 else 0.0
        val_accuracy = val_correct / val_total if val_total > 0 else 0.0
        
        # Report to Ray Tune - must pass dict, not keyword args
        tune.report({"loss": avg_val_loss, "accuracy": val_accuracy})

# --------------------------------------------------------------------------------
# Main Execution
# --------------------------------------------------------------------------------
if __name__ == "__main__":
    # Check GPU before Ray init
    use_gpu = torch.cuda.is_available()
    
    # Initialize Ray
    ray.init(ignore_reinit_error=True)
    
    # Check GPU
    if use_gpu:
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        # If memory allows, we can run fractional GPUs (e.g. 0.5) to run 2 trials in parallel
        resources_per_trial = {"cpu": 2, "gpu": 0.5} 
    else:
        logger.info("Using CPU")
        resources_per_trial = {"cpu": 2}

    # Configuration Search Space
    config = {
        "num_conv_layers": tune.choice([2, 3, 4]),
        "start_filters": tune.choice([16, 32, 64]),
        "fc_units": tune.choice([64, 128, 256]),
        "dropout": tune.uniform(0.1, 0.5),
        "lr": tune.loguniform(1e-4, 1e-2),
        "batch_size": tune.choice([16, 32]), # 224x224 images use memory
    }
    
    # Scheduler: ASHA (Async HyperBand)
    scheduler = AsyncHyperBandScheduler(
        time_attr="training_iteration",
        metric="accuracy",
        mode="max",
        max_t=10,
        grace_period=2,
        reduction_factor=3
    )

    # Search Algorithm: HyperOpt
    # We need to install hyperopt: pip install hyperopt
    search_alg = HyperOptSearch(metric="accuracy", mode="max")

    # Reporter
    reporter = CLIReporter(
        parameter_columns=["num_conv_layers", "start_filters", "fc_units", "lr"],
        metric_columns=["loss", "accuracy", "training_iteration"]
    )
    
    # Custom trial directory name creator to avoid long paths on Windows
    def trial_dirname_creator(trial):
        return f"trial_{trial.trial_id}"

    logger.info("Starting Ray Tune experiments...")
    
    # Use with_parameters to pass data_dir
    trainable = tune.with_parameters(train_experiment, data_dir=str(Path.home() / ".cache/mads_datasets"))
    
    result = tune.run(
        trainable,
        resources_per_trial=resources_per_trial,
        config=config,
        num_samples=10, # Number of trials
        scheduler=scheduler,
        search_alg=search_alg,
        progress_reporter=reporter,
        storage_path=str(Path("logs/ray_results").resolve()),
        name="flowers",
        trial_dirname_creator=trial_dirname_creator
    )

    best_trial = result.get_best_trial("accuracy", "max", "last")
    logger.info(f"Best trial config: {best_trial.config}")
    logger.info(f"Best trial final validation accuracy: {best_trial.last_result['accuracy']}")
    
    ray.shutdown()
