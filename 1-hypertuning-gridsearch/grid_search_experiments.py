"""
Grid Search Hyperparameter Tuning Experiments
This script systematically tests different hyperparameter combinations
as specified in the instructions.
"""

import os
import ssl
import warnings

# Fix SSL certificate verification issues (corporate proxy workaround)
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['PYTHONHTTPSVERIFY'] = '0'
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''

# Disable SSL warnings
warnings.filterwarnings('ignore')
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Monkey-patch requests before it's imported by mads_datasets
import sys
import requests
from unittest.mock import patch

# Store the original get function
_original_requests_get = requests.get

# Create a wrapper that forces verify=False
def patched_get(url, **kwargs):
    kwargs['verify'] = False
    return _original_requests_get(url, **kwargs)

# Replace the get function
requests.get = patched_get

import torch
import torch.optim as optim
from torch import nn
from mads_datasets import DatasetFactoryProvider, DatasetType
from mltrainer.preprocessors import BasePreprocessor
from mltrainer import Trainer, TrainerSettings, ReportTypes, metrics
from tomlserializer import TOMLSerializer
import itertools
from pathlib import Path


class NeuralNetwork(nn.Module):
    """Basic Neural Network with configurable hidden layers"""
    def __init__(self, num_classes: int, units1: int, units2: int, depth: int = 2) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.units1 = units1
        self.units2 = units2
        self.depth = depth
        self.flatten = nn.Flatten()
        
        # Build dynamic network based on depth
        layers = []
        layers.extend([
            nn.Linear(28 * 28, units1),
            nn.ReLU(),
        ])
        
        if depth >= 2:
            layers.extend([
                nn.Linear(units1, units2),
                nn.ReLU(),
            ])
        
        # Add extra layer for depth=3
        if depth >= 3:
            layers.extend([
                nn.Linear(units2, units2),
                nn.ReLU(),
            ])
        
        # Output layer
        if depth == 1:
            layers.append(nn.Linear(units1, num_classes))
        else:
            layers.append(nn.Linear(units2, num_classes))
        
        self.linear_relu_stack = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def setup_data(batchsize: int):
    """Set up the Fashion MNIST dataset with given batch size"""
    fashionfactory = DatasetFactoryProvider.create_factory(DatasetType.FASHION)
    preprocessor = BasePreprocessor()
    streamers = fashionfactory.create_datastreamer(batchsize=batchsize, preprocessor=preprocessor)
    train = streamers["train"]
    valid = streamers["valid"]
    return train.stream(), valid.stream(), len(train), len(valid)


def run_experiment(config: dict, experiment_name: str):
    """Run a single experiment with given configuration"""
    print(f"\n{'='*60}")
    print(f"Running experiment: {experiment_name}")
    print(f"Config: {config}")
    print(f"{'='*60}\n")
    
    # Setup data with specified batch size
    trainstreamer, validstreamer, train_len, valid_len = setup_data(config['batchsize'])
    
    # Create model
    model = NeuralNetwork(
        num_classes=10,
        units1=config['units1'],
        units2=config['units2'],
        depth=config.get('depth', 2)
    )
    
    # Setup accuracy metric
    accuracy = metrics.Accuracy()
    
    # Loss function
    loss_fn = torch.nn.CrossEntropyLoss()
    
    # Optimizer selection
    optimizer_map = {
        'SGD': optim.SGD,
        'Adam': optim.Adam,
        'AdamW': optim.AdamW,
        'RMSprop': optim.RMSprop,
    }
    optimizer_class = optimizer_map.get(config['optimizer'], optim.Adam)
    
    # Training settings
    settings = TrainerSettings(
        epochs=config['epochs'],
        metrics=[accuracy],
        logdir="modellogs",
        train_steps=config.get('train_steps', 100),
        valid_steps=config.get('valid_steps', 100),
        reporttypes=[ReportTypes.TENSORBOARD],
        optimizer_kwargs={'lr': config['learning_rate']},
    )
    
    # Create trainer and run
    trainer = Trainer(
        model=model,
        settings=settings,
        loss_fn=loss_fn,
        optimizer=optimizer_class,
        traindataloader=trainstreamer,
        validdataloader=validstreamer,
        scheduler=optim.lr_scheduler.ReduceLROnPlateau
    )
    
    trainer.loop()
    
    # Save results to TOML file
    log_dir = trainer.logger.logdir
    toml_file = Path(log_dir) / f"{experiment_name}_results.toml"
    results = {
        'experiment_name': experiment_name,
        'config': config,
        'final_train_loss': float(trainer.train_loss) if hasattr(trainer, 'train_loss') else None,
        'final_valid_loss': float(trainer.valid_loss) if hasattr(trainer, 'valid_loss') else None,
    }
    serializer = TOMLSerializer()
    serializer.serialize(results, str(toml_file))
    print(f"Results saved to: {toml_file}")
    print(f"Completed experiment: {experiment_name}\n")


def experiment_1_epochs():
    """Experiment 1: Test different epoch counts"""
    print("\n" + "="*70)
    print("EXPERIMENT 1: Impact of Epochs")
    print("="*70)
    
    epochs_list = [3, 5, 10]
    base_config = {
        'batchsize': 64,
        'units1': 256,
        'units2': 256,
        'learning_rate': 0.001,
        'optimizer': 'Adam',
        'train_steps': 100,
        'valid_steps': 100,
    }
    
    for epochs in epochs_list:
        config = base_config.copy()
        config['epochs'] = epochs
        run_experiment(config, f"epochs_{epochs}")


def experiment_2_units():
    """Experiment 2: Test different unit combinations"""
    print("\n" + "="*70)
    print("EXPERIMENT 2: Impact of Hidden Units")
    print("="*70)
    
    units_list = [16, 32, 64, 128, 256, 512]
    base_config = {
        'epochs': 5,
        'batchsize': 64,
        'learning_rate': 0.001,
        'optimizer': 'Adam',
        'train_steps': 100,
        'valid_steps': 100,
    }
    
    # Test combinations
    for units1, units2 in itertools.product(units_list, repeat=2):
        config = base_config.copy()
        config['units1'] = units1
        config['units2'] = units2
        run_experiment(config, f"units_{units1}_{units2}")


def experiment_3_batchsize():
    """Experiment 3: Test different batch sizes"""
    print("\n" + "="*70)
    print("EXPERIMENT 3: Impact of Batch Size")
    print("="*70)
    
    batchsize_list = [4, 8, 16, 32, 64, 128]
    base_config = {
        'epochs': 5,
        'units1': 256,
        'units2': 256,
        'learning_rate': 0.001,
        'optimizer': 'Adam',
        'train_steps': 100,
        'valid_steps': 100,
    }
    
    for batchsize in batchsize_list:
        config = base_config.copy()
        config['batchsize'] = batchsize
        run_experiment(config, f"batchsize_{batchsize}")


def experiment_4_depth():
    """Experiment 4: Test different model depths"""
    print("\n" + "="*70)
    print("EXPERIMENT 4: Impact of Model Depth")
    print("="*70)
    
    depth_list = [1, 2, 3]
    base_config = {
        'epochs': 5,
        'batchsize': 64,
        'units1': 256,
        'units2': 256,
        'learning_rate': 0.001,
        'optimizer': 'Adam',
        'train_steps': 100,
        'valid_steps': 100,
    }
    
    for depth in depth_list:
        config = base_config.copy()
        config['depth'] = depth
        run_experiment(config, f"depth_{depth}")


def experiment_5_learning_rate():
    """Experiment 5: Test different learning rates"""
    print("\n" + "="*70)
    print("EXPERIMENT 5: Impact of Learning Rate")
    print("="*70)
    
    lr_list = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 1e-5]
    base_config = {
        'epochs': 5,
        'batchsize': 64,
        'units1': 256,
        'units2': 256,
        'optimizer': 'Adam',
        'train_steps': 100,
        'valid_steps': 100,
    }
    
    for lr in lr_list:
        config = base_config.copy()
        config['learning_rate'] = lr
        run_experiment(config, f"lr_{lr}")


def experiment_6_optimizers():
    """Experiment 6: Test different optimizers"""
    print("\n" + "="*70)
    print("EXPERIMENT 6: Impact of Different Optimizers")
    print("="*70)
    
    optimizer_list = ['SGD', 'Adam', 'AdamW', 'RMSprop']
    base_config = {
        'epochs': 5,
        'batchsize': 64,
        'units1': 256,
        'units2': 256,
        'learning_rate': 0.001,
        'train_steps': 100,
        'valid_steps': 100,
    }
    
    for optimizer in optimizer_list:
        config = base_config.copy()
        config['optimizer'] = optimizer
        run_experiment(config, f"optimizer_{optimizer}")


def experiment_7_combined():
    """Experiment 7: Test promising combinations"""
    print("\n" + "="*70)
    print("EXPERIMENT 7: Combined Best Configurations")
    print("="*70)
    
    # Based on typical best practices, test a few promising combinations
    configs = [
        {
            'epochs': 10,
            'batchsize': 32,
            'units1': 512,
            'units2': 256,
            'depth': 2,
            'learning_rate': 0.001,
            'optimizer': 'Adam',
            'train_steps': 100,
            'valid_steps': 100,
        },
        {
            'epochs': 10,
            'batchsize': 64,
            'units1': 256,
            'units2': 128,
            'depth': 3,
            'learning_rate': 0.0005,
            'optimizer': 'AdamW',
            'train_steps': 100,
            'valid_steps': 100,
        },
        {
            'epochs': 5,
            'batchsize': 128,
            'units1': 128,
            'units2': 64,
            'depth': 2,
            'learning_rate': 0.001,
            'optimizer': 'Adam',
            'train_steps': 100,
            'valid_steps': 100,
        },
    ]
    
    for i, config in enumerate(configs, 1):
        run_experiment(config, f"combined_config_{i}")


def main():
    """Main function to run all experiments"""
    print("\n" + "="*70)
    print("HYPERPARAMETER TUNING GRID SEARCH EXPERIMENTS")
    print("="*70)
    print("\nThis script will run comprehensive experiments to test:")
    print("1. Different epoch counts")
    print("2. Different hidden unit sizes")
    print("3. Different batch sizes")
    print("4. Different model depths")
    print("5. Different learning rates")
    print("6. Different optimizers")
    print("7. Combined promising configurations")
    print("\nResults will be saved in the 'modellogs' directory.")
    print("="*70 + "\n")
    
    # Run all experiments
    # Note: Comment out experiments you don't want to run
    
    experiment_1_epochs()
    # experiment_2_units()  # This will take a long time (36 combinations)
    experiment_3_batchsize()
    experiment_4_depth()
    experiment_5_learning_rate()
    experiment_6_optimizers()
    experiment_7_combined()
    
    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETED!")
    print("="*70)
    print("\nNext steps:")
    print("1. Run tensorboard to visualize results:")
    print("   tensorboard --logdir=modellogs")
    print("2. Run the analysis script to generate visualizations")
    print("3. Review the generated report")


if __name__ == "__main__":
    main()
