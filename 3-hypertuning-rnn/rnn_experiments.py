"""
RNN Hyperparameter Tuning Experiments for Gesture Recognition
Systematically test GRU, LSTM, and Conv1D+RNN architectures

Using native PyTorch with GPU acceleration for faster training
"""

from pathlib import Path
import torch
import torch.nn as nn
from torch import Tensor, optim
from dataclasses import dataclass
import mlflow
from tqdm import tqdm

from mads_datasets import DatasetFactoryProvider, DatasetType
from mltrainer.preprocessors import PaddedPreprocessor


# ============================================================================
# Model Configurations
# ============================================================================

@dataclass
class ModelConfig:
    input_size: int
    hidden_size: int
    num_layers: int
    output_size: int
    dropout: float = 0.0


# ============================================================================
# Model Architectures
# ============================================================================

class GRUModel(nn.Module):
    """Basic GRU model for sequence classification"""
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.rnn = nn.GRU(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            batch_first=True,
            num_layers=config.num_layers,
        )
        self.linear = nn.Linear(config.hidden_size, config.output_size)

    def forward(self, x: Tensor) -> Tensor:
        x, _ = self.rnn(x)
        last_step = x[:, -1, :]
        yhat = self.linear(last_step)
        return yhat


class LSTMModel(nn.Module):
    """Basic LSTM model for sequence classification"""
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.rnn = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            batch_first=True,
            num_layers=config.num_layers,
        )
        self.linear = nn.Linear(config.hidden_size, config.output_size)

    def forward(self, x: Tensor) -> Tensor:
        x, _ = self.rnn(x)
        last_step = x[:, -1, :]
        yhat = self.linear(last_step)
        return yhat


class BidirectionalGRU(nn.Module):
    """Bidirectional GRU for better sequence understanding"""
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.rnn = nn.GRU(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            batch_first=True,
            num_layers=config.num_layers,
            bidirectional=True,
        )
        # Bidirectional doubles the hidden size
        self.linear = nn.Linear(config.hidden_size * 2, config.output_size)

    def forward(self, x: Tensor) -> Tensor:
        x, _ = self.rnn(x)
        last_step = x[:, -1, :]
        yhat = self.linear(last_step)
        return yhat


class Conv1DGRU(nn.Module):
    """Conv1D feature extraction followed by GRU"""
    def __init__(self, config: ModelConfig, conv_channels: int = 16) -> None:
        super().__init__()
        self.config = config
        
        # 1D Convolution for feature extraction
        self.conv1 = nn.Conv1d(
            in_channels=config.input_size,
            out_channels=conv_channels,
            kernel_size=3,
            padding=1
        )
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        # GRU on extracted features
        self.rnn = nn.GRU(
            input_size=conv_channels,
            hidden_size=config.hidden_size,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            batch_first=True,
            num_layers=config.num_layers,
        )
        self.linear = nn.Linear(config.hidden_size, config.output_size)

    def forward(self, x: Tensor) -> Tensor:
        # x shape: (batch, seq_len, features)
        # Conv1d expects: (batch, features, seq_len)
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        # Back to (batch, seq_len, features)
        x = x.transpose(1, 2)
        
        x, _ = self.rnn(x)
        last_step = x[:, -1, :]
        yhat = self.linear(last_step)
        return yhat


class DeepGRU(nn.Module):
    """Deep GRU with dropout and layer normalization"""
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.rnn = nn.GRU(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            batch_first=True,
            num_layers=config.num_layers,
        )
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.linear = nn.Linear(config.hidden_size, config.output_size)

    def forward(self, x: Tensor) -> Tensor:
        x, _ = self.rnn(x)
        last_step = x[:, -1, :]
        last_step = self.layer_norm(last_step)
        last_step = self.dropout(last_step)
        yhat = self.linear(last_step)
        return yhat


# ============================================================================
# Training Utilities
# ============================================================================

def setup_data(batchsize: int = 32):
    """Setup gesture recognition data streams"""
    preprocessor = PaddedPreprocessor()
    gesturesdatasetfactory = DatasetFactoryProvider.create_factory(DatasetType.GESTURES)
    streamers = gesturesdatasetfactory.create_datastreamer(
        batchsize=batchsize, 
        preprocessor=preprocessor
    )
    return streamers["train"], streamers["valid"]


def setup_device():
    """Configure device - GPU if available, else CPU"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
    else:
        print("✓ Using CPU")
    return device


def calculate_accuracy(outputs: Tensor, targets: Tensor) -> float:
    """Calculate classification accuracy"""
    predictions = torch.argmax(outputs, dim=1)
    correct = (predictions == targets).sum().item()
    total = targets.size(0)
    return correct / total


def train_one_epoch(model, datastream, criterion, optimizer, device, steps_per_epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    dataloader = datastream.stream()
    progress_bar = tqdm(range(steps_per_epoch), desc="Training", leave=False)
    
    for _ in progress_bar:
        batch_x, batch_y = next(dataloader)
        
        # Move data to device
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item() * batch_x.size(0)
        total_correct += (torch.argmax(outputs, dim=1) == batch_y).sum().item()
        total_samples += batch_x.size(0)
        
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


def validate(model, datastream, criterion, device, steps_per_epoch):
    """Validate the model"""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    dataloader = datastream.stream()
    
    with torch.no_grad():
        for _ in tqdm(range(steps_per_epoch), desc="Validating", leave=False):
            batch_x, batch_y = next(dataloader)
            
            # Move data to device
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            # Forward pass
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            # Track metrics
            total_loss += loss.item() * batch_x.size(0)
            total_correct += (torch.argmax(outputs, dim=1) == batch_y).sum().item()
            total_samples += batch_x.size(0)
    
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


def run_experiment(
    model_class,
    config: ModelConfig,
    model_name: str,
    description: str,
    epochs: int = 20,
    learning_rate: float = 0.001
):
    """Run a single experiment with MLflow tracking and GPU support"""
    
    # Setup
    device = setup_device()
    train, valid = setup_data(batchsize=64)
    
    # Initialize model and move to device
    model = model_class(config=config)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Calculate steps per epoch
    train_steps = len(train)
    valid_steps = len(valid)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # MLflow tracking
    mlflow.set_tag("model_type", model_name)
    mlflow.set_tag("description", description)
    mlflow.log_params({
        "hidden_size": config.hidden_size,
        "num_layers": config.num_layers,
        "dropout": config.dropout,
        "epochs": epochs,
        "batch_size": 64,
        "learning_rate": learning_rate,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "device": str(device)
    })
    
    print(f"\n{'='*70}")
    print(f"Experiment: {model_name}")
    print(f"Description: {description}")
    print(f"Parameters: {total_params:,}")
    print(f"Config: hidden={config.hidden_size}, layers={config.num_layers}, dropout={config.dropout}")
    print(f"Device: {device}")
    print(f"{'='*70}\n")
    
    # Early stopping
    best_valid_acc = 0.0
    patience = 7
    patience_counter = 0
    
    # Training loop
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train, criterion, optimizer, device, train_steps
        )
        
        # Validate
        valid_loss, valid_acc = validate(
            model, valid, criterion, device, valid_steps
        )
        
        # Log metrics
        mlflow.log_metrics({
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "valid_loss": valid_loss,
            "valid_accuracy": valid_acc,
        }, step=epoch)
        
        # Scheduler step
        scheduler.step(valid_loss)
        
        # Print progress
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}")
        
        # Early stopping check
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    # Log final best accuracy
    mlflow.log_metric("best_valid_accuracy", best_valid_acc)
    
    print(f"\n✓ Completed: {model_name}")
    print(f"Best validation accuracy: {best_valid_acc:.4f}")
    
    return {"best_valid_acc": best_valid_acc}


# ============================================================================
# Experiment Definitions
# ============================================================================

def experiment_1_baseline():
    """Baseline GRU models with varying hidden sizes"""
    print("\n" + "="*70)
    print("EXPERIMENT 1: Baseline GRU - Hidden Size Exploration")
    print("="*70)
    
    configs = [
        (64, "Medium baseline - 64 units"),
        (128, "Large baseline - 128 units"),
        (256, "Extra large baseline - 256 units"),
    ]
    
    for hidden_size, desc in configs:
        with mlflow.start_run(run_name=f"baseline_gru_h{hidden_size}"):
            config = ModelConfig(
                input_size=3,
                hidden_size=hidden_size,
                num_layers=1,
                output_size=20,
                dropout=0.0
            )
            run_experiment(
                GRUModel,
                config,
                f"baseline_gru_h{hidden_size}",
                desc,
                epochs=20
            )


def experiment_2_depth():
    """Test impact of RNN depth"""
    print("\n" + "="*70)
    print("EXPERIMENT 2: GRU Depth - Number of Layers")
    print("="*70)
    
    configs = [
        (1, 128, "Single layer GRU"),
        (2, 128, "Two layer GRU"),
    ]
    
    for num_layers, hidden_size, desc in configs:
        with mlflow.start_run(run_name=f"gru_depth_l{num_layers}"):
            config = ModelConfig(
                input_size=3,
                hidden_size=hidden_size,
                num_layers=num_layers,
                output_size=20,
                dropout=0.2 if num_layers > 1 else 0.0
            )
            run_experiment(
                GRUModel,
                config,
                f"gru_depth_l{num_layers}",
                desc,
                epochs=20
            )


def experiment_3_dropout():
    """Test dropout regularization"""
    print("\n" + "="*70)
    print("EXPERIMENT 3: Dropout Regularization")
    print("="*70)
    
    dropout_rates = [0.0, 0.2, 0.3]
    
    for dropout in dropout_rates:
        with mlflow.start_run(run_name=f"gru_dropout_{dropout}"):
            config = ModelConfig(
                input_size=3,
                hidden_size=128,
                num_layers=2,
                output_size=20,
                dropout=dropout
            )
            run_experiment(
                GRUModel,
                config,
                f"gru_dropout_{dropout}",
                f"2-layer GRU with dropout={dropout}",
                epochs=20
            )


def experiment_4_lstm_vs_gru():
    """Compare LSTM and GRU architectures"""
    print("\n" + "="*70)
    print("EXPERIMENT 4: LSTM vs GRU Comparison")
    print("="*70)
    
    configs = [
        (GRUModel, "gru_128", "GRU with 128 hidden units"),
        (LSTMModel, "lstm_128", "LSTM with 128 hidden units"),
    ]
    
    for model_class, name, desc in configs:
        hidden_size = 128
        with mlflow.start_run(run_name=name):
            config = ModelConfig(
                input_size=3,
                hidden_size=hidden_size,
                num_layers=2,
                output_size=20,
                dropout=0.2
            )
            run_experiment(
                model_class,
                config,
                name,
                desc,
                epochs=20
            )


def experiment_5_bidirectional():
    """Test bidirectional RNNs"""
    print("\n" + "="*70)
    print("EXPERIMENT 5: Bidirectional GRU")
    print("="*70)
    
    configs = [
        (128, "Bidirectional GRU - 128 hidden units"),
    ]
    
    for hidden_size, desc in configs:
        with mlflow.start_run(run_name=f"bidirectional_gru_h{hidden_size}"):
            config = ModelConfig(
                input_size=3,
                hidden_size=hidden_size,
                num_layers=2,
                output_size=20,
                dropout=0.2
            )
            run_experiment(
                BidirectionalGRU,
                config,
                f"bidirectional_gru_h{hidden_size}",
                desc,
                epochs=25
            )


def experiment_6_conv_gru():
    """Test Conv1D + GRU hybrid"""
    print("\n" + "="*70)
    print("EXPERIMENT 6: Conv1D + GRU Hybrid")
    print("="*70)
    
    with mlflow.start_run(run_name="conv1d_gru"):
        config = ModelConfig(
            input_size=3,
            hidden_size=128,
            num_layers=2,
            output_size=20,
            dropout=0.2
        )
        run_experiment(
            Conv1DGRU,
            config,
            "conv1d_gru",
            "Conv1D feature extraction + GRU",
            epochs=30
        )


def experiment_7_deep_gru():
    """Test deep GRU with layer normalization"""
    print("\n" + "="*70)
    print("EXPERIMENT 7: Deep GRU with Layer Normalization")
    print("="*70)
    
    with mlflow.start_run(run_name="deep_gru"):
        config = ModelConfig(
            input_size=3,
            hidden_size=128,
            num_layers=3,
            output_size=20,
            dropout=0.3
        )
        run_experiment(
            DeepGRU,
            config,
            "deep_gru",
            "Deep GRU with LayerNorm and Dropout",
            epochs=25
        )


def experiment_8_optimal():
    """Test optimal configurations based on previous experiments"""
    print("\n" + "="*70)
    print("EXPERIMENT 8: Optimal Configuration")
    print("="*70)
    
    with mlflow.start_run(run_name="optimal_config"):
        config = ModelConfig(
            input_size=3,
            hidden_size=256,
            num_layers=2,
            output_size=20,
            dropout=0.2
        )
        run_experiment(
            BidirectionalGRU,
            config,
            "optimal_bidirectional_gru",
            "Optimal: Bidirectional GRU 256 hidden, 2 layers, dropout 0.2",
            epochs=40
        )


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Run all experiments"""
    
    # Setup MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("gesture_recognition_rnn")
    
    # Create output directory
    modeldir = Path("gestures").resolve()
    modeldir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("RNN HYPERPARAMETER TUNING - GESTURE RECOGNITION")
    print("="*70)
    print("Dataset: SmartWatch Gestures (20 classes)")
    print("Input: 3-axis accelerometer data (variable length sequences)")
    print("Goal: Achieve >90% validation accuracy")
    print("="*70)
    
    # Run experiments
    experiments = [
        ("1", "Baseline GRU - Hidden Sizes", experiment_1_baseline),
        ("2", "GRU Depth - Layer Count", experiment_2_depth),
        ("3", "Dropout Regularization", experiment_3_dropout),
        ("4", "LSTM vs GRU", experiment_4_lstm_vs_gru),
        ("5", "Bidirectional GRU", experiment_5_bidirectional),
        ("6", "Conv1D + GRU Hybrid", experiment_6_conv_gru),
        ("7", "Deep GRU + LayerNorm", experiment_7_deep_gru),
        ("8", "Optimal Configuration", experiment_8_optimal),
    ]
    
    print("\nExperiments to run:")
    for num, name, _ in experiments:
        print(f"  {num}. {name}")
    
    print("\n" + "="*70)
    print("Starting experiments...")
    print("="*70 + "\n")
    
    for num, name, experiment_func in experiments:
        try:
            experiment_func()
        except Exception as e:
            print(f"✗ Error in Experiment {num}: {e}")
            continue
    
    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETED!")
    print("="*70)
    print("\nView results: mlflow ui")
    print("Then open: http://localhost:5000\n")


if __name__ == "__main__":
    main()
