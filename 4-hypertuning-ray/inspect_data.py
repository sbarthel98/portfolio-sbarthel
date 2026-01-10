
from mads_datasets import DatasetFactoryProvider, DatasetType
from pathlib import Path
import torch

try:
    factory = DatasetFactoryProvider.create_factory(DatasetType.FLOWERS)
    streamers = factory.create_datastreamer(batchsize=1)
    train = streamers["train"]
    features, labels = next(iter(train.stream()))
    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
    # Inspect number of classes roughly or if possible
    print("Dataset loaded successfully")
except Exception as e:
    print(f"Error: {e}")
