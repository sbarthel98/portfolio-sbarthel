
from mads_datasets import DatasetFactoryProvider, DatasetType
from pathlib import Path
import torch

try:
    factory = DatasetFactoryProvider.create_factory(DatasetType.FLOWERS)
    streamers = factory.create_datastreamer(batchsize=1)
    train = streamers["train"]
    item = next(iter(train.stream()))
    print(f"Type of item: {type(item)}")
    if isinstance(item, (list, tuple)):
        print(f"Length: {len(item)}")
        for i, sub in enumerate(item):
            if hasattr(sub, "shape"):
                print(f"Item {i} shape: {sub.shape}")
            else:
                print(f"Item {i}: {sub}")
    
    print("Dataset loaded successfully")
except Exception as e:
    print(f"Error: {e}")
