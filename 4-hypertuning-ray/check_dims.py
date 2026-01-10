
from mads_datasets import DatasetFactoryProvider, DatasetType
import torch

from torch.utils.data import DataLoader

try:
    factory = DatasetFactoryProvider.create_factory(DatasetType.FLOWERS)
    streamers = factory.create_datastreamer(batchsize=32)
    dataset = streamers["train"].dataset
    
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    item = next(iter(loader))
    
    print(f"Loader item type: {type(item)}")
    if isinstance(item, (list, tuple)):
        print(f"Tuple length: {len(item)}")
        print(f"X shape: {item[0].shape}")
        print(f"Y shape: {item[1].shape}")
        
except Exception as e:
    print(f"Error: {e}")

