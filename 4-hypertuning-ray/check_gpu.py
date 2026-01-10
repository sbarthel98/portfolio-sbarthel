"""
Quick check of GPU availability and configuration for Ray Tune experiments.
"""

import torch
from loguru import logger

def check_gpu():
    """Check GPU availability and configuration."""
    
    logger.info("="*60)
    logger.info("GPU Configuration Check")
    logger.info("="*60)
    
    cuda_available = torch.cuda.is_available()
    logger.info(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        device_count = torch.cuda.device_count()
        logger.info(f"GPU Count: {device_count}")
        
        for i in range(device_count):
            logger.info(f"\nGPU {i}:")
            logger.info(f"  Name: {torch.cuda.get_device_name(i)}")
            
            # Get GPU memory info
            total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            logger.info(f"  Total Memory: {total_memory:.2f} GB")
            
            # Current allocation
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            logger.info(f"  Allocated: {allocated:.2f} GB")
            logger.info(f"  Reserved: {reserved:.2f} GB")
            logger.info(f"  Free: {total_memory - reserved:.2f} GB")
        
        logger.info(f"\nCUDA Version: {torch.version.cuda}")
        logger.info(f"cuDNN Version: {torch.backends.cudnn.version()}")
        
        logger.info("\n✅ GPU is available and ready for Ray Tune!")
        logger.info("   Ray Tune will use 0.5 GPU per trial (2 parallel trials)")
        
    else:
        logger.warning("⚠️  No GPU detected - training will use CPU")
        logger.warning("   This will be significantly slower!")
        logger.info("\nTo use GPU:")
        logger.info("  1. Ensure you have a CUDA-capable GPU")
        logger.info("  2. Install CUDA toolkit")
        logger.info("  3. Install PyTorch with CUDA support:")
        logger.info("     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    
    logger.info("="*60)

if __name__ == "__main__":
    check_gpu()
