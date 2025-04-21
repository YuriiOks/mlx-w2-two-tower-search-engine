# utils/device_setup.py
# Copyright (c) 2025 Perceptron Party Team (Yurii, Pyry, Dimitris, Dimitar)
# Description: Detects and sets the appropriate PyTorch device.
# Created: 2025-04-21

import torch
import os
from .logging import logger

def get_device() -> torch.device:
    '''Checks for MPS, CUDA, or CPU and returns the device.'''
    selected_device = None
    logger.debug("⚙️  Checking for hardware accelerators...")
    if torch.backends.mps.is_available():
        selected_device = torch.device("mps")
        logger.info("✅ MPS device found. Using MPS.")
    elif torch.cuda.is_available():
        selected_device = torch.device("cuda")
        logger.info(f"✅ CUDA device found. Using CUDA ({torch.cuda.get_device_name(0)}).")
    else:
        selected_device = torch.device("cpu")
        logger.warning("⚠️ MPS/CUDA not available. Using CPU.")
    logger.info(f"✨ Selected compute device: {selected_device.type.upper()}")
    return selected_device

if __name__ == '__main__':
    logger.info("🚀 Running device setup check directly...")
    device = get_device()
    logger.info(f"🔍 Device object returned: {device}")
    try:
        x = torch.randn(2, 2, device=device)
        logger.info(f"✅ Test tensor created successfully on {device}.")
    except Exception as e: logger.error(f"❌ Failed test tensor on {device}: {e}")
