# genuity/utils/device.py

import torch
import os
from typing import Optional


def get_device() -> torch.device:
    """Get the best available device for PyTorch operations."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")  # Apple Silicon
    else:
        return torch.device("cpu")


def get_device_info() -> dict:
    """Get detailed device information."""
    device = get_device()
    info = {
        "device": str(device),
        "device_type": device.type,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": (
            torch.cuda.device_count() if torch.cuda.is_available() else 0
        ),
    }

    if device.type == "cuda":
        info.update(
            {
                "cuda_device_name": torch.cuda.get_device_name(device),
                "cuda_memory_allocated": torch.cuda.memory_allocated(device),
                "cuda_memory_reserved": torch.cuda.memory_reserved(device),
            }
        )

    return info


def clear_gpu_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def set_device_preference(preference: str = "auto") -> torch.device:
    """Set device preference for the library."""
    if preference == "cpu":
        return torch.device("cpu")
    elif preference == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif (
        preference == "mps"
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")
    else:
        return get_device()
