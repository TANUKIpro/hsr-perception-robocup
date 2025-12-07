"""
HSR Perception - Device Utilities

CUDA/GPU availability checking and device management utilities.
Consolidates device-related code from quick_finetune.py and app/config.py.
"""

from typing import Optional, Tuple

from colorama import Fore, Style


def check_cuda_available() -> Tuple[bool, Optional[str], Optional[float]]:
    """
    Check CUDA availability and return GPU information.

    Returns:
        Tuple of (available, gpu_name, memory_gb):
            - available: True if CUDA is available
            - gpu_name: Name of the GPU (None if not available)
            - memory_gb: Total GPU memory in GB (None if not available)
    """
    try:
        import torch

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            return True, gpu_name, gpu_memory
        else:
            return False, None, None
    except ImportError:
        return False, None, None


def get_default_device() -> str:
    """
    Get the default device for PyTorch operations.

    Returns:
        'cuda' if CUDA is available, 'cpu' otherwise
    """
    available, _, _ = check_cuda_available()
    return "cuda" if available else "cpu"


def log_gpu_status(verbose: bool = True) -> bool:
    """
    Log GPU status with colored output.

    Args:
        verbose: If True, print status messages

    Returns:
        True if GPU is available, False otherwise
    """
    available, gpu_name, gpu_memory = check_cuda_available()

    if verbose:
        if available:
            print(
                f"{Fore.GREEN}GPU Available: {gpu_name} "
                f"({gpu_memory:.1f} GB){Style.RESET_ALL}"
            )
        else:
            print(
                f"{Fore.YELLOW}Warning: CUDA not available. "
                f"Training will be slow.{Style.RESET_ALL}"
            )

    return available


def get_gpu_info() -> dict:
    """
    Get detailed GPU information as a dictionary.

    Returns:
        Dictionary containing GPU information:
            - available: bool
            - device_count: int (number of GPUs)
            - devices: list of dicts with name and memory for each GPU
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return {"available": False, "device_count": 0, "devices": []}

        device_count = torch.cuda.device_count()
        devices = []

        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            devices.append({
                "index": i,
                "name": props.name,
                "total_memory_gb": props.total_memory / 1e9,
                "major": props.major,
                "minor": props.minor,
                "multi_processor_count": props.multi_processor_count,
            })

        return {
            "available": True,
            "device_count": device_count,
            "devices": devices,
        }
    except ImportError:
        return {"available": False, "device_count": 0, "devices": [], "error": "PyTorch not installed"}


def get_optimal_batch_size(
    base_batch: int = 16,
    min_memory_gb: float = 8.0,
    scale_factor: float = 0.5,
) -> int:
    """
    Get optimal batch size based on available GPU memory.

    Args:
        base_batch: Base batch size for min_memory_gb
        min_memory_gb: Minimum memory for base_batch
        scale_factor: How much to scale batch size per doubling of memory

    Returns:
        Recommended batch size
    """
    available, _, memory_gb = check_cuda_available()

    if not available or memory_gb is None:
        return base_batch // 2  # Use smaller batch for CPU

    # Scale batch size based on memory
    memory_ratio = memory_gb / min_memory_gb
    if memory_ratio >= 2.0:
        return int(base_batch * (1 + scale_factor * (memory_ratio - 1)))
    elif memory_ratio < 1.0:
        return max(4, int(base_batch * memory_ratio))
    else:
        return base_batch
