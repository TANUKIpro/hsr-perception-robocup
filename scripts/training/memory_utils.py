"""
Memory management utilities for training pipeline.

Provides comprehensive memory cleanup functions to prevent memory leaks
during the competition day training workflow.
"""

import gc
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class MemoryStats:
    """Memory statistics snapshot."""

    timestamp: datetime
    allocated_mb: float
    reserved_mb: float
    max_allocated_mb: float
    max_reserved_mb: float
    free_mb: float

    def __str__(self) -> str:
        return (
            f"Memory Stats [{self.timestamp.strftime('%H:%M:%S')}]: "
            f"Allocated={self.allocated_mb:.1f}MB, "
            f"Reserved={self.reserved_mb:.1f}MB, "
            f"Free={self.free_mb:.1f}MB"
        )


class MemoryTracker:
    """Track memory usage over time and generate reports."""

    def __init__(self):
        self.snapshots: List[MemoryStats] = []
        self.cuda_available = False
        try:
            import torch
            self.cuda_available = torch.cuda.is_available()
        except ImportError:
            pass

    def snapshot(self, label: str = "") -> Optional[MemoryStats]:
        """Take a memory snapshot.

        Args:
            label: Optional label for logging

        Returns:
            MemoryStats if CUDA available, None otherwise
        """
        if not self.cuda_available:
            return None

        import torch

        stats = MemoryStats(
            timestamp=datetime.now(),
            allocated_mb=torch.cuda.memory_allocated() / 1024**2,
            reserved_mb=torch.cuda.memory_reserved() / 1024**2,
            max_allocated_mb=torch.cuda.max_memory_allocated() / 1024**2,
            max_reserved_mb=torch.cuda.max_memory_reserved() / 1024**2,
            free_mb=(torch.cuda.get_device_properties(0).total_memory -
                     torch.cuda.memory_reserved()) / 1024**2
        )

        self.snapshots.append(stats)

        if label:
            logger.info(f"{label}: {stats}")

        return stats

    def reset_peak_stats(self):
        """Reset peak memory statistics."""
        if self.cuda_available:
            import torch
            torch.cuda.reset_peak_memory_stats()

    def generate_report(self) -> str:
        """Generate a memory usage report.

        Returns:
            Formatted report string
        """
        if not self.snapshots:
            return "No memory snapshots recorded"

        report_lines = [
            "\n" + "="*60,
            "Memory Usage Report",
            "="*60,
        ]

        for i, stats in enumerate(self.snapshots):
            report_lines.append(f"\nSnapshot {i+1}: {stats}")

        if len(self.snapshots) >= 2:
            first = self.snapshots[0]
            last = self.snapshots[-1]
            delta_allocated = last.allocated_mb - first.allocated_mb
            delta_reserved = last.reserved_mb - first.reserved_mb

            report_lines.extend([
                "\n" + "-"*60,
                "Memory Change (First -> Last):",
                f"  Allocated: {delta_allocated:+.1f}MB",
                f"  Reserved: {delta_reserved:+.1f}MB",
            ])

        max_snapshot = max(self.snapshots, key=lambda s: s.allocated_mb)
        report_lines.extend([
            "\n" + "-"*60,
            "Peak Usage:",
            f"  {max_snapshot}",
        ])

        report_lines.append("="*60 + "\n")

        return "\n".join(report_lines)

    def clear(self):
        """Clear all snapshots."""
        self.snapshots.clear()


def cleanup_cuda_memory(synchronize: bool = True) -> Dict[str, float]:
    """Clean up CUDA memory and return statistics.

    Args:
        synchronize: Whether to synchronize CUDA before cleanup

    Returns:
        Dictionary with memory statistics (before/after cleanup)
    """
    stats = {
        'before_mb': 0.0,
        'after_mb': 0.0,
        'freed_mb': 0.0
    }

    try:
        import torch

        if not torch.cuda.is_available():
            logger.debug("CUDA not available, skipping CUDA memory cleanup")
            return stats

        # Record before state
        stats['before_mb'] = torch.cuda.memory_allocated() / 1024**2

        # Synchronize if requested
        if synchronize:
            torch.cuda.synchronize()

        # Clear cache
        torch.cuda.empty_cache()

        # Record after state
        stats['after_mb'] = torch.cuda.memory_allocated() / 1024**2
        stats['freed_mb'] = stats['before_mb'] - stats['after_mb']

        logger.debug(
            f"CUDA memory cleanup: {stats['before_mb']:.1f}MB -> "
            f"{stats['after_mb']:.1f}MB (freed {stats['freed_mb']:.1f}MB)"
        )

    except ImportError:
        logger.debug("PyTorch not available, skipping CUDA cleanup")
    except Exception as e:
        logger.warning(f"Error during CUDA cleanup: {e}")

    return stats


def cleanup_model(model: Any) -> None:
    """Clean up model resources.

    Moves model to CPU, clears gradients, and deletes the model object.
    Safe to call even if model is None.

    Args:
        model: Model object to clean up (can be None)
    """
    if model is None:
        return

    try:
        import torch

        # Move to CPU to free GPU memory
        if hasattr(model, 'cpu'):
            model.cpu()

        # Clear gradients
        if hasattr(model, 'zero_grad'):
            try:
                model.zero_grad(set_to_none=True)
            except:
                try:
                    model.zero_grad()
                except:
                    pass

        # Clear model parameters
        if hasattr(model, 'parameters'):
            for param in model.parameters():
                if param.grad is not None:
                    param.grad = None

        logger.debug("Model cleanup completed")

    except ImportError:
        logger.debug("PyTorch not available for model cleanup")
    except Exception as e:
        logger.warning(f"Error during model cleanup: {e}")


def cleanup_optimizer(optimizer: Any) -> None:
    """Clear optimizer state.

    Safe to call even if optimizer is None.

    Args:
        optimizer: Optimizer object to clean up (can be None)
    """
    if optimizer is None:
        return

    try:
        # Clear optimizer state
        if hasattr(optimizer, 'state'):
            optimizer.state.clear()

        # Zero gradients if possible
        if hasattr(optimizer, 'zero_grad'):
            try:
                optimizer.zero_grad(set_to_none=True)
            except:
                try:
                    optimizer.zero_grad()
                except:
                    pass

        logger.debug("Optimizer cleanup completed")

    except Exception as e:
        logger.warning(f"Error during optimizer cleanup: {e}")


def cleanup_swa_model(swa_callback: Any) -> None:
    """Clean up SWA callback resources.

    Safe to call even if swa_callback is None.

    Args:
        swa_callback: SWA callback object to clean up (can be None)
    """
    if swa_callback is None:
        return

    try:
        # Call cleanup method if available
        if hasattr(swa_callback, 'cleanup'):
            swa_callback.cleanup()
            logger.debug("SWA callback cleanup completed")
        else:
            logger.debug("SWA callback has no cleanup method")

    except Exception as e:
        logger.warning(f"Error during SWA callback cleanup: {e}")


def cleanup_tensorboard(callback: Any, server: Any) -> None:
    """Clean up TensorBoard resources.

    Safe to call even if callback or server is None.

    Args:
        callback: TensorBoard callback object (can be None)
        server: TensorBoard server object (can be None)
    """
    # Clean up callback
    if callback is not None:
        try:
            if hasattr(callback, 'cleanup'):
                callback.cleanup()
                logger.debug("TensorBoard callback cleanup completed")
        except Exception as e:
            logger.warning(f"Error during TensorBoard callback cleanup: {e}")

    # Clean up server
    if server is not None:
        try:
            if hasattr(server, 'stop'):
                server.stop()
                logger.debug("TensorBoard server stopped")
        except Exception as e:
            logger.warning(f"Error stopping TensorBoard server: {e}")


def full_training_cleanup(
    model: Any = None,
    optimizer: Any = None,
    swa_callback: Any = None,
    tensorboard_callback: Any = None,
    tensorboard_server: Any = None,
    extra_objects: Optional[List[Any]] = None,
    synchronize_cuda: bool = True,
    num_gc_passes: int = 2
) -> Dict[str, Any]:
    """Comprehensive cleanup of all training resources.

    This is the main entry point for cleaning up after training.
    Cleans up models, optimizers, callbacks, and runs garbage collection.

    Args:
        model: Model object to clean up
        optimizer: Optimizer object to clean up
        swa_callback: SWA callback to clean up
        tensorboard_callback: TensorBoard callback to clean up
        tensorboard_server: TensorBoard server to stop
        extra_objects: Additional objects to delete
        synchronize_cuda: Whether to synchronize CUDA before cleanup
        num_gc_passes: Number of garbage collection passes (default: 2)

    Returns:
        Dictionary with cleanup statistics
    """
    logger.info("Starting full training cleanup...")

    stats = {
        'cuda_before_mb': 0.0,
        'cuda_after_mb': 0.0,
        'cuda_freed_mb': 0.0,
        'objects_cleaned': []
    }

    try:
        import torch
        if torch.cuda.is_available():
            stats['cuda_before_mb'] = torch.cuda.memory_allocated() / 1024**2
    except:
        pass

    # Clean up in specific order
    cleanup_order = [
        ('optimizer', optimizer, cleanup_optimizer),
        ('swa_callback', swa_callback, cleanup_swa_model),
        ('tensorboard', (tensorboard_callback, tensorboard_server),
         lambda x: cleanup_tensorboard(x[0], x[1])),
        ('model', model, cleanup_model),
    ]

    for name, obj, cleanup_func in cleanup_order:
        if obj is not None:
            try:
                cleanup_func(obj)
                stats['objects_cleaned'].append(name)
            except Exception as e:
                logger.warning(f"Error cleaning {name}: {e}")

    # Clean up extra objects
    if extra_objects:
        for i, obj in enumerate(extra_objects):
            try:
                del obj
                stats['objects_cleaned'].append(f'extra_{i}')
            except:
                pass

    # Run garbage collection multiple times for thorough cleanup
    for i in range(num_gc_passes):
        collected = gc.collect()
        logger.debug(f"GC pass {i+1}/{num_gc_passes}: collected {collected} objects")

    # Clean up CUDA memory
    cuda_stats = cleanup_cuda_memory(synchronize=synchronize_cuda)
    stats.update({
        'cuda_after_mb': cuda_stats['after_mb'],
        'cuda_freed_mb': cuda_stats['freed_mb']
    })

    logger.info(
        f"Full cleanup completed. Cleaned: {', '.join(stats['objects_cleaned'])}. "
        f"CUDA freed: {stats['cuda_freed_mb']:.1f}MB"
    )

    return stats


def log_memory_snapshot(label: str = "Memory") -> Optional[MemoryStats]:
    """Quick utility to log current memory state.

    Args:
        label: Label for the log message

    Returns:
        MemoryStats if CUDA available, None otherwise
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return None

        stats = MemoryStats(
            timestamp=datetime.now(),
            allocated_mb=torch.cuda.memory_allocated() / 1024**2,
            reserved_mb=torch.cuda.memory_reserved() / 1024**2,
            max_allocated_mb=torch.cuda.max_memory_allocated() / 1024**2,
            max_reserved_mb=torch.cuda.max_memory_reserved() / 1024**2,
            free_mb=(torch.cuda.get_device_properties(0).total_memory -
                     torch.cuda.memory_reserved()) / 1024**2
        )

        logger.info(f"{label}: {stats}")
        return stats

    except ImportError:
        return None
    except Exception as e:
        logger.warning(f"Error logging memory snapshot: {e}")
        return None
