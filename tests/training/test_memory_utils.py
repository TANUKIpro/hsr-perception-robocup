"""
Tests for memory management utilities.

Tests the memory cleanup functions to ensure they prevent memory leaks
during the training pipeline.
"""

import gc
import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

# Add scripts directory to path
scripts_dir = Path(__file__).parent.parent.parent / "scripts"
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

from training.memory_utils import (
    MemoryStats,
    MemoryTracker,
    cleanup_cuda_memory,
    cleanup_model,
    cleanup_optimizer,
    cleanup_swa_model,
    cleanup_tensorboard,
    full_training_cleanup,
    log_memory_snapshot,
)


class TestMemoryStats:
    """Test MemoryStats dataclass."""

    def test_memory_stats_creation(self):
        """Test creating MemoryStats instance."""
        from datetime import datetime

        stats = MemoryStats(
            timestamp=datetime.now(),
            allocated_mb=100.0,
            reserved_mb=150.0,
            max_allocated_mb=120.0,
            max_reserved_mb=180.0,
            free_mb=200.0,
        )

        assert stats.allocated_mb == 100.0
        assert stats.reserved_mb == 150.0
        assert stats.max_allocated_mb == 120.0
        assert stats.max_reserved_mb == 180.0
        assert stats.free_mb == 200.0

    def test_memory_stats_string(self):
        """Test MemoryStats string representation."""
        from datetime import datetime

        stats = MemoryStats(
            timestamp=datetime.now(),
            allocated_mb=100.5,
            reserved_mb=150.3,
            max_allocated_mb=120.7,
            max_reserved_mb=180.9,
            free_mb=200.1,
        )

        stats_str = str(stats)
        assert "100.5MB" in stats_str
        assert "150.3MB" in stats_str
        assert "200.1MB" in stats_str


class TestMemoryTracker:
    """Test MemoryTracker class."""

    def test_memory_tracker_init(self):
        """Test MemoryTracker initialization."""
        tracker = MemoryTracker()
        assert isinstance(tracker.snapshots, list)
        assert len(tracker.snapshots) == 0

    def test_memory_tracker_snapshot_no_cuda(self):
        """Test snapshot when CUDA is not available."""
        tracker = MemoryTracker()
        tracker.cuda_available = False

        result = tracker.snapshot("test")
        assert result is None
        assert len(tracker.snapshots) == 0

    @patch("training.memory_utils.torch")
    def test_memory_tracker_snapshot_with_cuda(self, mock_torch):
        """Test snapshot when CUDA is available."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = 100 * 1024**2
        mock_torch.cuda.memory_reserved.return_value = 150 * 1024**2
        mock_torch.cuda.max_memory_allocated.return_value = 120 * 1024**2
        mock_torch.cuda.max_memory_reserved.return_value = 180 * 1024**2

        # Mock device properties
        mock_props = Mock()
        mock_props.total_memory = 1000 * 1024**2
        mock_torch.cuda.get_device_properties.return_value = mock_props

        tracker = MemoryTracker()
        tracker.cuda_available = True

        result = tracker.snapshot("test")

        assert result is not None
        assert len(tracker.snapshots) == 1
        assert result.allocated_mb == pytest.approx(100.0, rel=0.1)
        assert result.reserved_mb == pytest.approx(150.0, rel=0.1)

    def test_memory_tracker_clear(self):
        """Test clearing snapshots."""
        tracker = MemoryTracker()
        from datetime import datetime

        # Add some dummy snapshots
        tracker.snapshots.append(
            MemoryStats(
                timestamp=datetime.now(),
                allocated_mb=100.0,
                reserved_mb=150.0,
                max_allocated_mb=120.0,
                max_reserved_mb=180.0,
                free_mb=200.0,
            )
        )

        assert len(tracker.snapshots) == 1
        tracker.clear()
        assert len(tracker.snapshots) == 0

    def test_memory_tracker_generate_report_empty(self):
        """Test generating report with no snapshots."""
        tracker = MemoryTracker()
        report = tracker.generate_report()

        assert "No memory snapshots recorded" in report

    def test_memory_tracker_generate_report_with_snapshots(self):
        """Test generating report with snapshots."""
        from datetime import datetime

        tracker = MemoryTracker()
        tracker.snapshots.append(
            MemoryStats(
                timestamp=datetime.now(),
                allocated_mb=100.0,
                reserved_mb=150.0,
                max_allocated_mb=120.0,
                max_reserved_mb=180.0,
                free_mb=200.0,
            )
        )
        tracker.snapshots.append(
            MemoryStats(
                timestamp=datetime.now(),
                allocated_mb=200.0,
                reserved_mb=250.0,
                max_allocated_mb=220.0,
                max_reserved_mb=280.0,
                free_mb=100.0,
            )
        )

        report = tracker.generate_report()

        assert "Memory Usage Report" in report
        assert "Snapshot 1" in report
        assert "Snapshot 2" in report
        assert "Memory Change" in report
        assert "Peak Usage" in report


class TestCleanupCudaMemory:
    """Test cleanup_cuda_memory function."""

    @patch("training.memory_utils.torch")
    def test_cleanup_cuda_memory_no_cuda(self, mock_torch):
        """Test cleanup when CUDA is not available."""
        mock_torch.cuda.is_available.return_value = False

        stats = cleanup_cuda_memory()

        assert stats["before_mb"] == 0.0
        assert stats["after_mb"] == 0.0
        assert stats["freed_mb"] == 0.0

    @patch("training.memory_utils.torch")
    def test_cleanup_cuda_memory_with_cuda(self, mock_torch):
        """Test cleanup when CUDA is available."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.side_effect = [
            100 * 1024**2,  # before
            50 * 1024**2,  # after
        ]

        stats = cleanup_cuda_memory(synchronize=True)

        assert stats["before_mb"] == pytest.approx(100.0, rel=0.1)
        assert stats["after_mb"] == pytest.approx(50.0, rel=0.1)
        assert stats["freed_mb"] == pytest.approx(50.0, rel=0.1)

        mock_torch.cuda.synchronize.assert_called_once()
        mock_torch.cuda.empty_cache.assert_called_once()

    @patch("training.memory_utils.torch")
    def test_cleanup_cuda_memory_no_synchronize(self, mock_torch):
        """Test cleanup without synchronization."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.side_effect = [
            100 * 1024**2,
            50 * 1024**2,
        ]

        stats = cleanup_cuda_memory(synchronize=False)

        mock_torch.cuda.synchronize.assert_not_called()
        mock_torch.cuda.empty_cache.assert_called_once()


class TestCleanupModel:
    """Test cleanup_model function."""

    def test_cleanup_model_none(self):
        """Test cleanup with None model."""
        # Should not raise an error
        cleanup_model(None)

    @patch("training.memory_utils.torch")
    def test_cleanup_model_with_model(self, mock_torch):
        """Test cleanup with actual model."""
        mock_model = Mock()
        mock_param = Mock()
        mock_param.grad = Mock()
        mock_model.parameters.return_value = [mock_param]

        cleanup_model(mock_model)

        mock_model.cpu.assert_called_once()
        mock_model.zero_grad.assert_called()
        assert mock_param.grad is None

    @patch("training.memory_utils.torch")
    def test_cleanup_model_no_grad(self, mock_torch):
        """Test cleanup when model has no gradients."""
        mock_model = Mock()
        mock_model.zero_grad.side_effect = AttributeError()

        # Should not raise an error
        cleanup_model(mock_model)

        mock_model.cpu.assert_called_once()


class TestCleanupOptimizer:
    """Test cleanup_optimizer function."""

    def test_cleanup_optimizer_none(self):
        """Test cleanup with None optimizer."""
        # Should not raise an error
        cleanup_optimizer(None)

    def test_cleanup_optimizer_with_optimizer(self):
        """Test cleanup with actual optimizer."""
        mock_optimizer = Mock()
        mock_optimizer.state = {"param1": {}, "param2": {}}

        cleanup_optimizer(mock_optimizer)

        mock_optimizer.state.clear.assert_called_once()
        mock_optimizer.zero_grad.assert_called()

    def test_cleanup_optimizer_no_zero_grad(self):
        """Test cleanup when optimizer has no zero_grad."""
        mock_optimizer = Mock()
        mock_optimizer.state = {}
        mock_optimizer.zero_grad.side_effect = AttributeError()

        # Should not raise an error
        cleanup_optimizer(mock_optimizer)


class TestCleanupSwaModel:
    """Test cleanup_swa_model function."""

    def test_cleanup_swa_model_none(self):
        """Test cleanup with None SWA callback."""
        # Should not raise an error
        cleanup_swa_model(None)

    def test_cleanup_swa_model_with_cleanup(self):
        """Test cleanup with SWA callback that has cleanup method."""
        mock_callback = Mock()
        mock_callback.cleanup = Mock()

        cleanup_swa_model(mock_callback)

        mock_callback.cleanup.assert_called_once()

    def test_cleanup_swa_model_no_cleanup(self):
        """Test cleanup with SWA callback without cleanup method."""
        mock_callback = Mock(spec=[])  # No cleanup method

        # Should not raise an error
        cleanup_swa_model(mock_callback)


class TestCleanupTensorboard:
    """Test cleanup_tensorboard function."""

    def test_cleanup_tensorboard_none_callback(self):
        """Test cleanup with None callback."""
        # Should not raise an error
        cleanup_tensorboard(None, None)

    def test_cleanup_tensorboard_none_server(self):
        """Test cleanup with None server."""
        mock_callback = Mock()
        mock_callback.cleanup = Mock()

        cleanup_tensorboard(mock_callback, None)

        mock_callback.cleanup.assert_called_once()

    def test_cleanup_tensorboard_both(self):
        """Test cleanup with both callback and server."""
        mock_callback = Mock()
        mock_callback.cleanup = Mock()
        mock_server = Mock()
        mock_server.stop = Mock()

        cleanup_tensorboard(mock_callback, mock_server)

        mock_callback.cleanup.assert_called_once()
        mock_server.stop.assert_called_once()


class TestFullTrainingCleanup:
    """Test full_training_cleanup function."""

    @patch("training.memory_utils.cleanup_cuda_memory")
    @patch("training.memory_utils.gc")
    def test_full_training_cleanup_all_none(self, mock_gc, mock_cuda_cleanup):
        """Test cleanup when all resources are None."""
        mock_cuda_cleanup.return_value = {
            "before_mb": 0.0,
            "after_mb": 0.0,
            "freed_mb": 0.0,
        }
        mock_gc.collect.return_value = 0

        stats = full_training_cleanup()

        assert stats["cuda_freed_mb"] == 0.0
        assert stats["objects_cleaned"] == []
        assert mock_gc.collect.call_count == 2  # num_gc_passes=2

    @patch("training.memory_utils.cleanup_cuda_memory")
    @patch("training.memory_utils.cleanup_model")
    @patch("training.memory_utils.cleanup_optimizer")
    @patch("training.memory_utils.gc")
    def test_full_training_cleanup_with_resources(
        self, mock_gc, mock_cleanup_optimizer, mock_cleanup_model, mock_cuda_cleanup
    ):
        """Test cleanup with actual resources."""
        mock_model = Mock()
        mock_optimizer = Mock()
        mock_cuda_cleanup.return_value = {
            "before_mb": 100.0,
            "after_mb": 50.0,
            "freed_mb": 50.0,
        }
        mock_gc.collect.return_value = 10

        stats = full_training_cleanup(
            model=mock_model, optimizer=mock_optimizer, num_gc_passes=3
        )

        assert "model" in stats["objects_cleaned"]
        assert "optimizer" in stats["objects_cleaned"]
        assert stats["cuda_freed_mb"] == 50.0
        assert mock_gc.collect.call_count == 3

        mock_cleanup_model.assert_called_once_with(mock_model)
        mock_cleanup_optimizer.assert_called_once_with(mock_optimizer)

    @patch("training.memory_utils.cleanup_cuda_memory")
    @patch("training.memory_utils.cleanup_swa_model")
    @patch("training.memory_utils.cleanup_tensorboard")
    @patch("training.memory_utils.gc")
    def test_full_training_cleanup_with_callbacks(
        self, mock_gc, mock_cleanup_tb, mock_cleanup_swa, mock_cuda_cleanup
    ):
        """Test cleanup with SWA and TensorBoard callbacks."""
        mock_swa = Mock()
        mock_tb_callback = Mock()
        mock_tb_server = Mock()
        mock_cuda_cleanup.return_value = {
            "before_mb": 0.0,
            "after_mb": 0.0,
            "freed_mb": 0.0,
        }
        mock_gc.collect.return_value = 0

        stats = full_training_cleanup(
            swa_callback=mock_swa,
            tensorboard_callback=mock_tb_callback,
            tensorboard_server=mock_tb_server,
        )

        assert "swa_callback" in stats["objects_cleaned"]
        assert "tensorboard" in stats["objects_cleaned"]

        mock_cleanup_swa.assert_called_once_with(mock_swa)
        mock_cleanup_tb.assert_called_once_with(mock_tb_callback, mock_tb_server)


class TestLogMemorySnapshot:
    """Test log_memory_snapshot function."""

    @patch("training.memory_utils.torch")
    def test_log_memory_snapshot_no_cuda(self, mock_torch):
        """Test logging snapshot when CUDA is not available."""
        mock_torch.cuda.is_available.return_value = False

        result = log_memory_snapshot("test")

        assert result is None

    @patch("training.memory_utils.torch")
    def test_log_memory_snapshot_with_cuda(self, mock_torch):
        """Test logging snapshot when CUDA is available."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = 100 * 1024**2
        mock_torch.cuda.memory_reserved.return_value = 150 * 1024**2
        mock_torch.cuda.max_memory_allocated.return_value = 120 * 1024**2
        mock_torch.cuda.max_memory_reserved.return_value = 180 * 1024**2

        mock_props = Mock()
        mock_props.total_memory = 1000 * 1024**2
        mock_torch.cuda.get_device_properties.return_value = mock_props

        result = log_memory_snapshot("test")

        assert result is not None
        assert result.allocated_mb == pytest.approx(100.0, rel=0.1)
        assert result.reserved_mb == pytest.approx(150.0, rel=0.1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
