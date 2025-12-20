"""
Tests for training configuration management.

Tests the TrainingConfig dataclass and related configuration classes
for type safety and GPU-aware presets.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

# Add scripts directory to path
scripts_dir = Path(__file__).parent.parent.parent.parent / "scripts"
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

from training.training_config import (
    AugmentationConfig,
    OptimizerConfig,
    PerformanceConfig,
    CheckpointConfig,
    TrainingConfig,
    get_competition_config,
    get_fast_config,
)


class TestAugmentationConfig:
    """Test AugmentationConfig dataclass."""

    def test_default_values(self):
        """Test default augmentation values."""
        config = AugmentationConfig()

        assert config.hsv_h == 0.015
        assert config.hsv_s == 0.7
        assert config.hsv_v == 0.4
        assert config.degrees == 10.0
        assert config.translate == 0.1
        assert config.scale == 0.5
        assert config.shear == 2.0
        assert config.flipud == 0.0
        assert config.fliplr == 0.5
        assert config.mosaic == 1.0
        assert config.mixup == 0.1

    def test_hsv_ranges(self):
        """Test HSV value customization."""
        config = AugmentationConfig(hsv_h=0.02, hsv_s=0.5, hsv_v=0.3)

        assert config.hsv_h == 0.02
        assert config.hsv_s == 0.5
        assert config.hsv_v == 0.3

    def test_geometric_transforms(self):
        """Test geometric transformation parameters."""
        config = AugmentationConfig(
            degrees=15.0,
            translate=0.2,
            scale=0.7,
            shear=5.0,
        )

        assert config.degrees == 15.0
        assert config.translate == 0.2
        assert config.scale == 0.7
        assert config.shear == 5.0

    def test_mosaic_mixup(self):
        """Test Mosaic and MixUp augmentation settings."""
        config = AugmentationConfig(mosaic=0.8, mixup=0.2)

        assert config.mosaic == 0.8
        assert config.mixup == 0.2

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = AugmentationConfig()
        result = config.to_dict()

        assert isinstance(result, dict)
        assert "hsv_h" in result
        assert "hsv_s" in result
        assert "hsv_v" in result
        assert "mosaic" in result
        assert "mixup" in result
        assert result["hsv_h"] == 0.015


class TestOptimizerConfig:
    """Test OptimizerConfig dataclass."""

    def test_default_values(self):
        """Test default optimizer values."""
        config = OptimizerConfig()

        assert config.optimizer == "AdamW"
        assert config.lr0 == 0.001
        assert config.lrf == 0.01
        assert config.momentum == 0.937
        assert config.weight_decay == 0.0005

    def test_llrd_settings(self):
        """Test Layer-wise Learning Rate Decay settings."""
        config = OptimizerConfig(llrd_enabled=True, llrd_decay_rate=0.85)

        assert config.llrd_enabled is True
        assert config.llrd_decay_rate == 0.85

    def test_swa_settings(self):
        """Test Stochastic Weight Averaging settings."""
        config = OptimizerConfig(
            swa_enabled=True,
            swa_start_epoch=10,
            swa_lr=0.0003,
        )

        assert config.swa_enabled is True
        assert config.swa_start_epoch == 10
        assert config.swa_lr == 0.0003

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = OptimizerConfig()
        result = config.to_dict()

        assert isinstance(result, dict)
        assert "optimizer" in result
        assert "lr0" in result
        assert "llrd_enabled" in result
        assert "swa_enabled" in result


class TestPerformanceConfig:
    """Test PerformanceConfig dataclass."""

    def test_default_values(self):
        """Test default performance values."""
        config = PerformanceConfig()

        assert config.workers == 8
        assert config.cache is True
        assert config.amp is True

    def test_worker_count(self):
        """Test custom worker count."""
        config = PerformanceConfig(workers=4)

        assert config.workers == 4

    def test_amp_setting(self):
        """Test Automatic Mixed Precision setting."""
        config = PerformanceConfig(amp=False)

        assert config.amp is False

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = PerformanceConfig()
        result = config.to_dict()

        assert isinstance(result, dict)
        assert result["workers"] == 8
        assert result["cache"] is True
        assert result["amp"] is True


class TestCheckpointConfig:
    """Test CheckpointConfig dataclass."""

    def test_default_values(self):
        """Test default checkpoint values."""
        config = CheckpointConfig()

        assert config.save is True
        assert config.save_period == 5
        assert config.exist_ok is True

    def test_save_period(self):
        """Test save period customization."""
        config = CheckpointConfig(save_period=10)

        assert config.save_period == 10

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = CheckpointConfig()
        result = config.to_dict()

        assert isinstance(result, dict)
        assert result["save"] is True
        assert result["save_period"] == 5


class TestTrainingConfig:
    """Test TrainingConfig dataclass."""

    def test_default_configuration(self):
        """Test default training configuration."""
        config = TrainingConfig()

        assert config.model == "yolov8m.pt"
        assert config.imgsz == 640
        assert config.epochs == 50
        assert config.batch == 16
        assert config.patience == 10
        assert config.close_mosaic == 10

    def test_custom_values(self):
        """Test custom configuration values."""
        config = TrainingConfig(
            model="yolov8s.pt",
            imgsz=480,
            epochs=30,
            batch=32,
        )

        assert config.model == "yolov8s.pt"
        assert config.imgsz == 480
        assert config.epochs == 30
        assert config.batch == 32

    def test_to_dict(self):
        """Test conversion to dictionary for YOLO training."""
        config = TrainingConfig()
        result = config.to_dict()

        assert isinstance(result, dict)
        assert result["model"] == "yolov8m.pt"
        assert result["imgsz"] == 640
        assert result["epochs"] == 50
        assert result["batch"] == 16
        # Check merged sub-configurations
        assert "hsv_h" in result  # From AugmentationConfig
        assert "optimizer" in result  # From OptimizerConfig
        assert "workers" in result  # From PerformanceConfig
        assert "save_period" in result  # From CheckpointConfig

    def test_from_dict(self):
        """Test creation from dictionary."""
        config_dict = {
            "model": "yolov8l.pt",
            "imgsz": 800,
            "epochs": 100,
            "batch": 8,
            "patience": 20,
            "hsv_h": 0.02,
            "optimizer": "SGD",
            "workers": 4,
            "save_period": 10,
        }

        config = TrainingConfig.from_dict(config_dict)

        assert config.model == "yolov8l.pt"
        assert config.imgsz == 800
        assert config.epochs == 100
        assert config.batch == 8
        assert config.patience == 20
        assert config.augmentation.hsv_h == 0.02
        assert config.optimizer.optimizer == "SGD"
        assert config.performance.workers == 4
        assert config.checkpoint.save_period == 10

    def test_competition_default(self):
        """Test competition-optimized preset."""
        config = TrainingConfig.competition_default()

        assert config.model == "yolov8m.pt"
        assert config.imgsz == 640
        assert config.epochs == 50
        assert config.batch == 16
        assert config.patience == 10

    def test_fast_test(self):
        """Test fast testing preset."""
        config = TrainingConfig.fast_test()

        assert config.model == "yolov8s.pt"
        assert config.imgsz == 480
        assert config.epochs == 30
        assert config.batch == 32
        assert config.patience == 5
        assert config.performance.workers == 4

    @patch("training.training_config.GPUScaler")
    def test_from_gpu_profile(self, mock_gpu_scaler_class):
        """Test creation from GPU profile."""
        from training.gpu_scaler import GPUProfile, GPUTier

        # Mock GPUScaler
        mock_scaler = MagicMock()
        mock_scaler.get_optimal_config.return_value = {
            "model": "yolov8m.pt",
            "imgsz": 640,
            "epochs": 50,
            "batch": 16,
        }
        mock_gpu_scaler_class.return_value = mock_scaler

        profile = GPUProfile(
            name="Test GPU",
            device_index=0,
            total_memory_gb=12.0,
            available_memory_gb=10.0,
            tier=GPUTier.MEDIUM,
            compute_capability=(7, 5),
            multi_processor_count=40,
            supports_bf16=False,
            supports_tf32=True,
        )

        config = TrainingConfig.from_gpu_profile(profile)

        assert config.model == "yolov8m.pt"
        mock_scaler.get_optimal_config.assert_called_once()

    @patch("training.training_config.GPUScaler")
    def test_auto_detect(self, mock_gpu_scaler_class):
        """Test auto-detection of GPU settings."""
        mock_scaler = MagicMock()
        mock_scaler.get_optimal_config.return_value = {
            "model": "yolov8m.pt",
            "imgsz": 640,
            "epochs": 50,
            "batch": 16,
        }
        mock_gpu_scaler_class.return_value = mock_scaler

        config = TrainingConfig.auto_detect()

        assert config.model == "yolov8m.pt"
        mock_gpu_scaler_class.assert_called_once()

    @patch("training.training_config.GPUScaler")
    def test_auto_detect_fast_mode(self, mock_gpu_scaler_class):
        """Test auto-detection with fast mode enabled."""
        mock_scaler = MagicMock()
        mock_scaler.get_optimal_config.return_value = {
            "model": "yolov8s.pt",
            "imgsz": 480,
            "epochs": 30,
            "batch": 32,
        }
        mock_gpu_scaler_class.return_value = mock_scaler

        config = TrainingConfig.auto_detect(fast_mode=True)

        mock_scaler.get_optimal_config.assert_called_once_with(fast_mode=True)

    def test_for_tier(self):
        """Test configuration for specific GPU tier."""
        from training.gpu_scaler import GPUTier

        config = TrainingConfig.for_tier(GPUTier.MEDIUM)

        # Should return a valid configuration
        assert isinstance(config, TrainingConfig)
        assert config.model is not None

    def test_with_overrides(self):
        """Test creating config with overrides."""
        base_config = TrainingConfig()
        new_config = base_config.with_overrides(epochs=100, batch=8)

        # Original should be unchanged
        assert base_config.epochs == 50
        assert base_config.batch == 16

        # New config should have overrides
        assert new_config.epochs == 100
        assert new_config.batch == 8

    def test_summary(self):
        """Test human-readable summary generation."""
        config = TrainingConfig()
        summary = config.summary()

        assert "Training Configuration" in summary
        assert "yolov8m.pt" in summary
        assert "640" in summary
        assert "50" in summary  # epochs


class TestLegacyCompatibility:
    """Test legacy compatibility functions."""

    def test_get_competition_config(self):
        """Test get_competition_config returns dictionary."""
        config = get_competition_config()

        assert isinstance(config, dict)
        assert "model" in config
        assert "epochs" in config

    def test_get_fast_config(self):
        """Test get_fast_config returns dictionary."""
        config = get_fast_config()

        assert isinstance(config, dict)
        assert "model" in config
        assert "epochs" in config


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
