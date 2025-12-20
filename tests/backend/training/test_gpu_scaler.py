"""
Tests for GPU hardware auto-scaling module.

Tests GPU detection, tier classification, and training parameter optimization.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# Add scripts directory to path
scripts_dir = Path(__file__).parent.parent.parent.parent / "scripts"
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

from training.gpu_scaler import (
    GPUTier,
    GPUProfile,
    GPUScalingConfig,
    GPUScaler,
    OOMRecoveryStrategy,
    VRAM_THRESHOLDS,
    TIER_CONFIGS,
    MODEL_DOWNGRADE_PATH,
    STANDARD_IMAGE_SIZES,
    get_gpu_scaler,
)


class TestGPUTier:
    """Test GPUTier enum."""

    def test_tier_values(self):
        """Test tier enum values."""
        assert GPUTier.LOW.value == "low"
        assert GPUTier.MEDIUM.value == "medium"
        assert GPUTier.HIGH.value == "high"
        assert GPUTier.WORKSTATION.value == "workstation"
        assert GPUTier.CPU_ONLY.value == "cpu"

    def test_all_tiers_in_config(self):
        """Test that all tiers have configurations."""
        for tier in GPUTier:
            assert tier in TIER_CONFIGS


class TestGPUProfile:
    """Test GPUProfile dataclass."""

    def test_profile_creation(self):
        """Test GPUProfile creation."""
        profile = GPUProfile(
            name="NVIDIA GeForce RTX 2080",
            device_index=0,
            total_memory_gb=8.0,
            available_memory_gb=7.5,
            tier=GPUTier.MEDIUM,
            compute_capability=(7, 5),
            multi_processor_count=46,
            supports_bf16=False,
            supports_tf32=False,
        )

        assert profile.name == "NVIDIA GeForce RTX 2080"
        assert profile.tier == GPUTier.MEDIUM
        assert profile.total_memory_gb == 8.0

    def test_summary_format(self):
        """Test summary string generation."""
        profile = GPUProfile(
            name="RTX 3090",
            device_index=0,
            total_memory_gb=24.0,
            available_memory_gb=22.0,
            tier=GPUTier.HIGH,
            compute_capability=(8, 6),
            multi_processor_count=82,
            supports_bf16=True,
            supports_tf32=True,
        )

        summary = profile.summary()

        assert "RTX 3090" in summary
        assert "24.0" in summary
        assert "high" in summary
        assert "BF16=True" in summary


class TestGPUScalingConfig:
    """Test GPUScalingConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = GPUScalingConfig()

        assert config.auto_detect is True
        assert config.force_gpu_tier is None
        assert config.vram_safety_margin == 0.15
        assert config.max_vram_utilization == 0.85
        assert config.enable_oom_recovery is True
        assert config.oom_batch_reduction_factor == 0.5
        assert config.max_oom_retries == 3

    def test_custom_values(self):
        """Test custom configuration."""
        config = GPUScalingConfig(
            auto_detect=False,
            force_gpu_tier=GPUTier.HIGH,
            max_vram_utilization=0.9,
        )

        assert config.auto_detect is False
        assert config.force_gpu_tier == GPUTier.HIGH
        assert config.max_vram_utilization == 0.9


class TestGPUScalerInitialization:
    """Test GPUScaler initialization."""

    @patch("training.gpu_scaler.GPUScaler._detect_gpu")
    def test_auto_detect_enabled(self, mock_detect):
        """Test that GPU detection runs when auto_detect is True."""
        config = GPUScalingConfig(auto_detect=True)
        scaler = GPUScaler(config)

        mock_detect.assert_called_once()

    @patch("training.gpu_scaler.GPUScaler._detect_gpu")
    def test_auto_detect_disabled(self, mock_detect):
        """Test that GPU detection is skipped when auto_detect is False."""
        config = GPUScalingConfig(auto_detect=False)
        scaler = GPUScaler(config)

        mock_detect.assert_not_called()


class TestGPUScalerTierClassification:
    """Test GPUScaler tier classification."""

    def test_low_tier_classification(self):
        """Test classification of low-tier GPU (< 6GB)."""
        config = GPUScalingConfig(auto_detect=False)
        scaler = GPUScaler(config)

        tier = scaler._classify_tier(4.0)

        assert tier == GPUTier.LOW

    def test_medium_tier_classification(self):
        """Test classification of medium-tier GPU (6-12GB)."""
        config = GPUScalingConfig(auto_detect=False)
        scaler = GPUScaler(config)

        tier = scaler._classify_tier(8.0)

        assert tier == GPUTier.MEDIUM

    def test_high_tier_classification(self):
        """Test classification of high-tier GPU (12-24GB)."""
        config = GPUScalingConfig(auto_detect=False)
        scaler = GPUScaler(config)

        tier = scaler._classify_tier(16.0)

        assert tier == GPUTier.HIGH

    def test_workstation_tier_classification(self):
        """Test classification of workstation GPU (> 24GB)."""
        config = GPUScalingConfig(auto_detect=False)
        scaler = GPUScaler(config)

        tier = scaler._classify_tier(40.0)

        assert tier == GPUTier.WORKSTATION

    def test_forced_tier_overrides(self):
        """Test that forced tier overrides classification."""
        config = GPUScalingConfig(
            auto_detect=False,
            force_gpu_tier=GPUTier.HIGH,
        )
        scaler = GPUScaler(config)

        # Even with 4GB VRAM, forced tier should be HIGH
        tier = scaler._classify_tier(4.0)

        assert tier == GPUTier.HIGH


class TestGPUScalerWithMockedGPU:
    """Test GPUScaler with mocked GPU detection."""

    def _create_scaler_with_profile(self, tier: GPUTier, vram_gb: float):
        """Helper to create scaler with mock GPU profile."""
        config = GPUScalingConfig(auto_detect=False)
        scaler = GPUScaler(config)
        scaler.profile = GPUProfile(
            name="Mock GPU",
            device_index=0,
            total_memory_gb=vram_gb,
            available_memory_gb=vram_gb * 0.9,
            tier=tier,
            compute_capability=(7, 5),
            multi_processor_count=46,
            supports_bf16=tier in (GPUTier.HIGH, GPUTier.WORKSTATION),
            supports_tf32=tier in (GPUTier.HIGH, GPUTier.WORKSTATION),
        )
        return scaler

    def test_get_tier_with_profile(self):
        """Test get_tier returns profile tier."""
        scaler = self._create_scaler_with_profile(GPUTier.MEDIUM, 8.0)

        assert scaler.get_tier() == GPUTier.MEDIUM

    def test_get_tier_without_profile(self):
        """Test get_tier returns CPU_ONLY when no profile."""
        config = GPUScalingConfig(auto_detect=False)
        scaler = GPUScaler(config)

        assert scaler.get_tier() == GPUTier.CPU_ONLY

    def test_get_optimal_config_medium_tier(self):
        """Test optimal config for medium tier GPU."""
        scaler = self._create_scaler_with_profile(GPUTier.MEDIUM, 8.0)

        config = scaler.get_optimal_config()

        # Medium tier should use yolov8m
        assert config["model"] == "yolov8m.pt"
        assert config["imgsz"] == 640
        assert config["optimizer"] == "AdamW"
        assert config["amp"] is True

    def test_get_optimal_config_cpu_only(self):
        """Test optimal config for CPU only."""
        config_obj = GPUScalingConfig(auto_detect=False)
        scaler = GPUScaler(config_obj)
        # No profile means CPU only

        config = scaler.get_optimal_config()

        # CPU should use smallest model
        assert config["model"] == "yolov8n.pt"
        assert config["imgsz"] == 320
        assert config["batch"] == 4

    def test_fast_mode_reduces_settings(self):
        """Test that fast mode reduces model size and epochs."""
        scaler = self._create_scaler_with_profile(GPUTier.HIGH, 24.0)

        normal_config = scaler.get_optimal_config(fast_mode=False)
        fast_config = scaler.get_optimal_config(fast_mode=True)

        # Fast mode should use smaller model or same, never larger
        models = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"]
        normal_idx = models.index(normal_config["model"])
        fast_idx = models.index(fast_config["model"])
        assert fast_idx <= normal_idx

        # Epochs should be reduced
        assert fast_config["epochs"] <= normal_config["epochs"]


class TestGPUScalerBatchSize:
    """Test GPUScaler batch size calculation."""

    def test_batch_size_medium_gpu(self):
        """Test batch size calculation for medium GPU."""
        config = GPUScalingConfig(auto_detect=False)
        scaler = GPUScaler(config)
        scaler.profile = GPUProfile(
            name="RTX 2080",
            device_index=0,
            total_memory_gb=8.0,
            available_memory_gb=7.0,
            tier=GPUTier.MEDIUM,
            compute_capability=(7, 5),
            multi_processor_count=46,
            supports_bf16=False,
            supports_tf32=False,
        )

        batch_size = scaler.calculate_batch_size("yolov8m.pt", 640)

        # Should be power of 2 and reasonable for 8GB GPU
        assert batch_size in [4, 8, 16, 32]
        assert batch_size >= 4
        assert batch_size <= 128

    def test_batch_size_no_gpu(self):
        """Test batch size returns default when no GPU."""
        config = GPUScalingConfig(auto_detect=False)
        scaler = GPUScaler(config)

        batch_size = scaler.calculate_batch_size("yolov8m.pt", 640)

        assert batch_size == 8  # Conservative default

    def test_batch_size_scales_with_image_size(self):
        """Test that batch size decreases with larger images."""
        config = GPUScalingConfig(auto_detect=False)
        scaler = GPUScaler(config)
        scaler.profile = GPUProfile(
            name="RTX 3090",
            device_index=0,
            total_memory_gb=24.0,
            available_memory_gb=22.0,
            tier=GPUTier.HIGH,
            compute_capability=(8, 6),
            multi_processor_count=82,
            supports_bf16=True,
            supports_tf32=True,
        )

        batch_640 = scaler.calculate_batch_size("yolov8m.pt", 640)
        batch_1280 = scaler.calculate_batch_size("yolov8m.pt", 1280)

        # Larger image size should result in smaller batch
        assert batch_1280 <= batch_640


class TestGPUScalerTrainingTime:
    """Test GPUScaler training time estimation."""

    def test_estimate_training_time_with_gpu(self):
        """Test training time estimation with GPU."""
        config = GPUScalingConfig(auto_detect=False)
        scaler = GPUScaler(config)
        scaler.profile = GPUProfile(
            name="RTX 3090",
            device_index=0,
            total_memory_gb=24.0,
            available_memory_gb=22.0,
            tier=GPUTier.HIGH,
            compute_capability=(8, 6),
            multi_processor_count=82,
            supports_bf16=True,
            supports_tf32=True,
        )

        time_minutes = scaler.estimate_training_time(
            epochs=50,
            dataset_size=1000,
            batch_size=16,
        )

        # Should return reasonable estimate
        assert time_minutes > 0
        assert time_minutes < 1000  # Less than 1000 minutes for small dataset

    def test_estimate_training_time_without_gpu(self):
        """Test training time estimation without GPU (CPU mode)."""
        config = GPUScalingConfig(auto_detect=False)
        scaler = GPUScaler(config)

        time_minutes = scaler.estimate_training_time(
            epochs=10,
            dataset_size=100,
            batch_size=4,
        )

        # CPU training should be very slow
        assert time_minutes > 0

    def test_training_time_scales_with_epochs(self):
        """Test that training time scales with epochs."""
        config = GPUScalingConfig(auto_detect=False)
        scaler = GPUScaler(config)
        scaler.profile = GPUProfile(
            name="RTX 2080",
            device_index=0,
            total_memory_gb=8.0,
            available_memory_gb=7.0,
            tier=GPUTier.MEDIUM,
            compute_capability=(7, 5),
            multi_processor_count=46,
            supports_bf16=False,
            supports_tf32=False,
        )

        time_10 = scaler.estimate_training_time(10, 100, 16)
        time_50 = scaler.estimate_training_time(50, 100, 16)

        assert time_50 > time_10
        assert time_50 == pytest.approx(time_10 * 5, rel=0.01)


class TestOOMRecoveryStrategy:
    """Test OOMRecoveryStrategy class."""

    def test_first_retry_reduces_batch(self):
        """Test that first retry reduces batch size by 50%."""
        initial_config = {"model": "yolov8m.pt", "batch": 32, "imgsz": 640}
        strategy = OOMRecoveryStrategy(initial_config, max_retries=3)

        recovered = strategy.get_recovery_config()

        assert recovered is not None
        assert recovered["batch"] == 16  # 32 // 2

    def test_second_retry_reduces_image_size(self):
        """Test that second retry reduces image size."""
        initial_config = {"model": "yolov8m.pt", "batch": 32, "imgsz": 640}
        strategy = OOMRecoveryStrategy(initial_config, max_retries=3)

        # First retry
        strategy.get_recovery_config()
        # Second retry
        recovered = strategy.get_recovery_config()

        assert recovered is not None
        assert recovered["imgsz"] == 480  # One step down from 640

    def test_third_retry_downgrades_model(self):
        """Test that third retry downgrades model."""
        initial_config = {"model": "yolov8m.pt", "batch": 32, "imgsz": 640}
        strategy = OOMRecoveryStrategy(initial_config, max_retries=3)

        # First and second retries
        strategy.get_recovery_config()
        strategy.get_recovery_config()
        # Third retry
        recovered = strategy.get_recovery_config()

        assert recovered is not None
        assert recovered["model"] == "yolov8s.pt"  # Downgrade from m to s

    def test_max_retries_exceeded(self):
        """Test that None is returned when max retries exceeded."""
        initial_config = {"model": "yolov8m.pt", "batch": 32, "imgsz": 640}
        strategy = OOMRecoveryStrategy(initial_config, max_retries=2)

        # Exhaust retries
        strategy.get_recovery_config()  # 1
        strategy.get_recovery_config()  # 2
        result = strategy.get_recovery_config()  # Should be None

        assert result is None

    def test_changes_summary(self):
        """Test changes summary generation."""
        initial_config = {"model": "yolov8m.pt", "batch": 32, "imgsz": 640}
        strategy = OOMRecoveryStrategy(initial_config, max_retries=3)

        strategy.get_recovery_config()

        summary = strategy.get_changes_summary()

        assert "Batch" in summary
        assert "32" in summary
        assert "16" in summary

    def test_no_changes_summary(self):
        """Test summary when no changes made."""
        initial_config = {"model": "yolov8m.pt", "batch": 32, "imgsz": 640}
        strategy = OOMRecoveryStrategy(initial_config, max_retries=3)

        summary = strategy.get_changes_summary()

        assert summary == "No changes made"

    def test_reset_clears_state(self):
        """Test that reset clears all state."""
        initial_config = {"model": "yolov8m.pt", "batch": 32, "imgsz": 640}
        strategy = OOMRecoveryStrategy(initial_config, max_retries=3)

        strategy.get_recovery_config()
        strategy.get_recovery_config()

        strategy.reset()

        assert strategy.retry_count == 0
        assert strategy.changes == []
        assert strategy.current_config == initial_config

    def test_batch_minimum_enforced(self):
        """Test that batch size doesn't go below 4."""
        initial_config = {"model": "yolov8m.pt", "batch": 4, "imgsz": 640}
        strategy = OOMRecoveryStrategy(initial_config, max_retries=3)

        recovered = strategy.get_recovery_config()

        # Should stay at 4, not go lower
        assert recovered["batch"] == 4


class TestModelDowngradePath:
    """Test model downgrade path configuration."""

    def test_all_models_have_downgrade(self):
        """Test that all models have a downgrade path."""
        models = ["yolov8x.pt", "yolov8l.pt", "yolov8m.pt", "yolov8s.pt", "yolov8n.pt"]
        for model in models:
            assert model in MODEL_DOWNGRADE_PATH

    def test_smallest_model_stays_same(self):
        """Test that smallest model stays same when downgraded."""
        assert MODEL_DOWNGRADE_PATH["yolov8n.pt"] == "yolov8n.pt"

    def test_downgrade_order(self):
        """Test correct downgrade order."""
        assert MODEL_DOWNGRADE_PATH["yolov8x.pt"] == "yolov8l.pt"
        assert MODEL_DOWNGRADE_PATH["yolov8l.pt"] == "yolov8m.pt"
        assert MODEL_DOWNGRADE_PATH["yolov8m.pt"] == "yolov8s.pt"
        assert MODEL_DOWNGRADE_PATH["yolov8s.pt"] == "yolov8n.pt"


class TestVRAMThresholds:
    """Test VRAM threshold configuration."""

    def test_thresholds_cover_all_ranges(self):
        """Test that thresholds cover 0 to infinity."""
        # Check LOW starts at 0
        assert VRAM_THRESHOLDS[GPUTier.LOW][0] == 0

        # Check WORKSTATION goes to infinity
        assert VRAM_THRESHOLDS[GPUTier.WORKSTATION][1] == float("inf")

    def test_thresholds_are_continuous(self):
        """Test that threshold ranges are continuous."""
        tiers = [GPUTier.LOW, GPUTier.MEDIUM, GPUTier.HIGH, GPUTier.WORKSTATION]

        for i in range(len(tiers) - 1):
            current_max = VRAM_THRESHOLDS[tiers[i]][1]
            next_min = VRAM_THRESHOLDS[tiers[i + 1]][0]
            assert current_max == next_min


class TestGetGpuScaler:
    """Test get_gpu_scaler factory function."""

    @patch("training.gpu_scaler.GPUScaler._detect_gpu")
    def test_factory_function(self, mock_detect):
        """Test factory function creates GPUScaler."""
        scaler = get_gpu_scaler()

        assert isinstance(scaler, GPUScaler)

    @patch("training.gpu_scaler.GPUScaler._detect_gpu")
    def test_factory_with_config(self, mock_detect):
        """Test factory function accepts config."""
        config = GPUScalingConfig(auto_detect=False)
        scaler = get_gpu_scaler(config)

        assert scaler.config.auto_detect is False


class TestStandardImageSizes:
    """Test standard image size configuration."""

    def test_sizes_are_sorted(self):
        """Test that standard sizes are in ascending order."""
        assert STANDARD_IMAGE_SIZES == sorted(STANDARD_IMAGE_SIZES)

    def test_common_sizes_included(self):
        """Test that common training sizes are included."""
        assert 320 in STANDARD_IMAGE_SIZES
        assert 640 in STANDARD_IMAGE_SIZES

    def test_sizes_are_positive(self):
        """Test that all sizes are positive."""
        assert all(size > 0 for size in STANDARD_IMAGE_SIZES)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
