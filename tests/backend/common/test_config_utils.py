"""
Tests for configuration utilities.

Tests dataclass configurations for annotation, training, and evaluation.
"""

import json
import sys
from pathlib import Path

import pytest

# Add scripts directory to path
scripts_dir = Path(__file__).parent.parent.parent.parent / "scripts"
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

from common.config_utils import (
    AnnotatorConfig,
    BackgroundSubtractionConfig,
    EvaluationConfig,
    SAM2Config,
    TrainingConfig,
    get_class_id_map,
    get_class_names,
    get_sam2_model_config,
    load_class_config,
)


class TestAnnotatorConfig:
    """Test AnnotatorConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = AnnotatorConfig()

        assert config.bbox_margin_ratio == 0.02  # From constants.py
        assert config.min_contour_area == 500
        assert config.max_contour_area_ratio == 0.9

    def test_custom_values(self):
        """Test custom configuration values."""
        config = AnnotatorConfig(
            bbox_margin_ratio=0.1,
            min_contour_area=1000,
            max_contour_area_ratio=0.8,
        )

        assert config.bbox_margin_ratio == 0.1
        assert config.min_contour_area == 1000
        assert config.max_contour_area_ratio == 0.8


class TestBackgroundSubtractionConfig:
    """Test BackgroundSubtractionConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = BackgroundSubtractionConfig()

        assert config.blur_kernel_size == 5
        assert config.threshold_method == "otsu"
        assert config.fixed_threshold == 30
        assert config.morph_kernel_size == 5
        assert config.erosion_iterations == 2
        assert config.dilation_iterations == 3

    def test_inherits_annotator_config(self):
        """Test that it inherits from AnnotatorConfig."""
        config = BackgroundSubtractionConfig()

        # Should have parent class attributes
        assert hasattr(config, "bbox_margin_ratio")
        assert hasattr(config, "min_contour_area")
        assert hasattr(config, "max_contour_area_ratio")

    def test_valid_threshold_methods(self):
        """Test valid threshold method values."""
        for method in ["otsu", "adaptive", "fixed"]:
            config = BackgroundSubtractionConfig(threshold_method=method)
            assert config.threshold_method == method

    def test_invalid_threshold_method_raises(self):
        """Test that invalid threshold method raises ValueError."""
        with pytest.raises(ValueError, match="Invalid threshold_method"):
            BackgroundSubtractionConfig(threshold_method="invalid")

    def test_even_blur_kernel_raises(self):
        """Test that even blur kernel size raises ValueError."""
        with pytest.raises(ValueError, match="blur_kernel_size must be odd"):
            BackgroundSubtractionConfig(blur_kernel_size=4)

    def test_odd_blur_kernel_valid(self):
        """Test that odd blur kernel sizes are valid."""
        for kernel_size in [3, 5, 7, 9, 11]:
            config = BackgroundSubtractionConfig(blur_kernel_size=kernel_size)
            assert config.blur_kernel_size == kernel_size


class TestSAM2Config:
    """Test SAM2Config dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = SAM2Config()

        assert config.device == "cuda"
        assert config.points_per_side == 32
        assert config.pred_iou_thresh == 0.88
        assert config.stability_score_thresh == 0.92  # From constants.py
        assert config.min_mask_region_area == 100

    def test_custom_device(self):
        """Test setting custom device."""
        config = SAM2Config(device="cpu")
        assert config.device == "cpu"

        config = SAM2Config(device="cuda:1")
        assert config.device == "cuda:1"

    def test_inherits_annotator_config(self):
        """Test that it inherits from AnnotatorConfig."""
        config = SAM2Config()

        # Should have parent class attributes
        assert hasattr(config, "bbox_margin_ratio")
        assert hasattr(config, "min_contour_area")


class TestGetSam2ModelConfig:
    """Test get_sam2_model_config function."""

    def test_sam2_1_base_model(self):
        """Test SAM2.1 base model detection."""
        paths = [
            "sam2.1_hiera_base_plus.pt",
            "sam2.1_b+.pt",
            "models/sam2.1_base.pt",
        ]
        for path in paths:
            config = get_sam2_model_config(path)
            assert "sam2.1" in config
            assert "b+" in config.lower() or "base" in path.lower()

    def test_sam2_1_large_model(self):
        """Test SAM2.1 large model detection."""
        paths = [
            "sam2.1_hiera_large.pt",
            "sam2.1_l.pt",
        ]
        for path in paths:
            config = get_sam2_model_config(path)
            assert "sam2.1" in config
            assert "_l" in config

    def test_sam2_1_small_model(self):
        """Test SAM2.1 small model detection."""
        paths = [
            "sam2.1_hiera_small.pt",
            "sam2.1_s.pt",
        ]
        for path in paths:
            config = get_sam2_model_config(path)
            assert "sam2.1" in config
            assert "_s" in config

    def test_sam2_1_tiny_model(self):
        """Test SAM2.1 tiny model detection."""
        paths = [
            "sam2.1_hiera_tiny.pt",
            "sam2.1_t.pt",
        ]
        for path in paths:
            config = get_sam2_model_config(path)
            assert "sam2.1" in config
            assert "_t" in config

    def test_sam2_base_model(self):
        """Test SAM2 (non 2.1) base model detection."""
        config = get_sam2_model_config("sam2_b.pt")
        assert "sam2.1" in config  # Uses SAM2.1 config for compatibility

    def test_default_fallback(self):
        """Test default fallback for unknown model."""
        config = get_sam2_model_config("unknown_model.pt")
        assert "b+" in config.lower()  # Default to base


class TestTrainingConfig:
    """Test TrainingConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TrainingConfig()

        assert config.model == "yolov8m.pt"
        assert config.imgsz == 640
        assert config.epochs == 50
        assert config.batch == 16
        assert config.patience == 10
        assert config.optimizer == "AdamW"
        assert config.augment is True
        assert config.amp is True
        assert config.cache is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = TrainingConfig(
            model="yolov8n.pt",
            epochs=30,
            batch=32,
        )

        assert config.model == "yolov8n.pt"
        assert config.epochs == 30
        assert config.batch == 32

    def test_to_dict(self):
        """Test to_dict method returns all fields."""
        config = TrainingConfig()
        result = config.to_dict()

        # Check required keys exist
        required_keys = [
            "model", "imgsz", "epochs", "batch", "patience",
            "optimizer", "lr0", "lrf", "momentum", "weight_decay",
            "augment", "hsv_h", "hsv_s", "hsv_v",
            "degrees", "translate", "scale", "shear",
            "flipud", "fliplr", "mosaic", "mixup",
            "workers", "cache", "amp", "close_mosaic",
            "save", "save_period", "exist_ok",
        ]
        for key in required_keys:
            assert key in result

    def test_to_dict_values(self):
        """Test to_dict returns correct values."""
        config = TrainingConfig(epochs=100, batch=8)
        result = config.to_dict()

        assert result["epochs"] == 100
        assert result["batch"] == 8

    def test_competition_preset(self):
        """Test competition preset returns default config."""
        config = TrainingConfig.competition()

        # Competition config uses defaults
        assert config.model == "yolov8m.pt"
        assert config.epochs == 50
        assert config.imgsz == 640

    def test_fast_preset(self):
        """Test fast preset returns optimized settings."""
        config = TrainingConfig.fast()

        assert config.model == "yolov8s.pt"
        assert config.epochs == 30
        assert config.batch == 32
        assert config.patience == 5
        assert config.imgsz == 480


class TestEvaluationConfig:
    """Test EvaluationConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = EvaluationConfig()

        assert config.conf_threshold == 0.25
        assert config.iou_threshold == 0.5  # From constants.py
        assert config.device == "cuda"
        assert config.verbose is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = EvaluationConfig(
            conf_threshold=0.5,
            iou_threshold=0.6,
            device="cpu",
            verbose=False,
        )

        assert config.conf_threshold == 0.5
        assert config.iou_threshold == 0.6
        assert config.device == "cpu"
        assert config.verbose is False


class TestLoadClassConfig:
    """Test load_class_config function."""

    def test_file_not_found(self, tmp_path):
        """Test error when file does not exist."""
        with pytest.raises(FileNotFoundError, match="not found"):
            load_class_config(str(tmp_path / "nonexistent.json"))

    def test_load_valid_json(self, tmp_path):
        """Test loading valid JSON configuration."""
        config_path = tmp_path / "classes.json"
        config_data = {
            "categories": ["food", "container"],
            "objects": [
                {"class_id": 0, "class_name": "apple"},
                {"class_id": 1, "class_name": "cup"},
            ],
        }
        with open(config_path, "w") as f:
            json.dump(config_data, f)

        result = load_class_config(str(config_path))

        assert result == config_data
        assert len(result["objects"]) == 2

    def test_load_empty_json(self, tmp_path):
        """Test loading empty JSON object."""
        config_path = tmp_path / "empty.json"
        with open(config_path, "w") as f:
            json.dump({}, f)

        result = load_class_config(str(config_path))

        assert result == {}


class TestGetClassNames:
    """Test get_class_names function."""

    def test_extract_class_names(self):
        """Test extracting class names from config."""
        config = {
            "objects": [
                {"class_id": 0, "class_name": "apple"},
                {"class_id": 1, "class_name": "banana"},
                {"class_id": 2, "class_name": "orange"},
            ]
        }

        names = get_class_names(config)

        assert names == ["apple", "banana", "orange"]

    def test_empty_objects(self):
        """Test with empty objects list."""
        config = {"objects": []}

        names = get_class_names(config)

        assert names == []

    def test_missing_objects_key(self):
        """Test with missing objects key."""
        config = {}

        names = get_class_names(config)

        assert names == []

    def test_fallback_class_names(self):
        """Test fallback when class_name is missing."""
        config = {
            "objects": [
                {"class_id": 0},
                {"class_id": 1, "class_name": "banana"},
            ]
        }

        names = get_class_names(config)

        assert names == ["class_0", "banana"]


class TestGetClassIdMap:
    """Test get_class_id_map function."""

    def test_create_id_map(self):
        """Test creating class ID map."""
        config = {
            "objects": [
                {"class_id": 0, "class_name": "apple"},
                {"class_id": 1, "class_name": "banana"},
            ]
        }

        id_map = get_class_id_map(config)

        assert id_map == {"apple": 0, "banana": 1}

    def test_empty_objects(self):
        """Test with empty objects list."""
        config = {"objects": []}

        id_map = get_class_id_map(config)

        assert id_map == {}

    def test_missing_class_id(self):
        """Test fallback when class_id is missing."""
        config = {
            "objects": [
                {"class_name": "apple"},
                {"class_name": "banana"},
            ]
        }

        id_map = get_class_id_map(config)

        # Falls back to index-based IDs
        assert id_map == {"apple": 0, "banana": 1}

    def test_non_sequential_ids(self):
        """Test with non-sequential class IDs."""
        config = {
            "objects": [
                {"class_id": 5, "class_name": "apple"},
                {"class_id": 10, "class_name": "banana"},
            ]
        }

        id_map = get_class_id_map(config)

        assert id_map == {"apple": 5, "banana": 10}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
