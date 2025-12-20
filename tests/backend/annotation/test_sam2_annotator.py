"""
Tests for SAM2 annotator.

Tests the SAM2-based segmentation annotator.
"""

import sys
from importlib import reload
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

# Add scripts directory to path
scripts_dir = Path(__file__).parent.parent.parent.parent / "scripts"
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))


# Helper to create mock SAM2 modules
def create_sam2_mocks():
    """Create mock SAM2 modules for sys.modules patching."""
    mock_sam2 = MagicMock()
    mock_build_sam_module = MagicMock()
    mock_mask_gen_module = MagicMock()

    # Set up the build_sam2 function
    mock_build_sam2 = MagicMock()
    mock_build_sam_module.build_sam2 = mock_build_sam2

    # Set up the SAM2AutomaticMaskGenerator class
    mock_generator_class = MagicMock()
    mock_mask_gen_module.SAM2AutomaticMaskGenerator = mock_generator_class

    return {
        "sam2": mock_sam2,
        "sam2.build_sam": mock_build_sam_module,
        "sam2.automatic_mask_generator": mock_mask_gen_module,
    }, mock_build_sam2, mock_generator_class


class TestSAM2Annotator:
    """Test SAM2Annotator class."""

    def test_init_with_config(self):
        """Test initialization with custom configuration."""
        mocks, _, _ = create_sam2_mocks()

        with patch.dict("sys.modules", mocks):
            # Clear any cached import
            if "annotation.sam2_annotator" in sys.modules:
                del sys.modules["annotation.sam2_annotator"]

            from annotation.sam2_annotator import SAM2Annotator

            annotator = SAM2Annotator(
                model_path="sam2_l.pt",
                device="cpu",
                points_per_side=64,
                pred_iou_thresh=0.9,
                stability_score_thresh=0.95,
                min_mask_region_area=200,
                bbox_margin_ratio=0.05,
            )

            assert annotator.model_path == "sam2_l.pt"
            assert annotator.device == "cpu"
            assert annotator.points_per_side == 64
            assert annotator.pred_iou_thresh == 0.9
            assert annotator.stability_score_thresh == 0.95
            assert annotator.min_mask_region_area == 200
            assert annotator.bbox_margin_ratio == 0.05
            # Model should not be loaded yet (lazy loading)
            assert annotator.model is None

    def test_init_default_values(self):
        """Test initialization with default values."""
        mocks, _, _ = create_sam2_mocks()

        with patch.dict("sys.modules", mocks):
            if "annotation.sam2_annotator" in sys.modules:
                del sys.modules["annotation.sam2_annotator"]

            from annotation.sam2_annotator import SAM2Annotator
            from common.constants import (
                SAM2_DEFAULT_POINTS_PER_SIDE,
                SAM2_DEFAULT_PRED_IOU_THRESH,
                SAM2_DEFAULT_STABILITY_SCORE_THRESH,
                SAM2_DEFAULT_MIN_MASK_REGION_AREA,
                DEFAULT_BBOX_MARGIN_RATIO,
            )

            annotator = SAM2Annotator()

            assert annotator.points_per_side == SAM2_DEFAULT_POINTS_PER_SIDE
            assert annotator.pred_iou_thresh == SAM2_DEFAULT_PRED_IOU_THRESH
            assert annotator.stability_score_thresh == SAM2_DEFAULT_STABILITY_SCORE_THRESH
            assert annotator.min_mask_region_area == SAM2_DEFAULT_MIN_MASK_REGION_AREA
            assert annotator.bbox_margin_ratio == DEFAULT_BBOX_MARGIN_RATIO

    def test_load_model(self):
        """Test lazy model loading."""
        mocks, mock_build_sam2, mock_generator_class = create_sam2_mocks()

        mock_model = MagicMock()
        mock_build_sam2.return_value = mock_model
        mock_generator = MagicMock()
        mock_generator_class.return_value = mock_generator

        with patch.dict("sys.modules", mocks):
            if "annotation.sam2_annotator" in sys.modules:
                del sys.modules["annotation.sam2_annotator"]

            with patch("common.config_utils.get_sam2_model_config") as mock_get_config:
                mock_get_config.return_value = "configs/sam2/sam2_b.yaml"

                from annotation.sam2_annotator import SAM2Annotator

                annotator = SAM2Annotator(model_path="sam2_b.pt", device="cuda")

                # Trigger lazy loading
                annotator._load_model()

                mock_build_sam2.assert_called_once()
                mock_generator_class.assert_called_once()
                assert annotator.model is not None
                assert annotator.mask_generator is not None

    def test_select_best_mask_empty(self):
        """Test mask selection with empty list."""
        mocks, _, _ = create_sam2_mocks()

        with patch.dict("sys.modules", mocks):
            if "annotation.sam2_annotator" in sys.modules:
                del sys.modules["annotation.sam2_annotator"]

            from annotation.sam2_annotator import SAM2Annotator

            annotator = SAM2Annotator()
            result = annotator._select_best_mask([], (100, 100))

            assert result is None

    def test_select_best_mask_filters_by_area(self):
        """Test mask selection filters by area ratio."""
        mocks, _, _ = create_sam2_mocks()

        with patch.dict("sys.modules", mocks):
            if "annotation.sam2_annotator" in sys.modules:
                del sys.modules["annotation.sam2_annotator"]

            from annotation.sam2_annotator import SAM2Annotator

            annotator = SAM2Annotator()

            # Create masks with different areas
            masks = [
                {"area": 50, "predicted_iou": 0.9, "stability_score": 0.9},  # Too small
                {"area": 1000, "predicted_iou": 0.9, "stability_score": 0.9},  # Valid
                {"area": 9500, "predicted_iou": 0.9, "stability_score": 0.9},  # Too large
            ]

            # Image is 100x100 = 10000 pixels
            # min_area_ratio=0.01 means min 100 pixels
            # max_area_ratio=0.9 means max 9000 pixels
            result = annotator._select_best_mask(
                masks, (100, 100), min_area_ratio=0.01, max_area_ratio=0.9
            )

            assert result is not None
            assert result["area"] == 1000

    def test_select_best_mask_sorts_by_score(self):
        """Test mask selection sorts by combined score."""
        mocks, _, _ = create_sam2_mocks()

        with patch.dict("sys.modules", mocks):
            if "annotation.sam2_annotator" in sys.modules:
                del sys.modules["annotation.sam2_annotator"]

            from annotation.sam2_annotator import SAM2Annotator

            annotator = SAM2Annotator()

            masks = [
                {"area": 500, "predicted_iou": 0.7, "stability_score": 0.7},
                {"area": 600, "predicted_iou": 0.95, "stability_score": 0.95},  # Best score
                {"area": 550, "predicted_iou": 0.8, "stability_score": 0.8},
            ]

            result = annotator._select_best_mask(masks, (100, 100))

            assert result is not None
            assert result["area"] == 600  # Best combined score

    def test_annotate_image_no_detection(self):
        """Test annotation when no object is detected."""
        mocks, _, _ = create_sam2_mocks()

        # Create mock cv2
        mock_cv2 = MagicMock()
        mock_cv2.imread.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_cv2.cvtColor.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_cv2.COLOR_BGR2RGB = 4  # Actual OpenCV constant

        # Add cv2 to mocks
        all_mocks = {**mocks, "cv2": mock_cv2}

        with patch.dict("sys.modules", all_mocks):
            if "annotation.sam2_annotator" in sys.modules:
                del sys.modules["annotation.sam2_annotator"]

            from annotation.sam2_annotator import SAM2Annotator

            annotator = SAM2Annotator()
            annotator.model = MagicMock()
            annotator.mask_generator = MagicMock()
            annotator.mask_generator.generate.return_value = []  # No masks

            result = annotator.annotate_image("/path/to/image.jpg")

            assert result is None

    def test_annotate_image_success(self):
        """Test successful annotation."""
        mocks, _, _ = create_sam2_mocks()

        # Create mock cv2 with proper return values
        mock_cv2 = MagicMock()
        mock_cv2.imread.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_cv2.cvtColor.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_cv2.COLOR_BGR2RGB = 4
        mock_cv2.RETR_EXTERNAL = 0
        mock_cv2.CHAIN_APPROX_SIMPLE = 1
        # Mock findContours to return a valid contour for the mask
        mock_contour = np.array([[[20, 20]], [[80, 20]], [[80, 80]], [[20, 80]]])
        mock_cv2.findContours.return_value = ([mock_contour], None)
        mock_cv2.boundingRect.return_value = (20, 20, 60, 60)

        # Add cv2 to mocks - also need to clear annotation_utils
        all_mocks = {**mocks, "cv2": mock_cv2}

        with patch.dict("sys.modules", all_mocks):
            # Clear cached modules to ensure they pick up the mocked cv2
            for mod_name in list(sys.modules.keys()):
                if mod_name.startswith("annotation"):
                    del sys.modules[mod_name]

            from annotation.sam2_annotator import SAM2Annotator

            annotator = SAM2Annotator()
            annotator.model = MagicMock()
            annotator.mask_generator = MagicMock()

            # Create mock mask result
            mock_mask = np.zeros((100, 100), dtype=bool)
            mock_mask[20:80, 20:80] = True
            annotator.mask_generator.generate.return_value = [
                {
                    "segmentation": mock_mask,
                    "area": 3600,
                    "predicted_iou": 0.95,
                    "stability_score": 0.95,
                }
            ]

            result = annotator.annotate_image("/path/to/image.jpg")

            assert result is not None
            # Result should be YOLO format (x_center, y_center, width, height) normalized
            assert len(result) == 4
            # Check that values are normalized (between 0 and 1)
            assert all(0 <= v <= 1 for v in result)

    def test_annotate_image_returns_mask(self):
        """Test annotation with return_mask=True."""
        mocks, _, _ = create_sam2_mocks()

        # Create mock cv2 with proper return values
        mock_cv2 = MagicMock()
        mock_cv2.imread.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_cv2.cvtColor.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_cv2.COLOR_BGR2RGB = 4
        mock_cv2.RETR_EXTERNAL = 0
        mock_cv2.CHAIN_APPROX_SIMPLE = 1
        # Mock findContours to return a valid contour for the mask
        mock_contour = np.array([[[20, 20]], [[80, 20]], [[80, 80]], [[20, 80]]])
        mock_cv2.findContours.return_value = ([mock_contour], None)
        mock_cv2.boundingRect.return_value = (20, 20, 60, 60)

        # Add cv2 to mocks
        all_mocks = {**mocks, "cv2": mock_cv2}

        with patch.dict("sys.modules", all_mocks):
            # Clear cached modules to ensure they pick up the mocked cv2
            for mod_name in list(sys.modules.keys()):
                if mod_name.startswith("annotation"):
                    del sys.modules[mod_name]

            from annotation.sam2_annotator import SAM2Annotator

            annotator = SAM2Annotator()
            annotator.model = MagicMock()
            annotator.mask_generator = MagicMock()

            mock_mask = np.zeros((100, 100), dtype=bool)
            mock_mask[20:80, 20:80] = True
            annotator.mask_generator.generate.return_value = [
                {
                    "segmentation": mock_mask,
                    "area": 3600,
                    "predicted_iou": 0.95,
                    "stability_score": 0.95,
                }
            ]

            result = annotator.annotate_image("/path/to/image.jpg", return_mask=True)

            assert result is not None
            bbox, mask = result
            # Result should be YOLO format (x_center, y_center, width, height) normalized
            assert len(bbox) == 4
            assert all(0 <= v <= 1 for v in bbox)
            assert isinstance(mask, np.ndarray)

    def test_annotate_image_file_not_found(self):
        """Test annotation with non-existent file."""
        mocks, _, _ = create_sam2_mocks()

        # Create mock cv2 that returns None for imread
        mock_cv2 = MagicMock()
        mock_cv2.imread.return_value = None

        # Add cv2 to mocks
        all_mocks = {**mocks, "cv2": mock_cv2}

        with patch.dict("sys.modules", all_mocks):
            if "annotation.sam2_annotator" in sys.modules:
                del sys.modules["annotation.sam2_annotator"]

            from annotation.sam2_annotator import SAM2Annotator

            annotator = SAM2Annotator()
            annotator.model = MagicMock()
            annotator.mask_generator = MagicMock()

            result = annotator.annotate_image("/nonexistent/image.jpg")

            assert result is None


class TestSAM2AnnotatorGPU:
    """Test SAM2Annotator GPU/CPU usage."""

    def test_gpu_device_usage(self):
        """Test that GPU device is configured correctly."""
        mocks, _, _ = create_sam2_mocks()

        with patch.dict("sys.modules", mocks):
            if "annotation.sam2_annotator" in sys.modules:
                del sys.modules["annotation.sam2_annotator"]

            from annotation.sam2_annotator import SAM2Annotator

            annotator = SAM2Annotator(device="cuda")
            assert annotator.device == "cuda"

    def test_cpu_fallback(self):
        """Test that CPU fallback is configured correctly."""
        mocks, _, _ = create_sam2_mocks()

        with patch.dict("sys.modules", mocks):
            if "annotation.sam2_annotator" in sys.modules:
                del sys.modules["annotation.sam2_annotator"]

            from annotation.sam2_annotator import SAM2Annotator

            annotator = SAM2Annotator(device="cpu")
            assert annotator.device == "cpu"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
