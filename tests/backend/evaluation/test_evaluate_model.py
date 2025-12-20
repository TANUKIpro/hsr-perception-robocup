"""
Tests for model evaluation module.

Tests the ModelEvaluator class, evaluation reports, and metrics calculation.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

# Add scripts directory to path
scripts_dir = Path(__file__).parent.parent.parent.parent / "scripts"
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))


class TestClassMetrics:
    """Test ClassMetrics dataclass."""

    def test_creation(self):
        """Test creating ClassMetrics."""
        from evaluation.evaluate_model import ClassMetrics

        metrics = ClassMetrics(
            class_name="test_class",
            precision=0.95,
            recall=0.90,
            f1_score=0.92,
            ap50=0.88,
            ap50_95=0.75,
            num_samples=100,
        )

        assert metrics.class_name == "test_class"
        assert metrics.precision == 0.95
        assert metrics.recall == 0.90
        assert metrics.f1_score == 0.92
        assert metrics.ap50 == 0.88
        assert metrics.ap50_95 == 0.75
        assert metrics.num_samples == 100

    def test_to_dict(self):
        """Test conversion to dictionary."""
        from evaluation.evaluate_model import ClassMetrics

        metrics = ClassMetrics(
            class_name="test_class",
            precision=0.95,
            recall=0.90,
            f1_score=0.92,
            ap50=0.88,
            ap50_95=0.75,
            num_samples=100,
        )

        result = metrics.to_dict()

        assert isinstance(result, dict)
        assert result["class_name"] == "test_class"
        assert result["precision"] == 0.95
        assert result["recall"] == 0.90
        assert result["ap50"] == 0.88


class TestEvaluationReport:
    """Test EvaluationReport dataclass."""

    def test_creation(self):
        """Test creating EvaluationReport."""
        from evaluation.evaluate_model import EvaluationReport, ClassMetrics

        class_metrics = ClassMetrics(
            class_name="object",
            precision=0.90,
            recall=0.85,
            f1_score=0.87,
            ap50=0.88,
            ap50_95=0.70,
            num_samples=50,
        )

        report = EvaluationReport(
            model_path="/path/to/model.pt",
            dataset_path="/path/to/data.yaml",
            overall_map50=0.88,
            overall_map50_95=0.70,
            overall_precision=0.90,
            overall_recall=0.85,
            per_class_metrics={"object": class_metrics},
            inference_time_ms=25.0,
            inference_time_std=3.0,
            num_test_images=100,
        )

        assert report.model_path == "/path/to/model.pt"
        assert report.overall_map50 == 0.88
        assert report.inference_time_ms == 25.0

    def test_meets_requirements_pass(self):
        """Test meets_requirements when all requirements pass."""
        from evaluation.evaluate_model import EvaluationReport

        report = EvaluationReport(
            model_path="/path/to/model.pt",
            dataset_path="/path/to/data.yaml",
            overall_map50=0.90,  # Above 0.85 target
            overall_map50_95=0.70,
            overall_precision=0.90,
            overall_recall=0.85,
            per_class_metrics={},
            inference_time_ms=50.0,  # Below 100ms target
            inference_time_std=5.0,
            num_test_images=100,
        )

        meets, issues = report.meets_requirements()

        assert meets is True
        assert len(issues) == 0

    def test_meets_requirements_fail_map(self):
        """Test meets_requirements when mAP is too low."""
        from evaluation.evaluate_model import EvaluationReport

        report = EvaluationReport(
            model_path="/path/to/model.pt",
            dataset_path="/path/to/data.yaml",
            overall_map50=0.70,  # Below 0.85 target
            overall_map50_95=0.60,
            overall_precision=0.80,
            overall_recall=0.75,
            per_class_metrics={},
            inference_time_ms=50.0,
            inference_time_std=5.0,
            num_test_images=100,
        )

        meets, issues = report.meets_requirements()

        assert meets is False
        assert len(issues) == 1
        assert "mAP50" in issues[0]

    def test_meets_requirements_fail_inference(self):
        """Test meets_requirements when inference is too slow."""
        from evaluation.evaluate_model import EvaluationReport

        report = EvaluationReport(
            model_path="/path/to/model.pt",
            dataset_path="/path/to/data.yaml",
            overall_map50=0.90,
            overall_map50_95=0.70,
            overall_precision=0.90,
            overall_recall=0.85,
            per_class_metrics={},
            inference_time_ms=150.0,  # Above 100ms target
            inference_time_std=10.0,
            num_test_images=100,
        )

        meets, issues = report.meets_requirements()

        assert meets is False
        assert len(issues) == 1
        assert "Inference" in issues[0]

    def test_meets_requirements_fail_both(self):
        """Test meets_requirements when both requirements fail."""
        from evaluation.evaluate_model import EvaluationReport

        report = EvaluationReport(
            model_path="/path/to/model.pt",
            dataset_path="/path/to/data.yaml",
            overall_map50=0.60,  # Below target
            overall_map50_95=0.50,
            overall_precision=0.70,
            overall_recall=0.65,
            per_class_metrics={},
            inference_time_ms=200.0,  # Above target
            inference_time_std=20.0,
            num_test_images=100,
        )

        meets, issues = report.meets_requirements()

        assert meets is False
        assert len(issues) == 2

    def test_summary_generation(self):
        """Test summary report generation."""
        from evaluation.evaluate_model import EvaluationReport, ClassMetrics

        class_metrics = ClassMetrics(
            class_name="cup",
            precision=0.90,
            recall=0.85,
            f1_score=0.87,
            ap50=0.88,
            ap50_95=0.70,
            num_samples=50,
        )

        report = EvaluationReport(
            model_path="best.pt",
            dataset_path="data.yaml",
            overall_map50=0.88,
            overall_map50_95=0.70,
            overall_precision=0.90,
            overall_recall=0.85,
            per_class_metrics={"cup": class_metrics},
            inference_time_ms=45.0,
            inference_time_std=3.0,
            num_test_images=100,
        )

        summary = report.summary()

        assert "Model Evaluation Report" in summary
        assert "best.pt" in summary
        assert "0.8800" in summary or "0.88" in summary  # mAP50
        assert "45" in summary or "45.0" in summary  # inference time
        assert "cup" in summary

    def test_to_dict(self):
        """Test conversion to dictionary."""
        from evaluation.evaluate_model import EvaluationReport

        report = EvaluationReport(
            model_path="/path/to/model.pt",
            dataset_path="/path/to/data.yaml",
            overall_map50=0.88,
            overall_map50_95=0.70,
            overall_precision=0.90,
            overall_recall=0.85,
            per_class_metrics={},
            inference_time_ms=45.0,
            inference_time_std=3.0,
            num_test_images=100,
        )

        result = report.to_dict()

        assert isinstance(result, dict)
        assert result["model_path"] == "/path/to/model.pt"
        assert result["overall_map50"] == 0.88
        assert result["inference_time_ms"] == 45.0
        assert "meets_requirements" in result
        assert result["meets_requirements"] is True


class TestModelEvaluator:
    """Test ModelEvaluator class."""

    @patch("ultralytics.YOLO")
    def test_init(self, mock_yolo_class):
        """Test evaluator initialization."""
        from evaluation.evaluate_model import ModelEvaluator

        mock_model = MagicMock()
        mock_model.names = {0: "class_0", 1: "class_1"}
        mock_yolo_class.return_value = mock_model

        evaluator = ModelEvaluator(model_path="best.pt", device="cuda")

        assert evaluator.model_path == "best.pt"
        assert evaluator.device == "cuda"
        mock_yolo_class.assert_called_once_with("best.pt")
        assert evaluator.class_names == {0: "class_0", 1: "class_1"}

    @patch("ultralytics.YOLO")
    def test_measure_inference_time(self, mock_yolo_class):
        """Test inference time measurement."""
        from evaluation.evaluate_model import ModelEvaluator

        mock_model = MagicMock()
        mock_model.names = {0: "class_0"}
        mock_yolo_class.return_value = mock_model

        evaluator = ModelEvaluator(model_path="best.pt")

        # Run with small iterations for testing
        mean_ms, std_ms = evaluator.measure_inference_time(
            num_iterations=5,
            warmup=2,
        )

        # Model should have been called warmup + iterations times
        assert mock_model.call_count == 7  # 2 warmup + 5 iterations
        assert mean_ms >= 0  # Time should be positive
        assert std_ms >= 0  # Std should be non-negative

    @patch("ultralytics.YOLO")
    def test_predict_single(self, mock_yolo_class):
        """Test single image prediction."""
        from evaluation.evaluate_model import ModelEvaluator

        mock_model = MagicMock()
        mock_model.names = {0: "cup", 1: "bottle"}

        # Create mock detection results
        mock_box = MagicMock()
        mock_box.cls.item.return_value = 0
        mock_box.conf.item.return_value = 0.95
        mock_box.xyxy = [MagicMock()]
        mock_box.xyxy[0].tolist.return_value = [10, 20, 100, 150]

        mock_result = MagicMock()
        mock_result.boxes = [mock_box]
        mock_model.return_value = [mock_result]

        mock_yolo_class.return_value = mock_model

        evaluator = ModelEvaluator(model_path="best.pt")
        detections = evaluator.predict_single("/path/to/image.jpg")

        assert len(detections) == 1
        assert detections[0]["class_id"] == 0
        assert detections[0]["class_name"] == "cup"
        assert detections[0]["confidence"] == 0.95
        assert detections[0]["bbox"] == [10, 20, 100, 150]

    @patch("ultralytics.YOLO")
    @patch("cv2.imwrite")
    def test_predict_single_with_save(self, mock_imwrite, mock_yolo_class):
        """Test single image prediction with save."""
        from evaluation.evaluate_model import ModelEvaluator

        mock_model = MagicMock()
        mock_model.names = {0: "cup"}

        mock_box = MagicMock()
        mock_box.cls.item.return_value = 0
        mock_box.conf.item.return_value = 0.95
        mock_box.xyxy = [MagicMock()]
        mock_box.xyxy[0].tolist.return_value = [10, 20, 100, 150]

        mock_result = MagicMock()
        mock_result.boxes = [mock_box]
        mock_result.plot.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_model.return_value = [mock_result]

        mock_yolo_class.return_value = mock_model

        evaluator = ModelEvaluator(model_path="best.pt")
        evaluator.predict_single("/path/to/image.jpg", save_path="/path/to/output.jpg")

        mock_result.plot.assert_called_once()
        mock_imwrite.assert_called_once()

    @patch("ultralytics.YOLO")
    def test_evaluate(self, mock_yolo_class, tmp_path):
        """Test full evaluation."""
        from evaluation.evaluate_model import ModelEvaluator

        mock_model = MagicMock()
        mock_model.names = {0: "cup", 1: "bottle"}

        # Mock validation results
        mock_results = MagicMock()
        mock_results.box.map50 = 0.88
        mock_results.box.map = 0.70
        mock_results.box.mp = 0.90
        mock_results.box.mr = 0.85
        mock_results.box.ap50 = np.array([0.90, 0.86])
        mock_results.box.ap = np.array([0.72, 0.68])
        mock_model.val.return_value = mock_results

        mock_yolo_class.return_value = mock_model

        # Create dataset yaml
        dataset_yaml = tmp_path / "data.yaml"
        dataset_yaml.write_text("val: images/val\nnames: ['cup', 'bottle']")

        # Create validation images directory
        val_dir = tmp_path / "images" / "val"
        val_dir.mkdir(parents=True)
        (val_dir / "img1.jpg").touch()
        (val_dir / "img2.jpg").touch()

        evaluator = ModelEvaluator(model_path="best.pt")
        report = evaluator.evaluate(str(dataset_yaml))

        assert report.overall_map50 == 0.88
        assert report.overall_map50_95 == 0.70
        assert report.overall_precision == 0.90
        assert report.overall_recall == 0.85
        assert "cup" in report.per_class_metrics
        assert "bottle" in report.per_class_metrics
        assert report.num_test_images == 2


class TestCompetitionRequirements:
    """Test competition requirement verification."""

    def test_target_values_imported(self):
        """Test that target values are properly imported."""
        from evaluation.evaluate_model import EvaluationReport
        from common.constants import TARGET_MAP50, TARGET_INFERENCE_MS

        # Check that the class uses the imported constants
        assert EvaluationReport._TARGET_MAP50 == TARGET_MAP50
        assert EvaluationReport._TARGET_INFERENCE_MS == TARGET_INFERENCE_MS

    def test_edge_case_exactly_at_threshold(self):
        """Test edge case when metrics are exactly at threshold."""
        from evaluation.evaluate_model import EvaluationReport
        from common.constants import TARGET_MAP50, TARGET_INFERENCE_MS

        report = EvaluationReport(
            model_path="/path/to/model.pt",
            dataset_path="/path/to/data.yaml",
            overall_map50=TARGET_MAP50,  # Exactly at target
            overall_map50_95=0.60,
            overall_precision=0.85,
            overall_recall=0.80,
            per_class_metrics={},
            inference_time_ms=TARGET_INFERENCE_MS,  # Exactly at target
            inference_time_std=5.0,
            num_test_images=100,
        )

        meets, issues = report.meets_requirements()

        # At exactly the threshold, should pass (>= and <=)
        # mAP50: >= TARGET_MAP50 should pass
        # Inference: > TARGET_INFERENCE_MS should fail (exactly equal is OK)
        assert meets is True
        assert len(issues) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
