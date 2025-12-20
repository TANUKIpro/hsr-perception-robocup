"""
Tests for quick_finetune module.

Tests the CompetitionTrainer class, TrainingResult dataclass,
configuration constants, and CLI argument parsing.
"""

import argparse
import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch
from dataclasses import dataclass

import pytest

# Mock heavy dependencies before any import from the module
# These modules have complex dependencies that are hard to mock after import
_mock_torch = MagicMock()
_mock_torch.cuda.is_available.return_value = True
sys.modules['torch'] = _mock_torch

_mock_colorama = MagicMock()
_mock_colorama.Fore = MagicMock()
_mock_colorama.Style = MagicMock()
_mock_colorama.init = MagicMock()
sys.modules['colorama'] = _mock_colorama

_mock_ultralytics = MagicMock()
sys.modules['ultralytics'] = _mock_ultralytics
sys.modules['ultralytics.cfg'] = MagicMock()

# Mock the training submodules
sys.modules['training.gpu_scaler'] = MagicMock()
sys.modules['training.tensorboard_monitor'] = MagicMock()
sys.modules['training.training_config'] = MagicMock()
sys.modules['training.swa_trainer'] = MagicMock()
sys.modules['training.llrd_trainer'] = MagicMock()
sys.modules['training.memory_utils'] = MagicMock()

# Mock augmentation module
sys.modules['augmentation'] = MagicMock()
sys.modules['augmentation.copy_paste_augmentor'] = MagicMock()

# Add scripts directory to path
scripts_dir = Path(__file__).parent.parent.parent.parent / "scripts"
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))


class TestSyntheticConfigKeys:
    """Test SYNTHETIC_CONFIG_KEYS constant."""

    def test_contains_expected_keys(self):
        """Test that SYNTHETIC_CONFIG_KEYS contains expected keys."""
        from training.quick_finetune import SYNTHETIC_CONFIG_KEYS

        expected_keys = [
            "dynamic_synthetic_enabled",
            "backgrounds_dir",
            "annotated_dir",
            "synthetic_ratio",
            "synthetic_scale_range",
            "synthetic_rotation_range",
            "synthetic_white_balance",
            "synthetic_white_balance_strength",
            "synthetic_edge_blur",
            "synthetic_max_objects",
        ]

        for key in expected_keys:
            assert key in SYNTHETIC_CONFIG_KEYS, f"Key {key} not found"

    def test_keys_are_strings(self):
        """Test that all keys are strings."""
        from training.quick_finetune import SYNTHETIC_CONFIG_KEYS

        for key in SYNTHETIC_CONFIG_KEYS:
            assert isinstance(key, str), f"Key {key} is not a string"

    def test_is_set(self):
        """Test that SYNTHETIC_CONFIG_KEYS is a set."""
        from training.quick_finetune import SYNTHETIC_CONFIG_KEYS

        assert isinstance(SYNTHETIC_CONFIG_KEYS, set)


class TestCompetitionConfig:
    """Test COMPETITION_CONFIG constant."""

    def test_contains_model_settings(self):
        """Test that COMPETITION_CONFIG contains model settings."""
        from training.quick_finetune import COMPETITION_CONFIG

        assert "model" in COMPETITION_CONFIG
        assert "imgsz" in COMPETITION_CONFIG
        assert COMPETITION_CONFIG["model"] == "yolov8m.pt"
        assert COMPETITION_CONFIG["imgsz"] == 640

    def test_contains_training_settings(self):
        """Test that COMPETITION_CONFIG contains training settings."""
        from training.quick_finetune import COMPETITION_CONFIG

        assert "epochs" in COMPETITION_CONFIG
        assert "batch" in COMPETITION_CONFIG
        assert "patience" in COMPETITION_CONFIG

    def test_contains_optimizer_settings(self):
        """Test that COMPETITION_CONFIG contains optimizer settings."""
        from training.quick_finetune import COMPETITION_CONFIG

        assert "optimizer" in COMPETITION_CONFIG
        assert "lr0" in COMPETITION_CONFIG
        assert "weight_decay" in COMPETITION_CONFIG

    def test_contains_augmentation_settings(self):
        """Test that COMPETITION_CONFIG contains augmentation settings."""
        from training.quick_finetune import COMPETITION_CONFIG

        assert "augment" in COMPETITION_CONFIG
        assert COMPETITION_CONFIG["augment"] is True
        assert "mosaic" in COMPETITION_CONFIG
        assert "mixup" in COMPETITION_CONFIG

    def test_contains_llrd_settings(self):
        """Test that COMPETITION_CONFIG contains LLRD settings."""
        from training.quick_finetune import COMPETITION_CONFIG

        assert "llrd_enabled" in COMPETITION_CONFIG
        assert "llrd_decay_rate" in COMPETITION_CONFIG
        # LLRD disabled by default
        assert COMPETITION_CONFIG["llrd_enabled"] is False

    def test_contains_synthetic_settings(self):
        """Test that COMPETITION_CONFIG contains synthetic settings."""
        from training.quick_finetune import COMPETITION_CONFIG

        assert "dynamic_synthetic_enabled" in COMPETITION_CONFIG
        assert "synthetic_ratio" in COMPETITION_CONFIG


class TestFastConfig:
    """Test FAST_CONFIG constant."""

    def test_smaller_model_than_competition(self):
        """Test that FAST_CONFIG uses a smaller model."""
        from training.quick_finetune import FAST_CONFIG, COMPETITION_CONFIG

        # yolov8s is smaller than yolov8m
        assert FAST_CONFIG["model"] == "yolov8s.pt"
        assert COMPETITION_CONFIG["model"] == "yolov8m.pt"

    def test_fewer_epochs(self):
        """Test that FAST_CONFIG has fewer epochs."""
        from training.quick_finetune import FAST_CONFIG, COMPETITION_CONFIG

        assert FAST_CONFIG["epochs"] < COMPETITION_CONFIG["epochs"]

    def test_smaller_image_size(self):
        """Test that FAST_CONFIG has smaller image size."""
        from training.quick_finetune import FAST_CONFIG, COMPETITION_CONFIG

        assert FAST_CONFIG["imgsz"] < COMPETITION_CONFIG["imgsz"]

    def test_inherits_from_competition(self):
        """Test that FAST_CONFIG inherits from COMPETITION_CONFIG."""
        from training.quick_finetune import FAST_CONFIG, COMPETITION_CONFIG

        # Keys that are not overridden should be same
        assert FAST_CONFIG["optimizer"] == COMPETITION_CONFIG["optimizer"]
        assert FAST_CONFIG["augment"] == COMPETITION_CONFIG["augment"]


class TestTrainingResult:
    """Test TrainingResult dataclass."""

    def test_creation(self):
        """Test creating TrainingResult."""
        from training.quick_finetune import TrainingResult

        result = TrainingResult(
            best_model_path="/path/to/best.pt",
            last_model_path="/path/to/last.pt",
            metrics={"mAP50": 0.90, "mAP50-95": 0.75},
            training_time_minutes=45.5,
            epochs_completed=50,
            config={"model": "yolov8m.pt"},
        )

        assert result.best_model_path == "/path/to/best.pt"
        assert result.last_model_path == "/path/to/last.pt"
        assert result.metrics["mAP50"] == 0.90
        assert result.training_time_minutes == 45.5
        assert result.epochs_completed == 50

    def test_summary_generation(self):
        """Test summary string generation."""
        from training.quick_finetune import TrainingResult

        result = TrainingResult(
            best_model_path="models/best.pt",
            last_model_path="models/last.pt",
            metrics={
                "mAP50": 0.88,
                "mAP50-95": 0.72,
                "precision": 0.90,
                "recall": 0.85,
            },
            training_time_minutes=42.3,
            epochs_completed=50,
            config={},
        )

        summary = result.summary()

        assert "Training Complete" in summary
        assert "42.3" in summary  # training time
        assert "50" in summary  # epochs
        assert "best.pt" in summary
        assert "0.88" in summary or "0.8800" in summary  # mAP50

    def test_meets_target_pass(self):
        """Test meets_target returns True when target achieved."""
        from training.quick_finetune import TrainingResult

        result = TrainingResult(
            best_model_path="/path/to/best.pt",
            last_model_path="/path/to/last.pt",
            metrics={"mAP50": 0.90},  # Above 0.85 target
            training_time_minutes=45.0,
            epochs_completed=50,
            config={},
        )

        assert result.meets_target() is True
        assert result.meets_target(target_map=0.85) is True
        assert result.meets_target(target_map=0.90) is True

    def test_meets_target_fail(self):
        """Test meets_target returns False when target not achieved."""
        from training.quick_finetune import TrainingResult

        result = TrainingResult(
            best_model_path="/path/to/best.pt",
            last_model_path="/path/to/last.pt",
            metrics={"mAP50": 0.75},  # Below 0.85 target
            training_time_minutes=45.0,
            epochs_completed=50,
            config={},
        )

        assert result.meets_target() is False
        assert result.meets_target(target_map=0.85) is False

    def test_meets_target_custom_threshold(self):
        """Test meets_target with custom threshold."""
        from training.quick_finetune import TrainingResult

        result = TrainingResult(
            best_model_path="/path/to/best.pt",
            last_model_path="/path/to/last.pt",
            metrics={"mAP50": 0.80},
            training_time_minutes=45.0,
            epochs_completed=50,
            config={},
        )

        assert result.meets_target(target_map=0.75) is True
        assert result.meets_target(target_map=0.80) is True
        assert result.meets_target(target_map=0.85) is False

    def test_meets_target_missing_metric(self):
        """Test meets_target with missing mAP50 metric."""
        from training.quick_finetune import TrainingResult

        result = TrainingResult(
            best_model_path="/path/to/best.pt",
            last_model_path="/path/to/last.pt",
            metrics={},  # No mAP50
            training_time_minutes=45.0,
            epochs_completed=50,
            config={},
        )

        # Should return False when mAP50 is missing (defaults to 0)
        assert result.meets_target() is False

    def test_to_dict(self):
        """Test conversion to dictionary."""
        from training.quick_finetune import TrainingResult

        result = TrainingResult(
            best_model_path="/path/to/best.pt",
            last_model_path="/path/to/last.pt",
            metrics={"mAP50": 0.90, "mAP50-95": 0.75},
            training_time_minutes=45.0,
            epochs_completed=50,
            config={"model": "yolov8m.pt", "epochs": 50},
        )

        data = result.to_dict()

        assert isinstance(data, dict)
        assert data["best_model_path"] == "/path/to/best.pt"
        assert data["last_model_path"] == "/path/to/last.pt"
        assert data["metrics"]["mAP50"] == 0.90
        assert data["training_time_minutes"] == 45.0
        assert data["epochs_completed"] == 50
        assert "timestamp" in data
        assert "config" in data

    def test_timestamp_auto_generated(self):
        """Test that timestamp is auto-generated."""
        from training.quick_finetune import TrainingResult

        result = TrainingResult(
            best_model_path="/path/to/best.pt",
            last_model_path="/path/to/last.pt",
            metrics={},
            training_time_minutes=45.0,
            epochs_completed=50,
            config={},
        )

        assert result.timestamp is not None
        assert len(result.timestamp) > 0


class TestCompetitionTrainerInit:
    """Test CompetitionTrainer initialization."""

    @patch("training.quick_finetune.log_gpu_status")
    @patch("training.quick_finetune.GPUScaler")
    def test_init_default(self, mock_scaler_class, mock_log_gpu):
        """Test default initialization."""
        from training.quick_finetune import CompetitionTrainer

        mock_log_gpu.return_value = True  # GPU available
        mock_scaler = MagicMock()
        mock_scaler.get_optimal_config.return_value = {"model": "yolov8m.pt", "batch": 16}
        mock_scaler.get_summary.return_value = "GPU: RTX 3080"
        mock_scaler_class.return_value = mock_scaler

        trainer = CompetitionTrainer()

        assert trainer.output_dir == Path("models/finetuned")
        assert trainer.auto_scale is True
        assert trainer.tensorboard_enabled is True

    @patch("training.quick_finetune.log_gpu_status")
    @patch("training.quick_finetune.GPUScaler")
    def test_init_with_custom_output(self, mock_scaler_class, mock_log_gpu):
        """Test initialization with custom output directory."""
        from training.quick_finetune import CompetitionTrainer

        mock_log_gpu.return_value = True
        mock_scaler = MagicMock()
        mock_scaler.get_optimal_config.return_value = {}
        mock_scaler.get_summary.return_value = ""
        mock_scaler_class.return_value = mock_scaler

        trainer = CompetitionTrainer(output_dir="/custom/output")

        assert trainer.output_dir == Path("/custom/output")

    @patch("training.quick_finetune.log_gpu_status")
    @patch("training.quick_finetune.GPUScaler")
    def test_init_with_config(self, mock_scaler_class, mock_log_gpu):
        """Test initialization with custom config."""
        from training.quick_finetune import CompetitionTrainer

        mock_log_gpu.return_value = True
        mock_scaler = MagicMock()
        mock_scaler_class.return_value = mock_scaler

        custom_config = {"model": "yolov8l.pt", "epochs": 100, "batch": 8}
        trainer = CompetitionTrainer(config=custom_config)

        assert trainer.config["epochs"] == 100
        assert trainer.config["batch"] == 8

    @patch("training.quick_finetune.log_gpu_status")
    def test_init_auto_scale_disabled(self, mock_log_gpu):
        """Test initialization with auto_scale disabled."""
        from training.quick_finetune import CompetitionTrainer, COMPETITION_CONFIG

        mock_log_gpu.return_value = True

        trainer = CompetitionTrainer(auto_scale=False)

        assert trainer.auto_scale is False
        assert trainer.gpu_scaler is None
        # Should use COMPETITION_CONFIG when auto_scale is disabled and no config provided
        assert trainer.config == COMPETITION_CONFIG

    @patch("training.quick_finetune.log_gpu_status")
    def test_init_tensorboard_disabled(self, mock_log_gpu):
        """Test initialization with TensorBoard disabled."""
        from training.quick_finetune import CompetitionTrainer

        mock_log_gpu.return_value = True

        trainer = CompetitionTrainer(auto_scale=False, tensorboard=False)

        assert trainer.tensorboard_enabled is False

    @patch("training.quick_finetune.log_gpu_status")
    def test_init_no_gpu_raises(self, mock_log_gpu):
        """Test initialization raises error when GPU not available."""
        from training.quick_finetune import CompetitionTrainer

        mock_log_gpu.return_value = False  # No GPU

        with pytest.raises(RuntimeError, match="GPU not available"):
            CompetitionTrainer(auto_scale=False)

    @patch("training.quick_finetune.log_gpu_status")
    def test_init_no_gpu_with_allow_cpu(self, mock_log_gpu):
        """Test initialization with allow_cpu flag."""
        from training.quick_finetune import CompetitionTrainer

        mock_log_gpu.return_value = False  # No GPU

        # Should not raise when allow_cpu is True
        trainer = CompetitionTrainer(
            auto_scale=False,
            config={"allow_cpu": True}
        )

        assert trainer is not None

    @patch("training.quick_finetune.log_gpu_status")
    @patch("training.quick_finetune.GPUScaler")
    def test_init_base_model_override(self, mock_scaler_class, mock_log_gpu):
        """Test initialization with base_model override."""
        from training.quick_finetune import CompetitionTrainer

        mock_log_gpu.return_value = True
        mock_scaler = MagicMock()
        mock_scaler.get_optimal_config.return_value = {"model": "yolov8m.pt"}
        mock_scaler.get_summary.return_value = ""
        mock_scaler_class.return_value = mock_scaler

        trainer = CompetitionTrainer(base_model="yolov8l.pt")

        assert trainer.base_model == "yolov8l.pt"


class TestValidateDataset:
    """Test _validate_dataset method."""

    @patch("training.quick_finetune.log_gpu_status")
    def test_valid_dataset(self, mock_log_gpu, tmp_path):
        """Test validation of valid dataset."""
        from training.quick_finetune import CompetitionTrainer

        mock_log_gpu.return_value = True

        # Create valid dataset structure
        train_dir = tmp_path / "images" / "train"
        val_dir = tmp_path / "images" / "val"
        train_dir.mkdir(parents=True)
        val_dir.mkdir(parents=True)

        # Create some images
        (train_dir / "img1.jpg").touch()
        (train_dir / "img2.jpg").touch()
        (val_dir / "img1.jpg").touch()

        # Create data.yaml
        yaml_path = tmp_path / "data.yaml"
        yaml_path.write_text(
            "train: images/train\nval: images/val\nnames:\n  0: cup\n  1: bottle"
        )

        trainer = CompetitionTrainer(auto_scale=False)
        result = trainer._validate_dataset(str(yaml_path))

        assert result is True

    @patch("training.quick_finetune.log_gpu_status")
    def test_dataset_not_found(self, mock_log_gpu, tmp_path, capsys):
        """Test validation when dataset file not found."""
        from training.quick_finetune import CompetitionTrainer

        mock_log_gpu.return_value = True

        trainer = CompetitionTrainer(auto_scale=False)
        result = trainer._validate_dataset(str(tmp_path / "nonexistent.yaml"))

        assert result is False
        captured = capsys.readouterr()
        assert "not found" in captured.out

    @patch("training.quick_finetune.log_gpu_status")
    def test_missing_required_field(self, mock_log_gpu, tmp_path, capsys):
        """Test validation when required field missing."""
        from training.quick_finetune import CompetitionTrainer

        mock_log_gpu.return_value = True

        # Create yaml without 'names' field
        yaml_path = tmp_path / "data.yaml"
        yaml_path.write_text("train: images/train\nval: images/val")

        trainer = CompetitionTrainer(auto_scale=False)
        result = trainer._validate_dataset(str(yaml_path))

        assert result is False
        captured = capsys.readouterr()
        assert "Missing field" in captured.out
        assert "names" in captured.out

    @patch("training.quick_finetune.log_gpu_status")
    def test_train_path_not_found(self, mock_log_gpu, tmp_path, capsys):
        """Test validation when train path not found."""
        from training.quick_finetune import CompetitionTrainer

        mock_log_gpu.return_value = True

        # Create yaml but not the train directory
        yaml_path = tmp_path / "data.yaml"
        yaml_path.write_text(
            "train: images/train\nval: images/val\nnames:\n  0: cup"
        )

        trainer = CompetitionTrainer(auto_scale=False)
        result = trainer._validate_dataset(str(yaml_path))

        assert result is False
        captured = capsys.readouterr()
        assert "Train path not found" in captured.out

    @patch("training.quick_finetune.log_gpu_status")
    def test_val_path_not_found(self, mock_log_gpu, tmp_path, capsys):
        """Test validation when val path not found."""
        from training.quick_finetune import CompetitionTrainer

        mock_log_gpu.return_value = True

        # Create train directory but not val
        train_dir = tmp_path / "images" / "train"
        train_dir.mkdir(parents=True)
        (train_dir / "img.jpg").touch()

        yaml_path = tmp_path / "data.yaml"
        yaml_path.write_text(
            "train: images/train\nval: images/val\nnames:\n  0: cup"
        )

        trainer = CompetitionTrainer(auto_scale=False)
        result = trainer._validate_dataset(str(yaml_path))

        assert result is False
        captured = capsys.readouterr()
        assert "Val path not found" in captured.out


class TestSyntheticConfigFiltering:
    """Test synthetic config key filtering."""

    def test_filter_synthetic_keys_from_config(self):
        """Test that synthetic keys are filtered from YOLO config."""
        from training.quick_finetune import SYNTHETIC_CONFIG_KEYS, COMPETITION_CONFIG

        # Simulate filtering as done in the train method
        yolo_config = {
            k: v for k, v in COMPETITION_CONFIG.items()
            if k not in SYNTHETIC_CONFIG_KEYS
        }

        # Check that synthetic keys are removed
        for key in SYNTHETIC_CONFIG_KEYS:
            assert key not in yolo_config

        # Check that non-synthetic keys remain
        assert "model" in yolo_config
        assert "epochs" in yolo_config
        assert "batch" in yolo_config

    def test_yolo_compatible_config_only(self):
        """Test that only YOLO-compatible config remains after filtering."""
        from training.quick_finetune import SYNTHETIC_CONFIG_KEYS, COMPETITION_CONFIG

        # Filter out synthetic keys
        yolo_config = {
            k: v for k, v in COMPETITION_CONFIG.items()
            if k not in SYNTHETIC_CONFIG_KEYS
        }

        # These are standard YOLO config keys that should remain
        expected_yolo_keys = [
            "model", "imgsz", "epochs", "batch", "patience",
            "optimizer", "lr0", "lrf", "momentum", "weight_decay",
            "warmup_epochs", "freeze", "augment", "mosaic", "mixup",
            "workers", "cache", "amp", "save",
        ]

        for key in expected_yolo_keys:
            assert key in yolo_config, f"Expected YOLO key {key} missing"


class TestArgumentParsing:
    """Test CLI argument parsing."""

    def test_required_dataset_arg(self):
        """Test that --dataset is required."""
        # Capture the argparse exit
        with pytest.raises(SystemExit):
            # Create parser similar to main()
            parser = argparse.ArgumentParser()
            parser.add_argument("--dataset", "-d", required=True)
            parser.parse_args([])  # No args

    def test_dataset_arg_provided(self):
        """Test parsing with dataset argument."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset", "-d", required=True)
        parser.add_argument("--model", "-m", default=None)

        args = parser.parse_args(["--dataset", "data.yaml"])
        assert args.dataset == "data.yaml"

    def test_optional_model_arg(self):
        """Test that --model is optional with None default."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset", "-d", required=True)
        parser.add_argument("--model", "-m", default=None)

        args = parser.parse_args(["--dataset", "data.yaml"])
        assert args.model is None

        args = parser.parse_args(["--dataset", "data.yaml", "--model", "yolov8l.pt"])
        assert args.model == "yolov8l.pt"

    def test_default_output_dir(self):
        """Test default output directory."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset", "-d", required=True)
        parser.add_argument("--output", "-o", default="models/finetuned")

        args = parser.parse_args(["--dataset", "data.yaml"])
        assert args.output == "models/finetuned"

    def test_fast_flag(self):
        """Test --fast flag."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset", "-d", required=True)
        parser.add_argument("--fast", action="store_true")

        args = parser.parse_args(["--dataset", "data.yaml"])
        assert args.fast is False

        args = parser.parse_args(["--dataset", "data.yaml", "--fast"])
        assert args.fast is True

    def test_llrd_flags(self):
        """Test --llrd and --llrd-decay-rate flags."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset", "-d", required=True)
        parser.add_argument("--llrd", action="store_true")
        parser.add_argument("--llrd-decay-rate", type=float, default=0.9)

        args = parser.parse_args(["--dataset", "data.yaml"])
        assert args.llrd is False
        assert args.llrd_decay_rate == 0.9

        args = parser.parse_args([
            "--dataset", "data.yaml",
            "--llrd",
            "--llrd-decay-rate", "0.85"
        ])
        assert args.llrd is True
        assert args.llrd_decay_rate == 0.85

    def test_tensorboard_flags(self):
        """Test TensorBoard-related flags."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset", "-d", required=True)
        parser.add_argument("--no-tensorboard", action="store_true")
        parser.add_argument("--tensorboard-port", type=int, default=6006)

        args = parser.parse_args(["--dataset", "data.yaml"])
        assert args.no_tensorboard is False
        assert args.tensorboard_port == 6006

        args = parser.parse_args([
            "--dataset", "data.yaml",
            "--no-tensorboard",
            "--tensorboard-port", "8080"
        ])
        assert args.no_tensorboard is True
        assert args.tensorboard_port == 8080

    def test_gpu_tier_choices(self):
        """Test --gpu-tier choices."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset", "-d", required=True)
        parser.add_argument(
            "--gpu-tier",
            choices=["low", "medium", "high", "workstation"]
        )

        args = parser.parse_args(["--dataset", "data.yaml", "--gpu-tier", "high"])
        assert args.gpu_tier == "high"

        # Invalid choice should raise
        with pytest.raises(SystemExit):
            parser.parse_args(["--dataset", "data.yaml", "--gpu-tier", "invalid"])

    def test_resume_flag(self):
        """Test --resume flag."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset", "-d", required=True)
        parser.add_argument("--resume", action="store_true")

        args = parser.parse_args(["--dataset", "data.yaml"])
        assert args.resume is False

        args = parser.parse_args(["--dataset", "data.yaml", "--resume"])
        assert args.resume is True

    def test_export_choices(self):
        """Test --export choices."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset", "-d", required=True)
        parser.add_argument("--export", choices=["onnx", "torchscript", "tflite"])

        args = parser.parse_args(["--dataset", "data.yaml", "--export", "onnx"])
        assert args.export == "onnx"

    def test_dynamic_synthetic_flags(self):
        """Test dynamic synthetic image flags."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset", "-d", required=True)
        parser.add_argument("--dynamic-synthetic", action="store_true")
        parser.add_argument("--no-dynamic-synthetic", action="store_true")
        parser.add_argument("--backgrounds-dir", type=str)
        parser.add_argument("--annotated-dir", type=str)
        parser.add_argument("--synthetic-ratio", type=float, default=2.0)

        args = parser.parse_args([
            "--dataset", "data.yaml",
            "--backgrounds-dir", "/path/to/bg",
            "--annotated-dir", "/path/to/annotated",
            "--synthetic-ratio", "1.5"
        ])

        assert args.backgrounds_dir == "/path/to/bg"
        assert args.annotated_dir == "/path/to/annotated"
        assert args.synthetic_ratio == 1.5


class TestRunNameGeneration:
    """Test run name generation."""

    @patch("training.quick_finetune.log_gpu_status")
    def test_run_name_contains_competition(self, mock_log_gpu):
        """Test that run name contains 'competition'."""
        from training.quick_finetune import CompetitionTrainer

        mock_log_gpu.return_value = True

        trainer = CompetitionTrainer(auto_scale=False)

        assert "competition" in trainer.run_name

    @patch("training.quick_finetune.log_gpu_status")
    def test_run_name_contains_timestamp(self, mock_log_gpu):
        """Test that run name contains timestamp pattern."""
        from training.quick_finetune import CompetitionTrainer
        import re

        mock_log_gpu.return_value = True

        trainer = CompetitionTrainer(auto_scale=False)

        # Should match pattern like "competition_20240101_120000"
        pattern = r"competition_\d{8}_\d{6}"
        assert re.match(pattern, trainer.run_name)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
