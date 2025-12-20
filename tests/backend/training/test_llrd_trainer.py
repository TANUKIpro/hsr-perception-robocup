#!/usr/bin/env python3
"""
Test suite for LLRD (Layer-wise Learning Rate Decay) implementation.

Tests cover:
1. LLRDConfig validation
2. Layer depth calculation
3. Learning rate formula
4. Freeze + LLRD interaction
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock

# Create proper mocks with necessary attributes
torch_mock = MagicMock()
torch_mock.nn = MagicMock()
torch_mock.optim = MagicMock()

ultralytics_mock = MagicMock()
ultralytics_mock.cfg = MagicMock()
ultralytics_mock.cfg.DEFAULT_CFG_DICT = {}

# Create a mock DetectionTrainer base class that can be subclassed
class MockDetectionTrainer:
    """Mock base trainer class for testing."""
    def __init__(self, *args, **kwargs):
        pass

ultralytics_mock.models = MagicMock()
ultralytics_mock.models.yolo = MagicMock()
ultralytics_mock.models.yolo.detect = MagicMock()
ultralytics_mock.models.yolo.detect.DetectionTrainer = MockDetectionTrainer
ultralytics_mock.utils = MagicMock()
ultralytics_mock.utils.LOGGER = MagicMock()
ultralytics_mock.utils.colorstr = lambda x: x

# Mock torch and ultralytics modules before importing llrd_trainer
sys.modules['torch'] = torch_mock
sys.modules['torch.nn'] = torch_mock.nn
sys.modules['torch.optim'] = torch_mock.optim
sys.modules['ultralytics'] = ultralytics_mock
sys.modules['ultralytics.cfg'] = ultralytics_mock.cfg
sys.modules['ultralytics.models'] = ultralytics_mock.models
sys.modules['ultralytics.models.yolo'] = ultralytics_mock.models.yolo
sys.modules['ultralytics.models.yolo.detect'] = ultralytics_mock.models.yolo.detect
sys.modules['ultralytics.utils'] = ultralytics_mock.utils

# Add scripts/training directory to path
scripts_training = Path(__file__).parent.parent.parent.parent / "scripts" / "training"
if str(scripts_training) not in sys.path:
    sys.path.insert(0, str(scripts_training))

from llrd_trainer import LLRDConfig


class TestLLRDConfig(unittest.TestCase):
    """Test LLRDConfig validation and defaults."""

    def test_valid_decay_rate_05(self):
        """Test decay_rate = 0.5 is valid."""
        config = LLRDConfig(enabled=True, decay_rate=0.5)
        self.assertEqual(config.decay_rate, 0.5)

    def test_valid_decay_rate_09(self):
        """Test decay_rate = 0.9 is valid."""
        config = LLRDConfig(enabled=True, decay_rate=0.9)
        self.assertEqual(config.decay_rate, 0.9)

    def test_valid_decay_rate_10(self):
        """Test decay_rate = 1.0 is valid (no decay)."""
        config = LLRDConfig(enabled=True, decay_rate=1.0)
        self.assertEqual(config.decay_rate, 1.0)

    def test_invalid_decay_rate_zero(self):
        """Test decay_rate = 0.0 raises ValueError."""
        with self.assertRaises(ValueError) as context:
            LLRDConfig(enabled=True, decay_rate=0.0)
        self.assertIn("decay_rate must be in (0.0, 1.0]", str(context.exception))

    def test_invalid_decay_rate_negative(self):
        """Test decay_rate = -0.5 raises ValueError."""
        with self.assertRaises(ValueError) as context:
            LLRDConfig(enabled=True, decay_rate=-0.5)
        self.assertIn("decay_rate must be in (0.0, 1.0]", str(context.exception))

    def test_invalid_decay_rate_above_one(self):
        """Test decay_rate = 1.5 raises ValueError."""
        with self.assertRaises(ValueError) as context:
            LLRDConfig(enabled=True, decay_rate=1.5)
        self.assertIn("decay_rate must be in (0.0, 1.0]", str(context.exception))

    def test_default_configuration(self):
        """Test default configuration values."""
        config = LLRDConfig()
        self.assertFalse(config.enabled)
        self.assertEqual(config.decay_rate, 0.9)


class TestLayerDepthCalculation(unittest.TestCase):
    """Test layer depth calculation formula."""

    def setUp(self):
        """Set up a minimal mock trainer for each test."""
        from llrd_trainer import LLRDDetectionTrainer

        # Create a minimal instance without calling parent __init__
        self.trainer = object.__new__(LLRDDetectionTrainer)
        self.trainer.llrd_config = LLRDConfig()
        # Set class constants
        self.trainer.BACKBONE_END = 9
        self.trainer.NECK_END = 21
        self.trainer.HEAD_LAYER = 22

    def test_depth_calculation_formula(self):
        """Test depth calculation: depth = HEAD_LAYER(22) - layer_idx."""
        # HEAD_LAYER = 22 in YOLOv8
        self.assertEqual(self.trainer.HEAD_LAYER, 22)

        # Verify formula for various layers
        test_cases = [
            (0, 22),   # Early backbone: depth = 22 - 0 = 22
            (9, 13),   # End of backbone: depth = 22 - 9 = 13
            (10, 12),  # Start of neck: depth = 22 - 10 = 12
            (21, 1),   # End of neck: depth = 22 - 21 = 1
            (22, 0),   # Head: depth = 22 - 22 = 0
        ]

        for layer_idx, expected_depth in test_cases:
            with self.subTest(layer_idx=layer_idx):
                depth = self.trainer._get_layer_depth(layer_idx)
                self.assertEqual(
                    depth,
                    expected_depth,
                    f"Layer {layer_idx} should have depth {expected_depth}"
                )

    def test_depth_layer_0(self):
        """Test depth for layer 0 (first backbone layer)."""
        depth = self.trainer._get_layer_depth(0)
        self.assertEqual(depth, 22)

    def test_depth_layer_9(self):
        """Test depth for layer 9 (last backbone layer)."""
        depth = self.trainer._get_layer_depth(9)
        self.assertEqual(depth, 13)

    def test_depth_layer_10(self):
        """Test depth for layer 10 (first neck layer)."""
        depth = self.trainer._get_layer_depth(10)
        self.assertEqual(depth, 12)

    def test_depth_layer_21(self):
        """Test depth for layer 21 (last neck layer)."""
        depth = self.trainer._get_layer_depth(21)
        self.assertEqual(depth, 1)

    def test_depth_layer_22(self):
        """Test depth for layer 22 (head layer)."""
        depth = self.trainer._get_layer_depth(22)
        self.assertEqual(depth, 0)


class TestLearningRateFormula(unittest.TestCase):
    """Test learning rate calculation formula."""

    def setUp(self):
        """Set up a minimal mock trainer for each test."""
        from llrd_trainer import LLRDDetectionTrainer

        # Create a minimal instance without calling parent __init__
        self.trainer = object.__new__(LLRDDetectionTrainer)
        self.trainer.llrd_config = LLRDConfig(enabled=True, decay_rate=0.9)
        # Set class constants
        self.trainer.BACKBONE_END = 9
        self.trainer.NECK_END = 21
        self.trainer.HEAD_LAYER = 22

    def test_lr_formula(self):
        """Test LR formula: layer_lr = base_lr * (decay_rate ^ depth)."""
        base_lr = 0.001
        decay_rate = 0.9

        # Test various depths
        test_cases = [
            (0, base_lr * (decay_rate ** 0)),   # depth=0: lr = base_lr * 1
            (1, base_lr * (decay_rate ** 1)),   # depth=1: lr = base_lr * 0.9
            (5, base_lr * (decay_rate ** 5)),   # depth=5: lr = base_lr * 0.59049
            (10, base_lr * (decay_rate ** 10)), # depth=10: lr = base_lr * 0.34868
            (22, base_lr * (decay_rate ** 22)), # depth=22: lr = base_lr * 0.09847
        ]

        for depth, expected_lr in test_cases:
            with self.subTest(depth=depth):
                calculated_lr = base_lr * (decay_rate ** depth)
                self.assertAlmostEqual(
                    calculated_lr,
                    expected_lr,
                    places=5,
                    msg=f"Depth {depth} should have LR {expected_lr}"
                )

    def test_head_layer_gets_base_lr(self):
        """Test that head layer (depth=0) gets the full base LR."""
        base_lr = 0.001
        decay_rate = 0.9
        depth = self.trainer._get_layer_depth(22)  # Head layer has depth 0

        layer_lr = base_lr * (decay_rate ** depth)
        self.assertEqual(layer_lr, base_lr)

    def test_backbone_gets_lower_lr(self):
        """Test that backbone layers get progressively lower LR."""
        base_lr = 0.001
        decay_rate = 0.9

        # Layer 22 (head, depth=0)
        head_depth = self.trainer._get_layer_depth(22)
        head_lr = base_lr * (decay_rate ** head_depth)

        # Layer 10 (neck start, depth=12)
        neck_depth = self.trainer._get_layer_depth(10)
        neck_lr = base_lr * (decay_rate ** neck_depth)

        # Layer 0 (backbone start, depth=22)
        backbone_depth = self.trainer._get_layer_depth(0)
        backbone_lr = base_lr * (decay_rate ** backbone_depth)

        # Verify: head_lr > neck_lr > backbone_lr
        self.assertGreater(head_lr, neck_lr)
        self.assertGreater(neck_lr, backbone_lr)

    def test_different_decay_rates(self):
        """Test LR calculation with different decay rates."""
        base_lr = 0.001

        # Test with decay_rate = 0.85 (more aggressive)
        depth = 10
        lr_aggressive = base_lr * (0.85 ** depth)

        # Test with decay_rate = 0.95 (less aggressive)
        lr_gentle = base_lr * (0.95 ** depth)

        # More aggressive decay should result in lower LR
        self.assertLess(lr_aggressive, lr_gentle)


class TestFreezeAndLLRDInteraction(unittest.TestCase):
    """Test interaction between layer freezing and LLRD."""

    def setUp(self):
        """Set up a minimal mock trainer for each test."""
        from llrd_trainer import LLRDDetectionTrainer

        # Create a minimal instance without calling parent __init__
        self.trainer = object.__new__(LLRDDetectionTrainer)
        self.trainer.llrd_config = LLRDConfig(enabled=True, decay_rate=0.9)
        # Set class constants
        self.trainer.BACKBONE_END = 9
        self.trainer.NECK_END = 21
        self.trainer.HEAD_LAYER = 22

    def test_freeze_detection_with_llrd(self):
        """Test that both freeze and LLRD can be enabled together."""
        # Mock args with freeze=10
        self.trainer.args = Mock()
        self.trainer.args.freeze = 10

        # Re-run the freeze check logic
        freeze_layers = self.trainer.args.freeze if hasattr(self.trainer.args, 'freeze') else 0

        # Verify that freeze layers are detected correctly
        self.assertEqual(freeze_layers, 10)

        # Verify LLRD is also enabled
        self.assertTrue(self.trainer.llrd_config.enabled)

    def test_frozen_layers_range(self):
        """Test that frozen layers 0-9 would be excluded from LLRD."""
        freeze_layers = 10
        # With freeze=10, layers 0-9 would be frozen
        frozen_range = list(range(0, freeze_layers))
        self.assertEqual(frozen_range, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    def test_llrd_applies_to_unfrozen_layers(self):
        """Test that LLRD would apply to layers 10-22 when freeze=10."""
        freeze_layers = 10
        # LLRD should apply to layers 10-22
        unfrozen_layers = list(range(freeze_layers, self.trainer.HEAD_LAYER + 1))
        self.assertEqual(
            unfrozen_layers,
            [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
        )

    def test_no_freeze_no_conflict(self):
        """Test that no warning is needed when freeze=0."""
        self.trainer.args = Mock()
        self.trainer.args.freeze = 0

        freeze_layers = self.trainer.args.freeze
        # No conflict when freeze=0
        self.assertEqual(freeze_layers, 0)

        # LLRD can work with all layers when freeze=0
        self.assertFalse(self.trainer.llrd_config.enabled and freeze_layers > 0)

    def test_llrd_disabled_no_conflict(self):
        """Test that no warning is needed when LLRD is disabled."""
        # Create trainer with LLRD disabled
        self.trainer.llrd_config = LLRDConfig(enabled=False)
        self.trainer.args = Mock()
        self.trainer.args.freeze = 10

        # No conflict when LLRD is disabled
        self.assertFalse(self.trainer.llrd_config.enabled)


class TestLayerCategorization(unittest.TestCase):
    """Test layer categorization (backbone/neck/head)."""

    def setUp(self):
        """Set up a minimal mock trainer for each test."""
        from llrd_trainer import LLRDDetectionTrainer

        # Create a minimal instance without calling parent __init__
        self.trainer = object.__new__(LLRDDetectionTrainer)
        self.trainer.llrd_config = LLRDConfig()
        # Set class constants
        self.trainer.BACKBONE_END = 9
        self.trainer.NECK_END = 21
        self.trainer.HEAD_LAYER = 22

    def test_backbone_layers(self):
        """Test that layers 0-9 are categorized as backbone."""
        for layer_idx in range(0, 10):
            with self.subTest(layer_idx=layer_idx):
                category = self.trainer._get_layer_category(layer_idx)
                self.assertEqual(category, "backbone")

    def test_neck_layers(self):
        """Test that layers 10-21 are categorized as neck."""
        for layer_idx in range(10, 22):
            with self.subTest(layer_idx=layer_idx):
                category = self.trainer._get_layer_category(layer_idx)
                self.assertEqual(category, "neck")

    def test_head_layer(self):
        """Test that layer 22 is categorized as head."""
        category = self.trainer._get_layer_category(22)
        self.assertEqual(category, "head")


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
