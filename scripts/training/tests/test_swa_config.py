#!/usr/bin/env python3
"""
Tests for SWA Configuration

Tests the adaptive SWA start epoch calculation and SWAConfig validation.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

# Mock torch and ultralytics modules before importing swa_trainer
sys.modules['torch'] = MagicMock()
sys.modules['torch.nn'] = MagicMock()
sys.modules['torch.optim'] = MagicMock()
sys.modules['torch.optim.swa_utils'] = MagicMock()
sys.modules['ultralytics'] = MagicMock()
sys.modules['ultralytics.utils'] = MagicMock()

# Add parent directory to path to import swa_trainer
sys.path.insert(0, str(Path(__file__).parent.parent))

from swa_trainer import SWAConfig, calculate_adaptive_swa_start_epoch


class TestAdaptiveSWAStartEpoch(unittest.TestCase):
    """Test calculate_adaptive_swa_start_epoch function."""

    def test_standard_30_epochs(self):
        """Test with 30 epochs -> should be 6 (20%)."""
        result = calculate_adaptive_swa_start_epoch(30)
        self.assertEqual(result, 6, "30 epochs should give 6 SWA epochs (20%)")

    def test_standard_50_epochs(self):
        """Test with 50 epochs -> should be 10 (20%)."""
        result = calculate_adaptive_swa_start_epoch(50)
        self.assertEqual(result, 10, "50 epochs should give 10 SWA epochs (20%)")

    def test_large_100_epochs(self):
        """Test with 100 epochs -> should be capped at 15 (max)."""
        result = calculate_adaptive_swa_start_epoch(100)
        self.assertEqual(result, 15, "100 epochs should be capped at max 15 SWA epochs")

    def test_small_20_epochs(self):
        """Test with 20 epochs -> should be 5 (minimum)."""
        result = calculate_adaptive_swa_start_epoch(20)
        self.assertEqual(result, 5, "20 epochs should give minimum 5 SWA epochs")

    def test_very_small_10_epochs(self):
        """Test with 10 epochs -> should be 5 (minimum)."""
        result = calculate_adaptive_swa_start_epoch(10)
        self.assertEqual(result, 5, "10 epochs should give minimum 5 SWA epochs")

    def test_very_large_200_epochs(self):
        """Test with 200 epochs -> should be capped at 15 (max)."""
        result = calculate_adaptive_swa_start_epoch(200)
        self.assertEqual(result, 15, "200 epochs should be capped at max 15 SWA epochs")

    def test_custom_fraction(self):
        """Test with custom fraction (30%)."""
        result = calculate_adaptive_swa_start_epoch(50, swa_fraction=0.3)
        self.assertEqual(result, 15, "50 epochs with 30% should give 15 SWA epochs")

    def test_custom_min_max(self):
        """Test with custom min/max bounds."""
        result = calculate_adaptive_swa_start_epoch(
            50, min_swa_epochs=3, max_swa_epochs=8
        )
        self.assertEqual(result, 8, "Should be capped at custom max 8")

    def test_boundary_15_epochs(self):
        """Test with 15 epochs (boundary case)."""
        result = calculate_adaptive_swa_start_epoch(15)
        # 15 * 0.2 = 3, but min is 5
        self.assertEqual(result, 5, "15 epochs should give minimum 5 SWA epochs")

    def test_boundary_75_epochs(self):
        """Test with 75 epochs (boundary case)."""
        result = calculate_adaptive_swa_start_epoch(75)
        # 75 * 0.2 = 15, exactly at max
        self.assertEqual(result, 15, "75 epochs should give exactly 15 SWA epochs")


class TestSWAConfig(unittest.TestCase):
    """Test SWAConfig dataclass."""

    def test_default_config(self):
        """Test default configuration."""
        config = SWAConfig()
        self.assertFalse(config.enabled)
        self.assertEqual(config.swa_start_epoch, 10)
        self.assertEqual(config.swa_lr, 0.0005)
        self.assertTrue(config.update_bn)
        self.assertEqual(config.min_total_epochs, 15)

    def test_auto_calculate_config(self):
        """Test configuration with auto-calculate (swa_start_epoch=0)."""
        config = SWAConfig(enabled=True, swa_start_epoch=0)
        self.assertTrue(config.enabled)
        self.assertEqual(config.swa_start_epoch, 0)

    def test_explicit_swa_start_epoch(self):
        """Test configuration with explicit swa_start_epoch."""
        config = SWAConfig(enabled=True, swa_start_epoch=15)
        self.assertEqual(config.swa_start_epoch, 15)

    def test_invalid_negative_swa_start_epoch(self):
        """Test that negative swa_start_epoch raises ValueError."""
        with self.assertRaises(ValueError) as context:
            SWAConfig(swa_start_epoch=-1)
        self.assertIn("swa_start_epoch must be >= 0", str(context.exception))

    def test_invalid_zero_swa_lr(self):
        """Test that zero swa_lr raises ValueError."""
        with self.assertRaises(ValueError) as context:
            SWAConfig(swa_lr=0.0)
        self.assertIn("swa_lr must be > 0", str(context.exception))

    def test_invalid_negative_swa_lr(self):
        """Test that negative swa_lr raises ValueError."""
        with self.assertRaises(ValueError) as context:
            SWAConfig(swa_lr=-0.001)
        self.assertIn("swa_lr must be > 0", str(context.exception))

    def test_custom_min_total_epochs(self):
        """Test configuration with custom min_total_epochs."""
        config = SWAConfig(min_total_epochs=20)
        self.assertEqual(config.min_total_epochs, 20)


class TestSWAConfigShouldEnable(unittest.TestCase):
    """Test SWAConfig.should_enable_for_epochs method."""

    def test_disabled_config(self):
        """Test that disabled config returns False."""
        config = SWAConfig(enabled=False)
        self.assertFalse(config.should_enable_for_epochs(50))

    def test_enabled_sufficient_epochs(self):
        """Test that enabled config with sufficient epochs returns True."""
        config = SWAConfig(enabled=True, min_total_epochs=15)
        self.assertTrue(config.should_enable_for_epochs(50))

    def test_enabled_insufficient_epochs(self):
        """Test that enabled config with insufficient epochs returns False."""
        config = SWAConfig(enabled=True, min_total_epochs=15)
        self.assertFalse(config.should_enable_for_epochs(10))

    def test_boundary_exact_min_epochs(self):
        """Test boundary case with exactly min_total_epochs."""
        config = SWAConfig(enabled=True, min_total_epochs=15)
        self.assertTrue(
            config.should_enable_for_epochs(15),
            "Should enable when total_epochs equals min_total_epochs",
        )

    def test_boundary_one_below_min(self):
        """Test boundary case with one epoch below minimum."""
        config = SWAConfig(enabled=True, min_total_epochs=15)
        self.assertFalse(
            config.should_enable_for_epochs(14),
            "Should not enable when total_epochs is below min_total_epochs",
        )

    def test_custom_min_total_epochs(self):
        """Test with custom min_total_epochs."""
        config = SWAConfig(enabled=True, min_total_epochs=20)
        self.assertFalse(config.should_enable_for_epochs(15))
        self.assertTrue(config.should_enable_for_epochs(20))
        self.assertTrue(config.should_enable_for_epochs(50))


class TestIntegration(unittest.TestCase):
    """Integration tests for SWA configuration and calculation."""

    def test_typical_competition_scenario(self):
        """Test typical competition scenario (50 epochs)."""
        config = SWAConfig(enabled=True, swa_start_epoch=0, min_total_epochs=15)
        total_epochs = 50

        # Check if should enable
        self.assertTrue(config.should_enable_for_epochs(total_epochs))

        # Calculate adaptive start epoch
        swa_start_epoch = calculate_adaptive_swa_start_epoch(total_epochs)
        self.assertEqual(swa_start_epoch, 10)

        # Calculate SWA start point
        swa_start = total_epochs - swa_start_epoch
        self.assertEqual(swa_start, 40, "SWA should start at epoch 40")

    def test_fast_test_scenario(self):
        """Test fast test scenario (30 epochs)."""
        config = SWAConfig(enabled=True, swa_start_epoch=0, min_total_epochs=15)
        total_epochs = 30

        # Check if should enable
        self.assertTrue(config.should_enable_for_epochs(total_epochs))

        # Calculate adaptive start epoch
        swa_start_epoch = calculate_adaptive_swa_start_epoch(total_epochs)
        self.assertEqual(swa_start_epoch, 6)

        # Calculate SWA start point
        swa_start = total_epochs - swa_start_epoch
        self.assertEqual(swa_start, 24, "SWA should start at epoch 24")

    def test_insufficient_epochs_scenario(self):
        """Test scenario with insufficient epochs (10 epochs)."""
        config = SWAConfig(enabled=True, swa_start_epoch=0, min_total_epochs=15)
        total_epochs = 10

        # Check if should enable (should be False)
        self.assertFalse(
            config.should_enable_for_epochs(total_epochs),
            "SWA should be disabled for 10 epochs (< min 15)",
        )

    def test_explicit_swa_start_epoch_scenario(self):
        """Test scenario with explicit swa_start_epoch (not auto-calculate)."""
        config = SWAConfig(enabled=True, swa_start_epoch=15, min_total_epochs=15)
        total_epochs = 50

        # Check if should enable
        self.assertTrue(config.should_enable_for_epochs(total_epochs))

        # Use explicit value (not auto-calculated)
        swa_start_epoch = config.swa_start_epoch
        self.assertEqual(swa_start_epoch, 15)

        # Calculate SWA start point
        swa_start = total_epochs - swa_start_epoch
        self.assertEqual(swa_start, 35, "SWA should start at epoch 35")


def run_tests():
    """Run all tests and return exit code."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestAdaptiveSWAStartEpoch))
    suite.addTests(loader.loadTestsFromTestCase(TestSWAConfig))
    suite.addTests(loader.loadTestsFromTestCase(TestSWAConfigShouldEnable))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Return exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())
