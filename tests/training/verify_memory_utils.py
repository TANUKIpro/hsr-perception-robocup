#!/usr/bin/env python3
"""
Simple verification script for memory_utils without requiring torch/pytest.

This script verifies that:
1. The module can be imported
2. Basic functions are callable
3. Safe cleanup functions work with None inputs
"""

import sys
from pathlib import Path

# Add scripts directory to path
scripts_dir = Path(__file__).parent.parent.parent / "scripts"
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))


def verify_basic_imports():
    """Verify basic imports work."""
    print("Testing imports...")
    try:
        from training.memory_utils import (
            MemoryStats,
            MemoryTracker,
            cleanup_model,
            cleanup_optimizer,
            cleanup_swa_model,
            cleanup_tensorboard,
            full_training_cleanup,
        )
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def verify_safe_cleanup():
    """Verify cleanup functions are safe with None inputs."""
    print("\nTesting safe cleanup with None inputs...")
    try:
        from training.memory_utils import (
            cleanup_model,
            cleanup_optimizer,
            cleanup_swa_model,
            cleanup_tensorboard,
        )

        # All these should be safe to call with None
        cleanup_model(None)
        print("✓ cleanup_model(None) - OK")

        cleanup_optimizer(None)
        print("✓ cleanup_optimizer(None) - OK")

        cleanup_swa_model(None)
        print("✓ cleanup_swa_model(None) - OK")

        cleanup_tensorboard(None, None)
        print("✓ cleanup_tensorboard(None, None) - OK")

        return True
    except Exception as e:
        print(f"✗ Safe cleanup failed: {e}")
        return False


def verify_memory_tracker():
    """Verify MemoryTracker can be instantiated."""
    print("\nTesting MemoryTracker...")
    try:
        from training.memory_utils import MemoryTracker

        tracker = MemoryTracker()
        assert tracker.snapshots == []
        print("✓ MemoryTracker created successfully")

        # Test clear
        tracker.clear()
        print("✓ MemoryTracker.clear() - OK")

        # Test report generation with no snapshots
        report = tracker.generate_report()
        assert "No memory snapshots" in report
        print("✓ MemoryTracker.generate_report() - OK")

        return True
    except Exception as e:
        print(f"✗ MemoryTracker failed: {e}")
        return False


def verify_full_cleanup():
    """Verify full_training_cleanup works with None inputs."""
    print("\nTesting full_training_cleanup...")
    try:
        from training.memory_utils import full_training_cleanup

        # Should work with all None
        stats = full_training_cleanup()
        assert isinstance(stats, dict)
        assert "objects_cleaned" in stats
        print("✓ full_training_cleanup() with all None - OK")
        print(f"  Returned stats: {stats}")

        return True
    except Exception as e:
        print(f"✗ full_training_cleanup failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all verification tests."""
    print("="*60)
    print("Memory Utils Verification")
    print("="*60)

    results = []
    results.append(("Imports", verify_basic_imports()))
    results.append(("Safe Cleanup", verify_safe_cleanup()))
    results.append(("MemoryTracker", verify_memory_tracker()))
    results.append(("Full Cleanup", verify_full_cleanup()))

    print("\n" + "="*60)
    print("Summary")
    print("="*60)

    all_passed = True
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:20s}: {status}")
        if not passed:
            all_passed = False

    print("="*60)

    if all_passed:
        print("\n✓ All verification tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
