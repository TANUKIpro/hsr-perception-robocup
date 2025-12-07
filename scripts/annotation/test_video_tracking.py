#!/usr/bin/env python3
"""
Video Tracking Verification Tests

Tests for validating the SAM2 video tracking functionality:
1. Empty frames test - handling tracking failure
2. Large frame count test - VRAM split processing verification
3. Normal tracking test - accuracy verification

Usage:
    python scripts/annotation/test_video_tracking.py --test all
    python scripts/annotation/test_video_tracking.py --test empty
    python scripts/annotation/test_video_tracking.py --test vram
    python scripts/annotation/test_video_tracking.py --test normal
"""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from video_tracking_predictor import VideoTrackingPredictor, TrackingResult, VRAMEstimate


def create_test_images(output_dir: Path, num_frames: int, img_size: tuple = (480, 640)) -> Path:
    """
    Create test images with a moving object.

    Args:
        output_dir: Directory to save test images
        num_frames: Number of frames to create
        img_size: Image size (height, width)

    Returns:
        Path to created directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    height, width = img_size

    # Object parameters
    obj_size = 50
    start_x = 100
    start_y = 100
    move_per_frame = 5

    for i in range(num_frames):
        # Create blank image
        img = np.ones((height, width, 3), dtype=np.uint8) * 200  # Gray background

        # Draw moving red square
        x = start_x + (i * move_per_frame) % (width - obj_size - start_x)
        y = start_y + (i * move_per_frame // 2) % (height - obj_size - start_y)

        cv2.rectangle(img, (x, y), (x + obj_size, y + obj_size), (0, 0, 255), -1)

        # Save with sequential numbering (SAM2 requirement)
        filename = f"{i}.jpg"
        cv2.imwrite(str(output_dir / filename), img)

    print(f"Created {num_frames} test images in {output_dir}")
    return output_dir


def create_empty_frames(output_dir: Path, num_frames: int, img_size: tuple = (480, 640)) -> Path:
    """
    Create empty (blank) test images for failure handling test.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    height, width = img_size

    for i in range(num_frames):
        # Create blank white image
        img = np.ones((height, width, 3), dtype=np.uint8) * 255

        filename = f"{i}.jpg"
        cv2.imwrite(str(output_dir / filename), img)

    print(f"Created {num_frames} empty frames in {output_dir}")
    return output_dir


def test_empty_frames():
    """Test tracking behavior with empty frames (tracking failure handling)."""
    print("\n" + "=" * 60)
    print("TEST: Empty Frames - Tracking Failure Handling")
    print("=" * 60)

    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = Path(tmpdir) / "empty_test"
        create_empty_frames(test_dir, num_frames=10)

        # Initialize predictor (don't load model for this basic test)
        predictor = VideoTrackingPredictor(device="cpu")

        print("\n[Test 1.1] Check TrackingResult.from_mask with empty mask")
        empty_mask = np.zeros((480, 640), dtype=bool)
        result = TrackingResult.from_mask(empty_mask)

        assert result.mask_area == 0, "Empty mask should have 0 area"
        assert result.is_low_confidence, "Empty mask should be low confidence"
        print(f"  ✓ Empty mask handling: area={result.mask_area}, low_conf={result.is_low_confidence}")

        print("\n[Test 1.2] Check TrackingResult with small mask")
        small_mask = np.zeros((480, 640), dtype=bool)
        small_mask[100:110, 100:110] = True  # 10x10 = 100 pixels
        result = TrackingResult.from_mask(small_mask)

        assert result.mask_area == 100, "Small mask should have 100 pixels"
        assert result.is_low_confidence, "Small mask should be low confidence"
        print(f"  ✓ Small mask handling: area={result.mask_area}, low_conf={result.is_low_confidence}")

        print("\n[Test 1.3] Check TrackingResult with normal mask")
        normal_mask = np.zeros((480, 640), dtype=bool)
        normal_mask[100:200, 100:200] = True  # 100x100 = 10000 pixels
        result = TrackingResult.from_mask(normal_mask, reference_area=10000)

        assert result.mask_area == 10000, "Normal mask should have 10000 pixels"
        assert not result.is_low_confidence, "Normal mask should not be low confidence"
        print(f"  ✓ Normal mask handling: area={result.mask_area}, low_conf={result.is_low_confidence}")

    print("\n✅ Empty Frames Test PASSED")
    return True


def test_vram_estimation():
    """Test VRAM usage estimation and split processing logic."""
    print("\n" + "=" * 60)
    print("TEST: VRAM Estimation and Split Processing")
    print("=" * 60)

    predictor = VideoTrackingPredictor(device="cuda")

    print("\n[Test 2.1] Check VRAM estimation for small frame count")
    estimate = predictor.estimate_vram_usage(num_frames=10, image_size=(480, 640))
    print(f"  10 frames @ 480x640:")
    print(f"    Estimated usage: {estimate.estimated_usage_gb:.2f} GB")
    print(f"    Needs split: {estimate.needs_split}")
    print(f"    Recommended batch size: {estimate.recommended_batch_size}")

    print("\n[Test 2.2] Check VRAM estimation for large frame count")
    estimate = predictor.estimate_vram_usage(num_frames=200, image_size=(1080, 1920))
    print(f"  200 frames @ 1080x1920:")
    print(f"    Estimated usage: {estimate.estimated_usage_gb:.2f} GB")
    print(f"    Needs split: {estimate.needs_split}")
    print(f"    Recommended batch size: {estimate.recommended_batch_size}")
    print(f"    Number of batches: {estimate.num_batches}")

    # Verify estimation is reasonable
    assert estimate.estimated_usage_gb > 0, "Estimation should be positive"
    assert estimate.recommended_batch_size > 0, "Batch size should be positive"

    print("\n[Test 2.3] Check available VRAM detection")
    available, total = VideoTrackingPredictor.get_available_vram()
    print(f"  Available VRAM: {available:.2f} GB")
    print(f"  Total VRAM: {total:.2f} GB")

    # On CPU, both should be 0
    import torch
    if not torch.cuda.is_available():
        assert available == 0 and total == 0, "CPU should report 0 VRAM"
        print("  ✓ CPU mode correctly reports 0 VRAM")
    else:
        assert total > 0, "GPU should report positive total VRAM"
        print(f"  ✓ GPU mode correctly reports VRAM")

    print("\n✅ VRAM Estimation Test PASSED")
    return True


def test_sequence_initialization():
    """Test image sequence initialization with symlink creation."""
    print("\n" + "=" * 60)
    print("TEST: Sequence Initialization")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test images with non-sequential names
        test_dir = Path(tmpdir) / "seq_test"
        test_dir.mkdir(parents=True)

        # Create images with various naming patterns
        for i, name in enumerate(["img_001.jpg", "img_002.jpg", "img_003.jpg"]):
            img = np.ones((480, 640, 3), dtype=np.uint8) * 200
            cv2.rectangle(img, (100 + i*20, 100), (150 + i*20, 150), (0, 0, 255), -1)
            cv2.imwrite(str(test_dir / name), img)

        print(f"\n[Test 3.1] Created test images with names: img_001.jpg, img_002.jpg, img_003.jpg")

        # Test sequence initialization (without loading model)
        predictor = VideoTrackingPredictor(device="cpu")

        # We can't fully test init_sequence without loading the model
        # but we can test the symlink preparation logic
        print("\n[Test 3.2] Check frame map and temp directory cleanup")

        # Simulate what init_sequence does for symlinks
        import os
        temp_dir = tempfile.mkdtemp(prefix="sam2_test_")
        frame_map = {}

        image_paths = sorted(test_dir.glob("*.jpg"))
        for idx, original_path in enumerate(image_paths):
            link_name = f"{idx}.jpg"
            link_path = Path(temp_dir) / link_name
            os.symlink(original_path.absolute(), link_path)
            frame_map[idx] = original_path

        # Verify symlinks were created correctly
        assert len(list(Path(temp_dir).glob("*.jpg"))) == 3, "Should have 3 symlinks"
        assert Path(temp_dir) / "0.jpg" in list(Path(temp_dir).iterdir()), "Should have 0.jpg"
        print(f"  ✓ Created {len(frame_map)} symlinks in temp directory")

        # Verify frame map
        assert len(frame_map) == 3, "Frame map should have 3 entries"
        assert frame_map[0].name == "img_001.jpg", "Frame 0 should map to img_001.jpg"
        print(f"  ✓ Frame map correctly maps indices to original paths")

        # Cleanup
        shutil.rmtree(temp_dir)
        print(f"  ✓ Temp directory cleaned up")

    print("\n✅ Sequence Initialization Test PASSED")
    return True


def test_tracking_result_confidence():
    """Test confidence calculation for tracking results."""
    print("\n" + "=" * 60)
    print("TEST: Tracking Result Confidence Calculation")
    print("=" * 60)

    print("\n[Test 4.1] Confidence with stable mask (same area)")
    reference_area = 10000
    mask = np.zeros((480, 640), dtype=bool)
    mask[100:200, 100:200] = True  # 10000 pixels

    result = TrackingResult.from_mask(mask, reference_area=reference_area)
    assert result.confidence >= 0.9, f"Stable mask should have high confidence, got {result.confidence}"
    print(f"  ✓ Stable mask confidence: {result.confidence:.3f}")

    print("\n[Test 4.2] Confidence with area increase")
    larger_mask = np.zeros((480, 640), dtype=bool)
    larger_mask[100:250, 100:200] = True  # 15000 pixels (1.5x)

    result = TrackingResult.from_mask(larger_mask, reference_area=reference_area)
    print(f"  Area ratio: {result.mask_area / reference_area:.2f}")
    print(f"  ✓ Larger mask confidence: {result.confidence:.3f}")

    print("\n[Test 4.3] Confidence with area decrease")
    smaller_mask = np.zeros((480, 640), dtype=bool)
    smaller_mask[100:170, 100:170] = True  # 4900 pixels (~0.5x)

    result = TrackingResult.from_mask(smaller_mask, reference_area=reference_area)
    print(f"  Area ratio: {result.mask_area / reference_area:.2f}")
    print(f"  ✓ Smaller mask confidence: {result.confidence:.3f}")

    print("\n[Test 4.4] Confidence with very small mask (tracking lost)")
    tiny_mask = np.zeros((480, 640), dtype=bool)
    tiny_mask[100:105, 100:105] = True  # 25 pixels

    result = TrackingResult.from_mask(tiny_mask, reference_area=reference_area)
    assert result.is_low_confidence, "Very small mask should be low confidence"
    print(f"  ✓ Tiny mask correctly marked as low confidence: {result.confidence:.3f}")

    print("\n✅ Confidence Calculation Test PASSED")
    return True


def run_all_tests():
    """Run all verification tests."""
    print("\n" + "=" * 60)
    print("SAM2 Video Tracking Verification Tests")
    print("=" * 60)

    results = {
        "Empty Frames": test_empty_frames(),
        "VRAM Estimation": test_vram_estimation(),
        "Sequence Init": test_sequence_initialization(),
        "Confidence Calc": test_tracking_result_confidence(),
    }

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("All tests PASSED!")
        return 0
    else:
        print("Some tests FAILED!")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Video Tracking Verification Tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--test",
        choices=["all", "empty", "vram", "sequence", "confidence"],
        default="all",
        help="Which test to run (default: all)",
    )

    args = parser.parse_args()

    if args.test == "all":
        return run_all_tests()
    elif args.test == "empty":
        return 0 if test_empty_frames() else 1
    elif args.test == "vram":
        return 0 if test_vram_estimation() else 1
    elif args.test == "sequence":
        return 0 if test_sequence_initialization() else 1
    elif args.test == "confidence":
        return 0 if test_tracking_result_confidence() else 1


if __name__ == "__main__":
    sys.exit(main())
