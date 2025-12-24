"""
Performance benchmark tests for synthetic image generation.

Run via: docker compose run --rm hsr-perception pytest tests/benchmark/ -v

This module compares sequential vs parallel synthetic image generation.
Uses ProcessPoolExecutor-based ParallelSyntheticGenerator for multi-core
acceleration.
"""

import pytest
import time
import sys
from pathlib import Path
from typing import Dict, Tuple

# Add scripts to path for imports
_repo_root = Path(__file__).parent.parent.parent
_scripts_dir = _repo_root / "scripts"
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

from augmentation.copy_paste_augmentor import CopyPasteAugmentor, CopyPasteConfig


class TestSyntheticPerformance:
    """Performance benchmarks for synthetic image generation."""

    def _run_generation(
        self,
        backgrounds_dir: Path,
        annotated_dir: Path,
        output_dir: Path,
        num_images: int,
        class_names: list,
        num_workers: int = 1,
    ) -> Tuple[float, Dict]:
        """
        Run synthetic image generation and measure time.

        Args:
            backgrounds_dir: Directory with background images
            annotated_dir: Directory with annotated objects
            output_dir: Output directory for generated images
            num_images: Number of synthetic images to generate
            class_names: List of class names
            num_workers: Number of parallel workers (1 = sequential)

        Returns:
            Tuple of (elapsed_time, stats_dict)
        """
        # Create config for small-scale generation
        config = CopyPasteConfig(
            synthetic_to_real_ratio=num_images / 3,  # Based on 3 real images
            scale_range=(0.5, 1.5),
            rotation_range=(-15.0, 15.0),
            enable_white_balance=True,
            white_balance_strength=0.7,
            edge_blur_sigma=2.0,
            max_objects_per_image=3,
            min_objects_per_image=1,
            seed=42,
        )

        augmentor = CopyPasteAugmentor(config)

        # Measure generation time
        start_time = time.time()

        stats = augmentor.generate_batch(
            backgrounds_dir=backgrounds_dir,
            annotated_dir=annotated_dir,
            output_dir=output_dir,
            real_image_count=3,  # Match the dummy dataset
            class_names=class_names,
            num_workers=num_workers,
        )

        elapsed_time = time.time() - start_time

        return elapsed_time, stats

    def test_sequential_generation(
        self,
        dummy_background_images,
        dummy_annotated_objects,
        benchmark_temp_dir,
    ):
        """
        Baseline: sequential generation speed.

        This test measures the performance of the current sequential
        implementation.
        """
        annotated_dir, class_names = dummy_annotated_objects
        output_dir = benchmark_temp_dir / "output_sequential"
        output_dir.mkdir(parents=True, exist_ok=True)

        num_images = 20

        elapsed, stats = self._run_generation(
            backgrounds_dir=dummy_background_images,
            annotated_dir=annotated_dir,
            output_dir=output_dir,
            num_images=num_images,
            class_names=class_names,
            num_workers=1,
        )

        # Assertions
        assert elapsed > 0, "Generation should take measurable time"
        assert stats["generated"] > 0, "Should generate at least some images"
        assert stats["failed"] >= 0, "Failed count should be non-negative"

        # Print results for visibility
        print(f"\n{'='*60}")
        print(f"SEQUENTIAL GENERATION BENCHMARK")
        print(f"{'='*60}")
        print(f"Target images:    {num_images}")
        print(f"Generated:        {stats['generated']}")
        print(f"Failed:           {stats['failed']}")
        print(f"Elapsed time:     {elapsed:.3f} seconds")
        print(f"Images/second:    {stats['generated']/elapsed:.2f}")
        print(f"Avg objects/img:  {stats.get('avg_objects_per_image', 0):.2f}")
        print(f"{'='*60}\n")

    def test_parallel_generation_2_workers(
        self,
        dummy_background_images,
        dummy_annotated_objects,
        benchmark_temp_dir,
    ):
        """
        Parallel with 2 workers.

        Expected speedup: ~1.5-1.8x
        """
        annotated_dir, class_names = dummy_annotated_objects
        output_dir = benchmark_temp_dir / "output_parallel_2"
        output_dir.mkdir(parents=True, exist_ok=True)

        num_images = 20

        elapsed, stats = self._run_generation(
            backgrounds_dir=dummy_background_images,
            annotated_dir=annotated_dir,
            output_dir=output_dir,
            num_images=num_images,
            class_names=class_names,
            num_workers=2,
        )

        # Assertions
        assert elapsed > 0, "Generation should take measurable time"
        assert stats["generated"] > 0, "Should generate at least some images"

        # Print results
        print(f"\n{'='*60}")
        print(f"PARALLEL GENERATION (2 workers) BENCHMARK")
        print(f"{'='*60}")
        print(f"Target images:    {num_images}")
        print(f"Generated:        {stats['generated']}")
        print(f"Failed:           {stats['failed']}")
        print(f"Elapsed time:     {elapsed:.3f} seconds")
        print(f"Images/second:    {stats['generated']/elapsed:.2f}")
        print(f"{'='*60}\n")

    def test_parallel_generation_4_workers(
        self,
        dummy_background_images,
        dummy_annotated_objects,
        benchmark_temp_dir,
    ):
        """
        Parallel with 4 workers.

        Expected speedup: ~2.5-3.0x
        """
        annotated_dir, class_names = dummy_annotated_objects
        output_dir = benchmark_temp_dir / "output_parallel_4"
        output_dir.mkdir(parents=True, exist_ok=True)

        num_images = 20

        elapsed, stats = self._run_generation(
            backgrounds_dir=dummy_background_images,
            annotated_dir=annotated_dir,
            output_dir=output_dir,
            num_images=num_images,
            class_names=class_names,
            num_workers=4,
        )

        # Assertions
        assert elapsed > 0, "Generation should take measurable time"
        assert stats["generated"] > 0, "Should generate at least some images"

        # Print results
        print(f"\n{'='*60}")
        print(f"PARALLEL GENERATION (4 workers) BENCHMARK")
        print(f"{'='*60}")
        print(f"Target images:    {num_images}")
        print(f"Generated:        {stats['generated']}")
        print(f"Failed:           {stats['failed']}")
        print(f"Elapsed time:     {elapsed:.3f} seconds")
        print(f"Images/second:    {stats['generated']/elapsed:.2f}")
        print(f"{'='*60}\n")

    def test_speedup_ratio(
        self,
        dummy_background_images,
        dummy_annotated_objects,
        benchmark_temp_dir,
    ):
        """
        Compare speedup between sequential and parallel.

        This test measures the actual speedup achieved by parallelization.
        Typical expected results:
        - 2 workers: 1.5-1.8x speedup
        - 4 workers: 2.5-3.0x speedup

        Note: Actual speedup depends on:
        - CPU core count
        - I/O bottlenecks
        - Memory bandwidth
        - ProcessPoolExecutor overhead
        """
        annotated_dir, class_names = dummy_annotated_objects
        num_images = 20

        results = {}

        # Test configurations
        configs = [
            ("sequential", 1),
            ("parallel_2", 2),
            ("parallel_4", 4),
        ]

        for name, num_workers in configs:
            output_dir = benchmark_temp_dir / f"output_{name}"
            output_dir.mkdir(parents=True, exist_ok=True)

            elapsed, stats = self._run_generation(
                backgrounds_dir=dummy_background_images,
                annotated_dir=annotated_dir,
                output_dir=output_dir,
                num_images=num_images,
                class_names=class_names,
                num_workers=num_workers,
            )

            results[name] = {
                "workers": num_workers,
                "elapsed": elapsed,
                "generated": stats["generated"],
                "images_per_sec": stats["generated"] / elapsed if elapsed > 0 else 0,
            }

        # Calculate speedup ratios
        baseline = results["sequential"]["elapsed"]
        speedup_2 = baseline / results["parallel_2"]["elapsed"]
        speedup_4 = baseline / results["parallel_4"]["elapsed"]

        # Print comparison table
        print(f"\n{'='*70}")
        print(f"SPEEDUP COMPARISON")
        print(f"{'='*70}")
        print(f"{'Configuration':<20} {'Time (s)':<12} {'Images/s':<12} {'Speedup':<10}")
        print(f"{'-'*70}")

        for name, data in results.items():
            speedup = baseline / data["elapsed"] if data["elapsed"] > 0 else 0
            print(
                f"{name:<20} {data['elapsed']:>10.3f}  "
                f"{data['images_per_sec']:>10.2f}  "
                f"{speedup:>8.2f}x"
            )

        print(f"{'='*70}")
        print(f"Expected speedup ranges:")
        print(f"  2 workers: 1.5-1.8x (Actual: {speedup_2:.2f}x)")
        print(f"  4 workers: 2.5-3.0x (Actual: {speedup_4:.2f}x)")
        print(f"{'='*70}\n")

        # Assertions
        assert speedup_2 > 1.0, "2 workers should be faster than sequential"
        assert speedup_4 > speedup_2, "4 workers should be faster than 2 workers"
        assert speedup_4 < 4.0, "Speedup should be sub-linear due to overhead"

    def test_memory_efficiency(
        self,
        dummy_background_images,
        dummy_annotated_objects,
        benchmark_temp_dir,
    ):
        """
        Test memory usage during generation.

        This test validates that the lazy loading implementation
        keeps memory usage reasonable.
        """
        import tracemalloc

        annotated_dir, class_names = dummy_annotated_objects
        output_dir = benchmark_temp_dir / "output_memory"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Start memory tracking
        tracemalloc.start()

        # Take baseline snapshot
        snapshot_before = tracemalloc.take_snapshot()

        # Run generation
        elapsed, stats = self._run_generation(
            backgrounds_dir=dummy_background_images,
            annotated_dir=annotated_dir,
            output_dir=output_dir,
            num_images=10,
            class_names=class_names,
            num_workers=1,
        )

        # Take final snapshot
        snapshot_after = tracemalloc.take_snapshot()

        # Stop tracking
        tracemalloc.stop()

        # Calculate memory usage
        top_stats = snapshot_after.compare_to(snapshot_before, "lineno")

        # Get peak memory usage
        peak_memory = sum(stat.size_diff for stat in top_stats) / (1024 * 1024)  # MB

        print(f"\n{'='*60}")
        print(f"MEMORY EFFICIENCY BENCHMARK")
        print(f"{'='*60}")
        print(f"Images generated:  {stats['generated']}")
        print(f"Peak memory delta: {peak_memory:.2f} MB")
        print(f"Memory per image:  {peak_memory/stats['generated']:.2f} MB")
        print(f"{'='*60}\n")

        # With lazy loading, memory should stay reasonable
        # Small test images should use < 5MB per image
        assert peak_memory / stats["generated"] < 5.0, (
            f"Memory per image too high: {peak_memory/stats['generated']:.2f} MB"
        )

    def test_scalability(
        self,
        dummy_background_images,
        dummy_annotated_objects,
        benchmark_temp_dir,
    ):
        """
        Test performance scaling with different batch sizes.

        This test measures how generation time scales with the number
        of images to generate.
        """
        annotated_dir, class_names = dummy_annotated_objects

        batch_sizes = [5, 10, 20]
        results = []

        for batch_size in batch_sizes:
            output_dir = benchmark_temp_dir / f"output_scale_{batch_size}"
            output_dir.mkdir(parents=True, exist_ok=True)

            elapsed, stats = self._run_generation(
                backgrounds_dir=dummy_background_images,
                annotated_dir=annotated_dir,
                output_dir=output_dir,
                num_images=batch_size,
                class_names=class_names,
                num_workers=1,
            )

            results.append(
                {
                    "batch_size": batch_size,
                    "elapsed": elapsed,
                    "generated": stats["generated"],
                    "time_per_image": elapsed / stats["generated"] if stats["generated"] > 0 else 0,
                }
            )

        # Print results
        print(f"\n{'='*60}")
        print(f"SCALABILITY BENCHMARK")
        print(f"{'='*60}")
        print(f"{'Batch Size':<15} {'Time (s)':<12} {'Time/Image (s)':<15}")
        print(f"{'-'*60}")

        for result in results:
            print(
                f"{result['batch_size']:<15} "
                f"{result['elapsed']:>10.3f}  "
                f"{result['time_per_image']:>13.3f}"
            )

        print(f"{'='*60}\n")

        # Time per image should be relatively consistent
        times_per_image = [r["time_per_image"] for r in results]
        avg_time = sum(times_per_image) / len(times_per_image)
        max_deviation = max(abs(t - avg_time) for t in times_per_image)

        # Allow 50% deviation (some variation is expected)
        assert max_deviation < avg_time * 0.5, (
            f"Time per image varies too much: {max_deviation:.3f}s deviation from {avg_time:.3f}s average"
        )


class TestGenerationQuality:
    """Test quality and correctness of generated images."""

    def test_generated_images_exist(
        self,
        dummy_background_images,
        dummy_annotated_objects,
        benchmark_temp_dir,
    ):
        """Verify that images and labels are actually created."""
        annotated_dir, class_names = dummy_annotated_objects
        output_dir = benchmark_temp_dir / "output_quality"
        output_dir.mkdir(parents=True, exist_ok=True)

        config = CopyPasteConfig(
            synthetic_to_real_ratio=2.0,
            seed=42,
        )
        augmentor = CopyPasteAugmentor(config)

        stats = augmentor.generate_batch(
            backgrounds_dir=dummy_background_images,
            annotated_dir=annotated_dir,
            output_dir=output_dir,
            real_image_count=3,
            class_names=class_names,
        )

        # Check that files were created
        images_dir = output_dir / "images"
        labels_dir = output_dir / "labels"

        assert images_dir.exists(), "Images directory should exist"
        assert labels_dir.exists(), "Labels directory should exist"

        image_files = list(images_dir.glob("*.jpg"))
        label_files = list(labels_dir.glob("*.txt"))

        assert len(image_files) > 0, "Should generate at least one image"
        assert len(label_files) > 0, "Should generate at least one label"
        assert len(image_files) == len(label_files), "Each image should have a label"

        print(f"\nGenerated {len(image_files)} images with labels")

    def test_label_format_valid(
        self,
        dummy_background_images,
        dummy_annotated_objects,
        benchmark_temp_dir,
    ):
        """Verify that generated labels are in valid YOLO format."""
        annotated_dir, class_names = dummy_annotated_objects
        output_dir = benchmark_temp_dir / "output_format"
        output_dir.mkdir(parents=True, exist_ok=True)

        config = CopyPasteConfig(
            synthetic_to_real_ratio=2.0,
            seed=42,
        )
        augmentor = CopyPasteAugmentor(config)

        augmentor.generate_batch(
            backgrounds_dir=dummy_background_images,
            annotated_dir=annotated_dir,
            output_dir=output_dir,
            real_image_count=3,
            class_names=class_names,
        )

        # Check label format
        labels_dir = output_dir / "labels"
        label_files = list(labels_dir.glob("*.txt"))

        assert len(label_files) > 0, "Should have label files"

        # Check first label file
        with open(label_files[0], "r") as f:
            lines = f.readlines()

        assert len(lines) > 0, "Label file should have at least one annotation"

        for line in lines:
            parts = line.strip().split()
            assert len(parts) == 5, f"YOLO format should have 5 values, got {len(parts)}"

            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

            # Validate ranges
            assert 0 <= class_id < len(class_names), f"Invalid class_id: {class_id}"
            assert 0.0 <= x_center <= 1.0, f"x_center out of range: {x_center}"
            assert 0.0 <= y_center <= 1.0, f"y_center out of range: {y_center}"
            assert 0.0 < width <= 1.0, f"width out of range: {width}"
            assert 0.0 < height <= 1.0, f"height out of range: {height}"

        print(f"\nValidated {len(label_files)} label files in YOLO format")
