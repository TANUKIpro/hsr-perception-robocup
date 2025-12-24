"""
Synthetic Ratio Benchmark Script

Evaluates the impact of different synthetic_ratio values on:
- Training time (synthetic generation + YOLO training)
- Model performance (mAP)

Usage:
    python scripts/evaluation/synthetic_ratio_benchmark.py \
        --dataset path/to/data.yaml \
        --backgrounds path/to/backgrounds \
        --annotated path/to/annotated \
        --output results/ratio_benchmark.json
"""

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Add scripts directory to path
_scripts_dir = Path(__file__).parent.parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

from training.config_manager import COMPETITION_CONFIG
from training.quick_finetune import CompetitionTrainer


@dataclass
class RatioBenchmarkResult:
    """Result of a single ratio benchmark run."""

    ratio: float
    synthetic_count: int
    generation_time_seconds: float
    training_time_seconds: float
    total_time_seconds: float
    mAP50: Optional[float] = None
    mAP50_95: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    error: Optional[str] = None


def run_benchmark(
    dataset_yaml: Path,
    backgrounds_dir: Path,
    annotated_dir: Path,
    ratio: float,
    epochs: int = 20,
    output_dir: Optional[Path] = None,
) -> RatioBenchmarkResult:
    """
    Run a single benchmark with the specified ratio.

    Args:
        dataset_yaml: Path to data.yaml
        backgrounds_dir: Path to backgrounds directory
        annotated_dir: Path to annotated directory
        ratio: Synthetic to real ratio
        epochs: Number of training epochs
        output_dir: Output directory for models

    Returns:
        RatioBenchmarkResult with timing and metrics
    """
    print(f"\n{'='*60}")
    print(f"Running benchmark: ratio={ratio}")
    print(f"{'='*60}")

    # Create config with specified ratio
    config = COMPETITION_CONFIG.copy()
    config["synthetic_ratio"] = ratio
    config["backgrounds_dir"] = str(backgrounds_dir)
    config["annotated_dir"] = str(annotated_dir)
    config["dynamic_synthetic_enabled"] = True
    config["epochs"] = epochs
    config["patience"] = epochs  # Disable early stopping for fair comparison

    # Create output directory
    if output_dir is None:
        output_dir = Path("models/ratio_benchmark")
    run_output = output_dir / f"ratio_{ratio:.1f}"
    run_output.mkdir(parents=True, exist_ok=True)

    try:
        # Initialize trainer
        trainer = CompetitionTrainer(
            base_model=config.get("model", "yolov8m.pt"),
            output_dir=str(run_output),
            auto_scale=True,
            training_config=config,
        )

        # Time the training (includes synthetic generation)
        start_time = time.time()

        # Run training
        results = trainer.train(
            dataset_yaml=str(dataset_yaml),
            epochs=epochs,
            run_name=f"ratio_{ratio:.1f}",
        )

        total_time = time.time() - start_time

        # Extract metrics
        mAP50 = results.get("metrics/mAP50(B)", None)
        mAP50_95 = results.get("metrics/mAP50-95(B)", None)
        precision = results.get("metrics/precision(B)", None)
        recall = results.get("metrics/recall(B)", None)

        # Calculate synthetic count (estimate)
        dataset_path = Path(dataset_yaml).parent
        train_images = dataset_path / "images" / "train"
        real_count = len([f for f in train_images.glob("*") if not f.name.startswith(("dynamic_synth_", "cached_synth_"))])
        synthetic_count = int(real_count * ratio)

        # Estimate generation time (rough approximation)
        # In a real scenario, we'd instrument the generation separately
        estimated_gen_time = synthetic_count * 0.1  # ~0.1s per image estimate

        return RatioBenchmarkResult(
            ratio=ratio,
            synthetic_count=synthetic_count,
            generation_time_seconds=estimated_gen_time,
            training_time_seconds=total_time - estimated_gen_time,
            total_time_seconds=total_time,
            mAP50=mAP50,
            mAP50_95=mAP50_95,
            precision=precision,
            recall=recall,
        )

    except Exception as e:
        return RatioBenchmarkResult(
            ratio=ratio,
            synthetic_count=0,
            generation_time_seconds=0,
            training_time_seconds=0,
            total_time_seconds=0,
            error=str(e),
        )


def run_all_benchmarks(
    dataset_yaml: Path,
    backgrounds_dir: Path,
    annotated_dir: Path,
    ratios: List[float],
    epochs: int = 20,
    output_dir: Optional[Path] = None,
) -> List[RatioBenchmarkResult]:
    """
    Run benchmarks for all specified ratios.

    Args:
        dataset_yaml: Path to data.yaml
        backgrounds_dir: Path to backgrounds directory
        annotated_dir: Path to annotated directory
        ratios: List of ratios to benchmark
        epochs: Number of training epochs
        output_dir: Output directory for models

    Returns:
        List of RatioBenchmarkResult
    """
    results = []

    for ratio in ratios:
        result = run_benchmark(
            dataset_yaml=dataset_yaml,
            backgrounds_dir=backgrounds_dir,
            annotated_dir=annotated_dir,
            ratio=ratio,
            epochs=epochs,
            output_dir=output_dir,
        )
        results.append(result)

        # Print summary
        if result.error:
            print(f"  ERROR: {result.error}")
        else:
            print(f"  Synthetic images: {result.synthetic_count}")
            print(f"  Total time: {result.total_time_seconds:.1f}s")
            print(f"  mAP50: {result.mAP50:.3f}" if result.mAP50 else "  mAP50: N/A")
            print(f"  mAP50-95: {result.mAP50_95:.3f}" if result.mAP50_95 else "  mAP50-95: N/A")

    return results


def save_results(results: List[RatioBenchmarkResult], output_path: Path) -> None:
    """Save benchmark results to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "timestamp": datetime.now().isoformat(),
        "results": [asdict(r) for r in results],
        "summary": {
            "best_mAP50": max((r.mAP50 for r in results if r.mAP50), default=None),
            "best_ratio_mAP50": next(
                (r.ratio for r in sorted(results, key=lambda x: x.mAP50 or 0, reverse=True) if r.mAP50),
                None,
            ),
            "fastest_ratio": min(
                (r for r in results if not r.error),
                key=lambda x: x.total_time_seconds,
                default=None,
            ),
        },
    }

    if data["summary"]["fastest_ratio"]:
        data["summary"]["fastest_ratio"] = data["summary"]["fastest_ratio"].ratio

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to: {output_path}")


def print_comparison_table(results: List[RatioBenchmarkResult]) -> None:
    """Print a comparison table of results."""
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS COMPARISON")
    print("=" * 80)
    print(f"{'Ratio':<8} {'Synth#':<8} {'Time(s)':<10} {'mAP50':<10} {'mAP50-95':<10} {'Status':<10}")
    print("-" * 80)

    for r in results:
        status = "OK" if not r.error else "ERROR"
        mAP50_str = f"{r.mAP50:.4f}" if r.mAP50 else "N/A"
        mAP50_95_str = f"{r.mAP50_95:.4f}" if r.mAP50_95 else "N/A"
        print(f"{r.ratio:<8.1f} {r.synthetic_count:<8} {r.total_time_seconds:<10.1f} {mAP50_str:<10} {mAP50_95_str:<10} {status:<10}")

    print("=" * 80)

    # Recommendation
    successful = [r for r in results if not r.error and r.mAP50]
    if successful:
        best_mAP = max(successful, key=lambda x: x.mAP50)
        fastest = min(successful, key=lambda x: x.total_time_seconds)

        print(f"\nRecommendations:")
        print(f"  Best mAP50: ratio={best_mAP.ratio} (mAP50={best_mAP.mAP50:.4f})")
        print(f"  Fastest: ratio={fastest.ratio} (time={fastest.total_time_seconds:.1f}s)")

        # Find optimal balance (80% of best mAP with minimum time)
        threshold = best_mAP.mAP50 * 0.95
        acceptable = [r for r in successful if r.mAP50 >= threshold]
        if acceptable:
            optimal = min(acceptable, key=lambda x: x.total_time_seconds)
            print(f"  Optimal (95%+ mAP with min time): ratio={optimal.ratio}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark different synthetic_ratio values"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to data.yaml",
    )
    parser.add_argument(
        "--backgrounds",
        type=Path,
        required=True,
        help="Path to backgrounds directory",
    )
    parser.add_argument(
        "--annotated",
        type=Path,
        required=True,
        help="Path to annotated directory",
    )
    parser.add_argument(
        "--ratios",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 1.5, 2.0],
        help="List of ratios to benchmark (default: 0.5 1.0 1.5 2.0)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs (default: 20)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/ratio_benchmark.json"),
        help="Output JSON file path",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=None,
        help="Directory for trained models",
    )

    args = parser.parse_args()

    # Validate paths
    if not args.dataset.exists():
        print(f"Error: Dataset not found: {args.dataset}")
        sys.exit(1)

    if not args.backgrounds.exists():
        print(f"Error: Backgrounds directory not found: {args.backgrounds}")
        sys.exit(1)

    if not args.annotated.exists():
        print(f"Error: Annotated directory not found: {args.annotated}")
        sys.exit(1)

    print("Synthetic Ratio Benchmark")
    print(f"  Dataset: {args.dataset}")
    print(f"  Backgrounds: {args.backgrounds}")
    print(f"  Annotated: {args.annotated}")
    print(f"  Ratios: {args.ratios}")
    print(f"  Epochs: {args.epochs}")

    # Run benchmarks
    results = run_all_benchmarks(
        dataset_yaml=args.dataset,
        backgrounds_dir=args.backgrounds,
        annotated_dir=args.annotated,
        ratios=args.ratios,
        epochs=args.epochs,
        output_dir=args.model_output,
    )

    # Print comparison table
    print_comparison_table(results)

    # Save results
    save_results(results, args.output)


if __name__ == "__main__":
    main()
