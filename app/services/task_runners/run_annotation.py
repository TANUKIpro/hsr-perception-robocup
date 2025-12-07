#!/usr/bin/env python3
"""
Annotation Task Runner

Subprocess wrapper for running the annotation pipeline.
Updates task status file for progress tracking.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts" / "annotation"))

from app.services.task_manager import update_task_status


def main():
    parser = argparse.ArgumentParser(description="Run annotation pipeline")
    parser.add_argument("--task-id", required=True, help="Task ID for status updates")
    parser.add_argument("--tasks-dir", help="Tasks directory for status files")
    parser.add_argument("--method", default="background", choices=["background", "sam2"])
    parser.add_argument("--input-dir", required=True, help="Input directory with raw captures")
    parser.add_argument("--output-dir", required=True, help="Output directory for dataset")
    parser.add_argument("--class-config", required=True, help="Path to class config JSON")
    parser.add_argument("--background", help="Background image path (for background method)")
    parser.add_argument("--split", type=float, default=0.85, help="Train/val split ratio")
    parser.add_argument("--min-area", type=int, default=500, help="Minimum contour area")
    args = parser.parse_args()

    task_id = args.task_id
    tasks_dir = args.tasks_dir

    try:
        update_task_status(
            task_id,
            progress=0.05,
            current_step="Loading annotation modules...",
            tasks_dir=tasks_dir
        )

        # Import annotation modules
        from auto_annotate import AutoAnnotator
        from background_subtraction import AnnotatorConfig

        update_task_status(
            task_id,
            progress=0.1,
            current_step="Initializing annotator...",
            tasks_dir=tasks_dir
        )

        # Create annotator config
        annotator_config = AnnotatorConfig(min_contour_area=args.min_area)

        # Create annotator
        annotator = AutoAnnotator(
            method=args.method,
            background_path=args.background,
            annotator_config=annotator_config,
        )

        update_task_status(
            task_id,
            progress=0.15,
            current_step="Running annotation pipeline...",
            tasks_dir=tasks_dir
        )

        # Run annotation
        # Note: The actual annotator doesn't have progress callbacks,
        # so we update progress after completion
        report = annotator.run(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            class_config_path=args.class_config,
            train_val_split=args.split,
            verify=True,
            update_config=True,
        )

        # Calculate result path
        result_path = str(Path(args.output_dir) / "data.yaml")

        update_task_status(
            task_id,
            progress=1.0,
            current_step="Completed",
            status="completed",
            result_path=result_path,
            extra_data={
                "total_classes": report.total_classes,
                "total_images": report.total_images,
                "successful": report.successful,
                "failed": report.failed,
                "success_rate": report.success_rate,
                "train_count": report.train_count,
                "val_count": report.val_count,
            },
            tasks_dir=tasks_dir
        )

        print(report.summary())
        print(f"\nAnnotation completed successfully!")
        print(f"Dataset saved to: {args.output_dir}")
        print(f"Success rate: {report.success_rate:.1f}%")

    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {str(e)}"
        traceback_str = traceback.format_exc()

        update_task_status(
            task_id,
            progress=0.0,
            current_step="Failed",
            status="failed",
            error_message=error_msg,
            extra_data={"traceback": traceback_str},
            tasks_dir=tasks_dir
        )

        print(f"Error: {error_msg}", file=sys.stderr)
        print(traceback_str, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
