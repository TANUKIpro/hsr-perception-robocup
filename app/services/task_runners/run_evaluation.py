#!/usr/bin/env python3
"""
Evaluation Task Runner

Subprocess wrapper for running model evaluation.
Updates task status file for progress tracking.
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts" / "evaluation"))

from app.services.task_manager import update_task_status


def main():
    parser = argparse.ArgumentParser(description="Run model evaluation")
    parser.add_argument("--task-id", required=True, help="Task ID for status updates")
    parser.add_argument("--tasks-dir", help="Tasks directory for status files")
    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument("--dataset", required=True, help="Path to dataset YAML")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    args = parser.parse_args()

    task_id = args.task_id
    tasks_dir = args.tasks_dir

    try:
        update_task_status(
            task_id,
            progress=0.05,
            current_step="Loading evaluation modules...",
            tasks_dir=tasks_dir
        )

        # Import evaluation modules
        from evaluate_model import ModelEvaluator, EvaluationReport

        update_task_status(
            task_id,
            progress=0.1,
            current_step="Loading model...",
            tasks_dir=tasks_dir
        )

        # Create evaluator
        evaluator = ModelEvaluator(args.model)

        update_task_status(
            task_id,
            progress=0.2,
            current_step="Running evaluation on validation set...",
            tasks_dir=tasks_dir
        )

        # Run evaluation
        report = evaluator.evaluate(
            dataset_yaml=args.dataset,
            conf_threshold=args.conf,
        )

        update_task_status(
            task_id,
            progress=0.7,
            current_step="Measuring inference time...",
            tasks_dir=tasks_dir
        )

        # Inference time is already measured in evaluate()
        # Just update progress

        update_task_status(
            task_id,
            progress=0.9,
            current_step="Generating report...",
            tasks_dir=tasks_dir
        )

        # Check if requirements are met
        meets_requirements, issues = report.meets_requirements()

        # Save report to JSON
        model_dir = Path(args.model).parent.parent  # weights/best.pt -> model_dir
        report_path = model_dir / "evaluation_report.json"
        with open(report_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)

        update_task_status(
            task_id,
            progress=1.0,
            current_step="Completed",
            status="completed",
            result_path=str(report_path),
            extra_data={
                "overall_map50": report.overall_map50,
                "overall_map50_95": report.overall_map50_95,
                "overall_precision": report.overall_precision,
                "overall_recall": report.overall_recall,
                "inference_time_ms": report.inference_time_ms,
                "inference_time_std": report.inference_time_std,
                "num_test_images": report.num_test_images,
                "meets_requirements": meets_requirements,
                "issues": issues,
                "per_class_metrics": {
                    name: {
                        "ap50": m.ap50,
                        "ap50_95": m.ap50_95,
                        "precision": m.precision,
                        "recall": m.recall,
                    }
                    for name, m in report.per_class_metrics.items()
                }
            },
            tasks_dir=tasks_dir
        )

        # Print summary
        print(report.summary())

        if meets_requirements:
            print("\n[PASS] Model meets competition requirements!")
        else:
            print("\n[FAIL] Model does not meet competition requirements:")
            for issue in issues:
                print(f"  - {issue}")

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
