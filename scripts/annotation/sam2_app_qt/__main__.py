"""
SAM2 Interactive Annotation Application - Entry Point.

Usage:
    python -m sam2_app_qt --input-dir images/ --output-dir labels/ --class-id 0
"""

import argparse
import sys
from pathlib import Path

from PyQt6.QtWidgets import QApplication

from sam2_app_qt.main_window import SAM2AnnotationWindow


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="SAM2 Interactive Annotation Tool (PyQt6)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python -m sam2_app_qt \\
      --input-dir datasets/raw_captures/object \\
      --output-dir datasets/annotated/object \\
      --class-id 0

  # With custom model
  python -m sam2_app_qt \\
      --input-dir images/ \\
      --output-dir labels/ \\
      --model models/sam2_b.pt \\
      --device cuda

Controls:
  Left click:  Add foreground point (include in mask)
  Right click: Add background point (exclude from mask)
  Ctrl+Z:      Undo last point
  Escape:      Reset all points
  Enter:       Accept and save annotation
  Arrow keys:  Navigate between images
  Space:       Toggle mask overlay
        """,
    )

    parser.add_argument(
        "--input-dir", "-i",
        required=True,
        help="Directory containing images to annotate",
    )
    parser.add_argument(
        "--output-dir", "-o",
        required=True,
        help="Directory for output YOLO label files",
    )
    parser.add_argument(
        "--class-id", "-c",
        type=int,
        default=0,
        help="YOLO class ID (default: 0)",
    )
    parser.add_argument(
        "--model", "-m",
        default="sam2_b.pt",
        help="Path to SAM2 model checkpoint (default: sam2_b.pt)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run model on (default: cuda)",
    )

    args = parser.parse_args()

    # Check input directory
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return 1

    # Check model file
    model_path = Path(args.model)
    if not model_path.exists():
        # Try relative to project root (go up 4 levels from __main__.py)
        # sam2_app_qt/__main__.py → sam2_app_qt/ → annotation/ → scripts/ → project_root
        project_root = Path(__file__).parent.parent.parent.parent
        model_name = Path(args.model).name
        model_path = project_root / "models" / model_name
        if not model_path.exists():
            print(f"Error: Model file not found: {args.model}")
            print(f"Tried: {model_path}")
            return 1

    # Create application
    app = QApplication(sys.argv)

    # Create and show window
    window = SAM2AnnotationWindow(
        input_dir=str(input_dir),
        output_dir=args.output_dir,
        class_id=args.class_id,
        model_path=str(model_path),
        device=args.device,
    )
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
