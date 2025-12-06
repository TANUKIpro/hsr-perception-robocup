"""
HSR Perception App Pages

Page modules for the Streamlit application.
Uses importlib to handle numbered file names (e.g., 4_Annotation.py).
"""

import importlib.util
from pathlib import Path


def _import_from_file(file_name: str, func_name: str):
    """Import a function from a numbered page file."""
    file_path = Path(__file__).parent / file_name
    if not file_path.exists():
        raise ImportError(f"Page file not found: {file_path}")

    spec = importlib.util.spec_from_file_location(file_name.replace(".py", ""), file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return getattr(module, func_name)


# Import page functions
show_annotation_page = _import_from_file("4_Annotation.py", "show_annotation_page")
show_training_page = _import_from_file("5_Training.py", "show_training_page")
show_evaluation_page = _import_from_file("6_Evaluation.py", "show_evaluation_page")


__all__ = [
    "show_annotation_page",
    "show_training_page",
    "show_evaluation_page",
]