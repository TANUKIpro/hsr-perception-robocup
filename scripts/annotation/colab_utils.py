#!/usr/bin/env python3
"""
Colab Utilities for SAM3 Tracker

Helper functions for Google Colab environment including:
- File upload/download utilities
- Environment detection
- Progress display helpers
"""

import os
import zipfile
from pathlib import Path
from typing import List, Optional


def is_colab() -> bool:
    """Check if running in Google Colab."""
    try:
        import google.colab  # noqa: F401

        return True
    except ImportError:
        return False


def upload_zip_and_extract(extract_to: str = "/content/images") -> str:
    """
    Upload a zip file and extract it to specified directory.

    Args:
        extract_to: Directory to extract files to

    Returns:
        Path to extracted directory

    Raises:
        RuntimeError: If not running in Google Colab
        ValueError: If no file uploaded
    """
    if not is_colab():
        raise RuntimeError("This function only works in Google Colab")

    from google.colab import files

    print("Please upload a zip file containing your images...")
    uploaded = files.upload()

    if not uploaded:
        raise ValueError("No file uploaded")

    zip_filename = list(uploaded.keys())[0]

    # Create extract directory
    os.makedirs(extract_to, exist_ok=True)

    # Extract
    with zipfile.ZipFile(zip_filename, "r") as zip_ref:
        zip_ref.extractall(extract_to)

    # Clean up zip file
    os.remove(zip_filename)

    print(f"Extracted to: {extract_to}")

    # List extracted files
    extracted_files = list(Path(extract_to).rglob("*"))
    image_files = [
        f
        for f in extracted_files
        if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
    ]
    print(f"Found {len(image_files)} image files")

    return extract_to


def download_results(output_dir: str, zip_name: str = "annotations.zip") -> None:
    """
    Download annotation results as a zip file.

    Args:
        output_dir: Directory containing annotation results
        zip_name: Name for the downloaded zip file

    Raises:
        RuntimeError: If not running in Google Colab
    """
    if not is_colab():
        raise RuntimeError("This function only works in Google Colab")

    from google.colab import files
    import shutil

    # Create zip file
    zip_path = f"/content/{zip_name}"
    shutil.make_archive(zip_path.replace(".zip", ""), "zip", output_dir)

    # Download
    files.download(zip_path)
    print(f"Downloaded: {zip_name}")


def mount_drive_simple() -> str:
    """
    Mount Google Drive with minimal prompts.

    Returns:
        Path to mounted drive (/content/drive/MyDrive)

    Raises:
        RuntimeError: If not running in Google Colab
    """
    if not is_colab():
        raise RuntimeError("This function only works in Google Colab")

    from google.colab import drive

    drive.mount("/content/drive")

    return "/content/drive/MyDrive"


def list_image_files(directory: str) -> List[str]:
    """
    List all image files in a directory.

    Args:
        directory: Path to directory

    Returns:
        List of image file paths (sorted)
    """
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    path = Path(directory)

    if not path.exists():
        return []

    images = []
    for ext in image_extensions:
        images.extend(path.glob(f"*{ext}"))
        images.extend(path.glob(f"*{ext.upper()}"))

    return sorted([str(p) for p in images])


def show_progress_bar(current: int, total: int, prefix: str = "Progress") -> None:
    """
    Display a simple progress bar in Colab.

    Args:
        current: Current progress count
        total: Total count
        prefix: Prefix text for progress bar
    """
    if not is_colab():
        # Fallback to simple print for non-Colab environments
        percent = int(100 * current / total) if total > 0 else 0
        print(f"{prefix}: {current}/{total} ({percent}%)")
        return

    from IPython.display import display, HTML, clear_output

    percent = int(100 * current / total) if total > 0 else 0

    clear_output(wait=True)
    display(
        HTML(
            f"""
        <div style="margin: 10px 0;">
            <b>{prefix}:</b> {current}/{total} ({percent}%)
            <div style="background: #ddd; border-radius: 5px; padding: 2px;">
                <div style="background: #4CAF50; height: 20px; border-radius: 5px;
                            width: {percent}%;"></div>
            </div>
        </div>
    """
        )
    )


def get_first_image(directory: str) -> Optional[str]:
    """
    Get the first image file in a directory.

    Args:
        directory: Path to directory

    Returns:
        Path to first image, or None if no images found
    """
    images = list_image_files(directory)
    return images[0] if images else None


def create_output_dir(base_dir: str = "/content/output") -> str:
    """
    Create output directory for annotations.

    Args:
        base_dir: Base directory path

    Returns:
        Path to created directory
    """
    os.makedirs(base_dir, exist_ok=True)
    return base_dir
