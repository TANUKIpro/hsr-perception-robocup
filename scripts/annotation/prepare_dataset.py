#!/usr/bin/env python3
"""
YOLO Dataset Preparation Utility

Combines raw capture images and annotation labels into a YOLO training-ready dataset.
Creates proper directory structure with train/val split and data.yaml.

Usage:
    python scripts/annotation/prepare_dataset.py \
        --profile prof_2 \
        --objects bottle01 bottle02 postbox \
        --output-name combined_dataset \
        --val-ratio 0.2
"""

import argparse
import shutil
import random
from pathlib import Path
from typing import List, Dict, Tuple
import yaml


def find_matching_pairs(
    raw_captures_dir: Path,
    annotated_dir: Path,
    object_name: str,
) -> List[Tuple[Path, Path, str]]:
    """
    Find matching image-label pairs for an object.

    Returns:
        List of (image_path, label_path, object_name) tuples
    """
    pairs = []

    # Image directory
    image_dir = raw_captures_dir / object_name
    if not image_dir.exists():
        print(f"  Warning: Image directory not found: {image_dir}")
        return pairs

    # Label directory
    label_dir = annotated_dir / object_name / "labels"
    if not label_dir.exists():
        print(f"  Warning: Label directory not found: {label_dir}")
        return pairs

    # Get all images
    images = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))

    for image_path in images:
        # Find corresponding label
        label_name = image_path.stem + ".txt"
        label_path = label_dir / label_name

        if label_path.exists():
            pairs.append((image_path, label_path, object_name))
        else:
            print(f"  Warning: No label for {image_path.name}")

    return pairs


def create_class_mapping(object_names: List[str]) -> Dict[str, int]:
    """
    Create class ID mapping from object names.

    Returns:
        Dict mapping object_name -> class_id
    """
    return {name: idx for idx, name in enumerate(sorted(object_names))}


def remap_labels(
    label_path: Path,
    original_class_id: int,
    new_class_id: int,
) -> List[str]:
    """
    Read label file and remap class IDs.

    Returns:
        List of remapped label lines
    """
    lines = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                # Replace class ID with new one
                parts[0] = str(new_class_id)
                lines.append(' '.join(parts))
    return lines


def prepare_dataset(
    profile_dir: Path,
    object_names: List[str],
    output_name: str,
    val_ratio: float = 0.2,
    copy_files: bool = True,
    seed: int = 42,
) -> Path:
    """
    Prepare YOLO training dataset from raw captures and annotations.

    Args:
        profile_dir: Path to profile directory (e.g., profiles/prof_2)
        object_names: List of object names to include
        output_name: Name for the output dataset directory
        val_ratio: Ratio of validation data (0.0-1.0)
        copy_files: If True, copy files. If False, create symlinks.
        seed: Random seed for train/val split

    Returns:
        Path to the created dataset directory
    """
    random.seed(seed)

    # Setup directories
    datasets_dir = profile_dir / "datasets"
    raw_captures_dir = datasets_dir / "raw_captures"
    annotated_dir = datasets_dir / "annotated"

    # Output directory
    output_dir = annotated_dir / output_name

    if output_dir.exists():
        print(f"Removing existing output directory: {output_dir}")
        shutil.rmtree(output_dir)

    # Create output structure
    (output_dir / "images" / "train").mkdir(parents=True)
    (output_dir / "images" / "val").mkdir(parents=True)
    (output_dir / "labels" / "train").mkdir(parents=True)
    (output_dir / "labels" / "val").mkdir(parents=True)

    # Create class mapping
    class_mapping = create_class_mapping(object_names)
    print(f"\nClass mapping:")
    for name, idx in class_mapping.items():
        print(f"  {idx}: {name}")

    # Collect all image-label pairs
    print(f"\nCollecting image-label pairs...")
    all_pairs = []

    for obj_name in object_names:
        print(f"  Processing {obj_name}...")
        pairs = find_matching_pairs(raw_captures_dir, annotated_dir, obj_name)
        print(f"    Found {len(pairs)} pairs")
        all_pairs.extend(pairs)

    if not all_pairs:
        raise ValueError("No image-label pairs found!")

    print(f"\nTotal pairs: {len(all_pairs)}")

    # Shuffle and split
    random.shuffle(all_pairs)
    val_count = int(len(all_pairs) * val_ratio)
    train_count = len(all_pairs) - val_count

    train_pairs = all_pairs[:train_count]
    val_pairs = all_pairs[train_count:]

    print(f"Train: {len(train_pairs)}, Val: {len(val_pairs)}")

    # Copy/link files
    print(f"\nCopying files...")

    def process_pairs(pairs: List[Tuple[Path, Path, str]], split: str):
        for image_path, label_path, obj_name in pairs:
            # Generate unique filename to avoid conflicts
            new_name = f"{obj_name}_{image_path.stem}"

            # Image
            image_dest = output_dir / "images" / split / f"{new_name}{image_path.suffix}"
            if copy_files:
                shutil.copy2(image_path, image_dest)
            else:
                image_dest.symlink_to(image_path.resolve())

            # Label (remap class ID)
            new_class_id = class_mapping[obj_name]
            remapped_lines = remap_labels(label_path, 0, new_class_id)

            label_dest = output_dir / "labels" / split / f"{new_name}.txt"
            with open(label_dest, 'w') as f:
                f.write('\n'.join(remapped_lines))

    process_pairs(train_pairs, "train")
    process_pairs(val_pairs, "val")

    # Generate data.yaml
    print(f"\nGenerating data.yaml...")

    data_yaml = {
        'path': str(output_dir.resolve()),
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(object_names),
        'names': sorted(object_names),
    }

    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False, allow_unicode=True)

    print(f"\nDataset created successfully!")
    print(f"  Output: {output_dir}")
    print(f"  data.yaml: {yaml_path}")
    print(f"  Train images: {len(train_pairs)}")
    print(f"  Val images: {len(val_pairs)}")
    print(f"  Classes: {sorted(object_names)}")

    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Prepare YOLO training dataset from raw captures and annotations"
    )
    parser.add_argument(
        "--profile",
        required=True,
        help="Profile name (e.g., prof_2)"
    )
    parser.add_argument(
        "--objects",
        nargs="+",
        required=True,
        help="Object names to include (e.g., bottle01 bottle02 postbox)"
    )
    parser.add_argument(
        "--output-name",
        default="combined_dataset",
        help="Output dataset name (default: combined_dataset)"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation data ratio (default: 0.2)"
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="Use symlinks instead of copying files"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val split (default: 42)"
    )
    parser.add_argument(
        "--profiles-dir",
        default=None,
        help="Path to profiles directory (default: auto-detect)"
    )

    args = parser.parse_args()

    # Find profiles directory
    if args.profiles_dir:
        profiles_dir = Path(args.profiles_dir)
    else:
        # Auto-detect from script location
        script_dir = Path(__file__).parent
        project_root = script_dir.parent.parent
        profiles_dir = project_root / "profiles"

    profile_dir = profiles_dir / args.profile

    if not profile_dir.exists():
        print(f"Error: Profile directory not found: {profile_dir}")
        return 1

    print(f"Profile: {profile_dir}")
    print(f"Objects: {args.objects}")
    print(f"Output: {args.output_name}")
    print(f"Val ratio: {args.val_ratio}")

    try:
        prepare_dataset(
            profile_dir=profile_dir,
            object_names=args.objects,
            output_name=args.output_name,
            val_ratio=args.val_ratio,
            copy_files=not args.symlink,
            seed=args.seed,
        )
        return 0
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
