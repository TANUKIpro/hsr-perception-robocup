"""
Dynamic synthetic data generation for training.

Manages Copy-Paste augmentation for generating synthetic training images.
"""
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
from colorama import Fore, Style


# Keys that should NOT be passed to YOLO's train() method
SYNTHETIC_CONFIG_KEYS = {
    "dynamic_synthetic_enabled",
    "backgrounds_dir",
    "annotated_dir",
    "synthetic_ratio",
    "synthetic_scale_range",
    "synthetic_rotation_range",
    "synthetic_white_balance",
    "synthetic_white_balance_strength",
    "synthetic_edge_blur",
    "synthetic_max_objects",
    "synthetic_num_workers",
}


@dataclass
class SyntheticConfig:
    """Configuration for synthetic data generation."""

    enabled: bool = True
    backgrounds_dir: Optional[Path] = None
    annotated_dir: Optional[Path] = None
    ratio: float = 2.0
    scale_range: Tuple[float, float] = (0.5, 1.5)
    rotation_range: Tuple[float, float] = (-15.0, 15.0)
    white_balance: bool = True
    white_balance_strength: float = 0.7
    edge_blur: float = 2.0
    max_objects: int = 3
    num_workers: int = 1  # 1 = sequential, >1 = parallel processing

    @classmethod
    def from_dict(cls, config: Dict) -> "SyntheticConfig":
        """
        Create from dictionary (compatibility with existing config).

        Args:
            config: Configuration dictionary (e.g., COMPETITION_CONFIG)

        Returns:
            SyntheticConfig instance
        """
        backgrounds_dir = config.get("backgrounds_dir")
        annotated_dir = config.get("annotated_dir")

        # Validate and sanitize num_workers
        num_workers_raw = config.get("synthetic_num_workers", 1)
        try:
            num_workers = int(num_workers_raw)
            if num_workers < 1:
                num_workers = 1
        except (TypeError, ValueError):
            num_workers = 1

        return cls(
            enabled=config.get("dynamic_synthetic_enabled", False),
            backgrounds_dir=Path(backgrounds_dir) if backgrounds_dir else None,
            annotated_dir=Path(annotated_dir) if annotated_dir else None,
            ratio=config.get("synthetic_ratio", 2.0),
            scale_range=config.get("synthetic_scale_range", (0.5, 1.5)),
            rotation_range=config.get("synthetic_rotation_range", (-15.0, 15.0)),
            white_balance=config.get("synthetic_white_balance", True),
            white_balance_strength=config.get("synthetic_white_balance_strength", 0.7),
            edge_blur=config.get("synthetic_edge_blur", 2.0),
            max_objects=config.get("synthetic_max_objects", 3),
            num_workers=num_workers,
        )


@dataclass
class SyntheticGenerationResult:
    """Result of synthetic data generation."""

    images_added: int = 0
    output_dir: Optional[Path] = None
    error: Optional[str] = None


class SyntheticDataManager:
    """Manages dynamic Copy-Paste synthetic image generation."""

    def __init__(self, config: SyntheticConfig, verbose: bool = True):
        """
        Initialize manager.

        Args:
            config: Synthetic data configuration
            verbose: Whether to print progress messages
        """
        self.config = config
        self.verbose = verbose

    def generate(self, dataset_path: Path) -> SyntheticGenerationResult:
        """
        Generate synthetic images and merge to training set.

        Args:
            dataset_path: Path to dataset directory

        Returns:
            SyntheticGenerationResult with count of images added
        """
        if self.verbose:
            print(f"\n{Fore.CYAN}[DEBUG] SyntheticDataManager.generate() called{Style.RESET_ALL}")
            print(f"  dataset_path: {dataset_path}")
            print(f"  dataset_path (absolute): {dataset_path.resolve()}")

        if not self.config.enabled:
            if self.verbose:
                print(f"{Fore.YELLOW}[DEBUG] Dynamic synthetic DISABLED (config.enabled=False){Style.RESET_ALL}")
            return SyntheticGenerationResult(images_added=0)

        if self.verbose:
            print(f"{Fore.GREEN}[DEBUG] Dynamic synthetic ENABLED{Style.RESET_ALL}")

        # Validate directories
        if not self.config.backgrounds_dir or not self.config.annotated_dir:
            if self.verbose:
                print(
                    f"{Fore.YELLOW}[DEBUG] Dynamic synthetic: backgrounds_dir or annotated_dir "
                    f"not specified{Style.RESET_ALL}"
                )
                print(f"  backgrounds_dir: {self.config.backgrounds_dir}")
                print(f"  annotated_dir: {self.config.annotated_dir}")
            return SyntheticGenerationResult(images_added=0)

        if self.verbose:
            print(f"{Fore.GREEN}[DEBUG] Required directories specified{Style.RESET_ALL}")
            print(f"  backgrounds_dir: {self.config.backgrounds_dir}")
            print(f"  backgrounds_dir exists: {self.config.backgrounds_dir.exists()}")
            print(f"  annotated_dir: {self.config.annotated_dir}")
            print(f"  annotated_dir exists: {self.config.annotated_dir.exists()}")

        if (
            not self.config.backgrounds_dir.exists()
            or not self.config.annotated_dir.exists()
        ):
            if self.verbose:
                print(
                    f"{Fore.YELLOW}[DEBUG] Dynamic synthetic: directories do not exist"
                    f"{Style.RESET_ALL}"
                )
            return SyntheticGenerationResult(images_added=0)

        if self.verbose:
            print(f"{Fore.GREEN}[DEBUG] All required directories exist{Style.RESET_ALL}")

        if self.verbose:
            print(f"{Fore.CYAN}Generating dynamic synthetic images...{Style.RESET_ALL}")

        # Count training images
        train_images_dir = dataset_path / "images" / "train"
        if self.verbose:
            print(f"[DEBUG] Looking for training images in: {train_images_dir}")
            print(f"  train_images_dir exists: {train_images_dir.exists()}")

        real_count = (
            len(list(train_images_dir.glob("*"))) if train_images_dir.exists() else 0
        )

        if self.verbose:
            print(f"[DEBUG] Found {real_count} training images")

        if real_count == 0:
            if self.verbose:
                print(
                    f"{Fore.YELLOW}No training images found for synthetic generation"
                    f"{Style.RESET_ALL}"
                )
            return SyntheticGenerationResult(images_added=0)

        # Generate synthetic images
        output_dir = dataset_path / "synthetic_dynamic"
        if self.verbose:
            print(f"\n{Fore.CYAN}[DEBUG] Creating output directory{Style.RESET_ALL}")
            print(f"  output_dir: {output_dir}")
            print(f"  output_dir (absolute): {output_dir.resolve()}")
            print(f"  output_dir exists (before mkdir): {output_dir.exists()}")

        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            if self.verbose:
                print(f"{Fore.GREEN}[DEBUG] Successfully created output directory{Style.RESET_ALL}")
                print(f"  output_dir exists (after mkdir): {output_dir.exists()}")
                print(f"  output_dir is_dir: {output_dir.is_dir()}")
        except Exception as e:
            if self.verbose:
                print(f"{Fore.RED}[DEBUG] Failed to create output directory: {e}{Style.RESET_ALL}")
            return SyntheticGenerationResult(images_added=0, error=str(e))

        try:
            # Load class names from dataset YAML
            if self.verbose:
                print(f"\n{Fore.CYAN}[DEBUG] Loading class names from dataset YAML{Style.RESET_ALL}")

            class_names = self._load_class_names(dataset_path)
            if not class_names:
                if self.verbose:
                    print(f"{Fore.YELLOW}[DEBUG] No class names found{Style.RESET_ALL}")
                return SyntheticGenerationResult(
                    images_added=0,
                    error="No class names found in dataset YAML",
                )

            if self.verbose:
                print(f"{Fore.GREEN}[DEBUG] Loaded {len(class_names)} class names: {class_names}{Style.RESET_ALL}")

            # Create CopyPaste config and augmentor
            if self.verbose:
                print(f"\n{Fore.CYAN}[DEBUG] Creating CopyPaste augmentor{Style.RESET_ALL}")

            cp_config = self._create_copy_paste_config()
            augmentor = self._create_augmentor(cp_config)

            if self.verbose:
                print(f"\n{Fore.CYAN}[DEBUG] Starting batch generation{Style.RESET_ALL}")
                print(f"  backgrounds_dir: {self.config.backgrounds_dir}")
                print(f"  annotated_dir: {self.config.annotated_dir}")
                print(f"  output_dir: {output_dir}")
                print(f"  real_image_count: {real_count}")
                print(f"  synthetic_ratio: {self.config.ratio}")
                print(f"  num_synthetic_to_generate: {int(real_count * self.config.ratio)}")
                print(f"  num_workers: {self.config.num_workers}")

            stats = augmentor.generate_batch(
                backgrounds_dir=self.config.backgrounds_dir,
                annotated_dir=self.config.annotated_dir,
                output_dir=output_dir,
                real_image_count=real_count,
                class_names=class_names,
                num_workers=self.config.num_workers,
            )

            if self.verbose:
                print(f"\n{Fore.CYAN}[DEBUG] Batch generation complete{Style.RESET_ALL}")
                print(f"  stats: {stats}")

            if "error" in stats:
                if self.verbose:
                    print(
                        f"{Fore.RED}[DEBUG] Synthetic generation error: {stats['error']}"
                        f"{Style.RESET_ALL}"
                    )
                return SyntheticGenerationResult(
                    images_added=0,
                    error=stats["error"],
                )

            if self.verbose:
                print(f"\n{Fore.CYAN}[DEBUG] Merging synthetic images to training set{Style.RESET_ALL}")

            # Merge generated images into training set
            added = self._merge_to_train(output_dir, dataset_path)
            if self.verbose:
                print(
                    f"{Fore.GREEN}[DEBUG] Merge complete. Added {added} dynamic synthetic images to training set"
                    f"{Style.RESET_ALL}"
                )

            return SyntheticGenerationResult(
                images_added=added,
                output_dir=output_dir,
            )

        except (IOError, OSError) as io_error:
            if self.verbose:
                print(
                    f"{Fore.RED}I/O error during synthetic generation: {io_error}"
                    f"{Style.RESET_ALL}"
                )
            return SyntheticGenerationResult(images_added=0, error=str(io_error))
        except ValueError as val_error:
            if self.verbose:
                print(
                    f"{Fore.RED}Invalid value during synthetic generation: {val_error}"
                    f"{Style.RESET_ALL}"
                )
            return SyntheticGenerationResult(images_added=0, error=str(val_error))
        except RuntimeError as runtime_error:
            if self.verbose:
                print(
                    f"{Fore.RED}Runtime error during synthetic generation: {runtime_error}"
                    f"{Style.RESET_ALL}"
                )
            return SyntheticGenerationResult(images_added=0, error=str(runtime_error))
        except Exception as e:
            if self.verbose:
                print(
                    f"{Fore.RED}Unexpected error during synthetic generation: {e}"
                    f"{Style.RESET_ALL}"
                )
                import traceback

                traceback.print_exc()
            return SyntheticGenerationResult(images_added=0, error=str(e))

    def _load_class_names(self, dataset_path: Path) -> List[str]:
        """
        Load class names from dataset YAML.

        Args:
            dataset_path: Path to dataset directory

        Returns:
            List of class names, empty list if not found
        """
        yaml_path = dataset_path / "data.yaml"
        if not yaml_path.exists():
            if self.verbose:
                print(
                    f"{Fore.YELLOW}Dataset YAML not found: {yaml_path}{Style.RESET_ALL}"
                )
            return []

        with open(yaml_path, "r") as f:
            data_config = yaml.safe_load(f)

        # Support both list and dict formats for names
        names_data = data_config.get("names", [])
        if isinstance(names_data, dict):
            class_names = list(names_data.values())
        else:
            class_names = list(names_data)
        if not class_names and self.verbose:
            print(
                f"{Fore.YELLOW}No class names found in dataset YAML{Style.RESET_ALL}"
            )

        return class_names

    def _create_copy_paste_config(self):
        """
        Create CopyPasteConfig from synthetic config.

        Returns:
            CopyPasteConfig instance
        """
        # Import here to avoid circular imports
        _scripts_dir = Path(__file__).parent.parent
        if str(_scripts_dir) not in sys.path:
            sys.path.insert(0, str(_scripts_dir))

        from augmentation.copy_paste_augmentor import CopyPasteConfig

        return CopyPasteConfig(
            synthetic_to_real_ratio=self.config.ratio,
            scale_range=self.config.scale_range,
            rotation_range=self.config.rotation_range,
            enable_white_balance=self.config.white_balance,
            white_balance_strength=self.config.white_balance_strength,
            edge_blur_sigma=self.config.edge_blur,
            max_objects_per_image=self.config.max_objects,
            seed=int(time.time()),
        )

    def _create_augmentor(self, cp_config):
        """
        Create CopyPasteAugmentor.

        Args:
            cp_config: CopyPasteConfig instance

        Returns:
            CopyPasteAugmentor instance
        """
        from augmentation.copy_paste_augmentor import CopyPasteAugmentor

        return CopyPasteAugmentor(cp_config)

    def _merge_to_train(self, synthetic_dir: Path, dataset_path: Path) -> int:
        """
        Merge generated images into training set.

        Args:
            synthetic_dir: Directory containing generated synthetic images
            dataset_path: Path to dataset directory

        Returns:
            Number of images added
        """
        if self.verbose:
            print(f"[DEBUG] _merge_to_train called")
            print(f"  synthetic_dir: {synthetic_dir}")
            print(f"  dataset_path: {dataset_path}")

        train_images_dir = dataset_path / "images" / "train"
        train_labels_dir = dataset_path / "labels" / "train"

        if self.verbose:
            print(f"  train_images_dir: {train_images_dir}")
            print(f"  train_labels_dir: {train_labels_dir}")

        train_images_dir.mkdir(parents=True, exist_ok=True)
        train_labels_dir.mkdir(parents=True, exist_ok=True)

        images_dir = synthetic_dir / "images"
        labels_dir = synthetic_dir / "labels"

        if self.verbose:
            print(f"  images_dir: {images_dir}")
            print(f"  images_dir exists: {images_dir.exists()}")
            print(f"  labels_dir: {labels_dir}")
            print(f"  labels_dir exists: {labels_dir.exists()}")

        if not images_dir.exists() or not labels_dir.exists():
            if self.verbose:
                print(f"{Fore.YELLOW}[DEBUG] Synthetic subdirectories do not exist, returning 0{Style.RESET_ALL}")
            return 0

        added = 0
        images = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))

        if self.verbose:
            print(f"  Found {len(images)} synthetic images to merge")

        for image_path in images:
            label_path = labels_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                if self.verbose:
                    print(f"  Skipping {image_path.name} (no matching label)")
                continue

            # Unique filename for saving
            new_name = f"dynamic_synth_{image_path.stem}"
            image_dest = train_images_dir / f"{new_name}{image_path.suffix}"
            label_dest = train_labels_dir / f"{new_name}.txt"

            if not image_dest.exists():
                shutil.copy2(image_path, image_dest)
                shutil.copy2(label_path, label_dest)
                added += 1
                if self.verbose and added <= 5:  # Only log first 5 to avoid spam
                    print(f"  Copied: {image_path.name} -> {image_dest.name}")

        if self.verbose:
            print(f"[DEBUG] Merge complete: {added} images added")

        return added


def filter_synthetic_keys(config: Dict) -> Dict:
    """
    Remove synthetic-specific keys from config for YOLO.

    Args:
        config: Configuration dictionary

    Returns:
        Configuration with synthetic keys removed
    """
    return {k: v for k, v in config.items() if k not in SYNTHETIC_CONFIG_KEYS}
