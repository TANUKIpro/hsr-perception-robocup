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
        if not self.config.enabled:
            return SyntheticGenerationResult(images_added=0)

        # Validate directories
        if not self.config.backgrounds_dir or not self.config.annotated_dir:
            if self.verbose:
                print(
                    f"{Fore.YELLOW}Dynamic synthetic: backgrounds_dir or annotated_dir "
                    f"not specified{Style.RESET_ALL}"
                )
            return SyntheticGenerationResult(images_added=0)

        if (
            not self.config.backgrounds_dir.exists()
            or not self.config.annotated_dir.exists()
        ):
            if self.verbose:
                print(
                    f"{Fore.YELLOW}Dynamic synthetic: directories do not exist"
                    f"{Style.RESET_ALL}"
                )
            return SyntheticGenerationResult(images_added=0)

        if self.verbose:
            print(f"{Fore.CYAN}Generating dynamic synthetic images...{Style.RESET_ALL}")

        # Count training images
        train_images_dir = dataset_path / "images" / "train"
        real_count = (
            len(list(train_images_dir.glob("*"))) if train_images_dir.exists() else 0
        )

        if real_count == 0:
            if self.verbose:
                print(
                    f"{Fore.YELLOW}No training images found for synthetic generation"
                    f"{Style.RESET_ALL}"
                )
            return SyntheticGenerationResult(images_added=0)

        # Generate synthetic images
        output_dir = dataset_path / "synthetic_dynamic"
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Load class names from dataset YAML
            class_names = self._load_class_names(dataset_path)
            if not class_names:
                return SyntheticGenerationResult(
                    images_added=0,
                    error="No class names found in dataset YAML",
                )

            # Create CopyPaste config and augmentor
            cp_config = self._create_copy_paste_config()
            augmentor = self._create_augmentor(cp_config)

            stats = augmentor.generate_batch(
                backgrounds_dir=self.config.backgrounds_dir,
                annotated_dir=self.config.annotated_dir,
                output_dir=output_dir,
                real_image_count=real_count,
                class_names=class_names,
            )

            if "error" in stats:
                if self.verbose:
                    print(
                        f"{Fore.RED}Synthetic generation error: {stats['error']}"
                        f"{Style.RESET_ALL}"
                    )
                return SyntheticGenerationResult(
                    images_added=0,
                    error=stats["error"],
                )

            # Merge generated images into training set
            added = self._merge_to_train(output_dir, dataset_path)
            if self.verbose:
                print(
                    f"{Fore.GREEN}Added {added} dynamic synthetic images to training set"
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
        train_images_dir = dataset_path / "images" / "train"
        train_labels_dir = dataset_path / "labels" / "train"

        train_images_dir.mkdir(parents=True, exist_ok=True)
        train_labels_dir.mkdir(parents=True, exist_ok=True)

        images_dir = synthetic_dir / "images"
        labels_dir = synthetic_dir / "labels"

        if not images_dir.exists() or not labels_dir.exists():
            return 0

        added = 0
        images = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))

        for image_path in images:
            label_path = labels_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue

            # Unique filename for saving
            new_name = f"dynamic_synth_{image_path.stem}"
            image_dest = train_images_dir / f"{new_name}{image_path.suffix}"
            label_dest = train_labels_dir / f"{new_name}.txt"

            if not image_dest.exists():
                shutil.copy2(image_path, image_dest)
                shutil.copy2(label_path, label_dest)
                added += 1

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
