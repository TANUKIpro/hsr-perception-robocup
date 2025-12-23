"""
Parallel Synthetic Image Generator

Generates synthetic training images in parallel using ProcessPoolExecutor.
Optimized for multi-core systems to speed up data augmentation during competition day.
"""

import os
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Import from other modules
import sys
_scripts_dir = Path(__file__).parent.parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

from annotation.annotation_utils import bbox_to_yolo
from .object_extractor import ExtractedObject, ObjectExtractor, ObjectReference
from .copy_paste_augmentor import CopyPasteConfig, CopyPasteAugmentor, PasteResult


@dataclass
class GenerationTask:
    """Task definition for parallel synthetic image generation."""
    task_id: int
    background_path: str
    object_ref_indices: List[int]  # Indices into shared object_refs list
    target_resolution: Tuple[int, int]  # (height, width)
    config_dict: Dict  # Serializable config
    seed: int


@dataclass
class GenerationResult:
    """Result from a single synthetic image generation task."""
    task_id: int
    success: bool
    image_path: Optional[str] = None
    label_path: Optional[str] = None
    objects_placed: int = 0
    class_counts: Dict[str, int] = field(default_factory=dict)
    error: Optional[str] = None


def worker_generate_single(
    task: GenerationTask,
    object_refs_data: List[Dict],  # Serialized ObjectReference data
    output_dir: str,
    class_names: List[str],
) -> GenerationResult:
    """
    Generate a single synthetic image in worker process.

    This function must be at top-level for pickle serialization.
    Each worker creates its own numpy RNG and augmentor instance.

    Args:
        task: GenerationTask containing all task parameters
        object_refs_data: List of serialized ObjectReference dictionaries
        output_dir: Output directory for generated images/labels
        class_names: List of class names

    Returns:
        GenerationResult with success status and statistics
    """
    try:
        # Deserialize object references
        object_refs = [
            ObjectReference(**ref_dict) for ref_dict in object_refs_data
        ]

        # Create worker-specific random state
        rng = np.random.RandomState(task.seed)

        # Reconstruct config from dict
        config = CopyPasteConfig(**task.config_dict)
        config.seed = task.seed  # Use task-specific seed

        # Create augmentor instance for this worker
        augmentor = CopyPasteAugmentor(config=config)
        augmentor.rng = rng  # Replace RNG with task-specific one

        # Load background image
        background = cv2.imread(task.background_path)
        if background is None:
            return GenerationResult(
                task_id=task.task_id,
                success=False,
                error=f"Failed to load background: {task.background_path}"
            )

        # Resize background to target resolution
        target_h, target_w = task.target_resolution
        background = cv2.resize(
            background, (target_w, target_h), interpolation=cv2.INTER_LINEAR
        )

        # Load selected objects on-demand
        loaded_objects = []
        for idx in task.object_ref_indices:
            if idx >= len(object_refs):
                continue

            ref = object_refs[idx]
            obj_tuple = _load_single_object_worker(
                ref=ref,
                alpha_blur_sigma=config.edge_blur_sigma,
                padding=5,
            )
            if obj_tuple is not None:
                loaded_objects.append(obj_tuple)

        if not loaded_objects:
            return GenerationResult(
                task_id=task.task_id,
                success=False,
                error="No objects could be loaded"
            )

        # Generate synthetic image
        result = augmentor.generate_synthetic_image(
            background=background,
            objects=loaded_objects,
            background_path=task.background_path,
        )

        # Validate result
        is_valid, issues = augmentor.validate_synthetic_image(result)
        if not is_valid:
            return GenerationResult(
                task_id=task.task_id,
                success=False,
                error=f"Validation failed: {issues}"
            )

        # Generate output filenames with worker PID for uniqueness
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:18]
        worker_pid = os.getpid()
        filename = f"synthetic_{timestamp}_{worker_pid}_{task.task_id:05d}"

        # Create output directories
        images_out = Path(output_dir) / "images"
        labels_out = Path(output_dir) / "labels"
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)

        # Save image
        img_path = images_out / f"{filename}.{config.output_image_format}"
        if config.output_image_format == "jpg":
            cv2.imwrite(
                str(img_path),
                result.image,
                [cv2.IMWRITE_JPEG_QUALITY, config.output_quality],
            )
        else:
            cv2.imwrite(str(img_path), result.image)

        # Generate and save YOLO labels
        labels = augmentor.generate_yolo_labels(
            result.paste_results, result.image.shape[:2]
        )
        label_path = labels_out / f"{filename}.txt"
        with open(label_path, "w") as f:
            f.write("\n".join(labels))

        # Collect statistics
        class_counts = {}
        for pr in result.paste_results:
            class_counts[pr.class_name] = class_counts.get(pr.class_name, 0) + 1

        return GenerationResult(
            task_id=task.task_id,
            success=True,
            image_path=str(img_path),
            label_path=str(label_path),
            objects_placed=len(result.paste_results),
            class_counts=class_counts,
        )

    except (IOError, OSError) as io_err:
        logger.error(f"Task {task.task_id} I/O error: {io_err}")
        return GenerationResult(
            task_id=task.task_id,
            success=False,
            error=f"I/O error: {io_err}"
        )
    except cv2.error as cv_err:
        logger.error(f"Task {task.task_id} OpenCV error: {cv_err}")
        return GenerationResult(
            task_id=task.task_id,
            success=False,
            error=f"OpenCV error: {cv_err}"
        )
    except (ValueError, TypeError, KeyError) as data_err:
        logger.error(f"Task {task.task_id} data error: {data_err}")
        return GenerationResult(
            task_id=task.task_id,
            success=False,
            error=f"Data error: {data_err}"
        )
    except Exception as e:
        # Catch any unexpected errors
        logger.error(f"Task {task.task_id} unexpected error: {e}")
        return GenerationResult(
            task_id=task.task_id,
            success=False,
            error=f"Unexpected error: {e}"
        )


def _load_single_object_worker(
    ref: ObjectReference,
    alpha_blur_sigma: float = 2.0,
    padding: int = 5,
) -> Optional[Tuple[ExtractedObject, str]]:
    """
    Load a single object on-demand in worker process.

    Args:
        ref: ObjectReference containing file paths
        alpha_blur_sigma: Gaussian blur sigma for edge softening
        padding: Padding around object when cropping

    Returns:
        Tuple of (ExtractedObject, mask_path) or None if loading fails
    """
    try:
        image_bgr = cv2.imread(ref.image_path)
        mask_gray = cv2.imread(ref.mask_path, cv2.IMREAD_GRAYSCALE)

        if image_bgr is None or mask_gray is None:
            return None

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mask_bool = mask_gray > 127

        extractor = ObjectExtractor(
            alpha_blur_sigma=alpha_blur_sigma,
            padding=padding,
        )

        obj = extractor.extract_object(
            image_rgb=image_rgb,
            mask=mask_bool,
            class_id=ref.class_id,
            class_name=ref.class_name,
            source_path=ref.image_path,
        )

        # Explicitly release intermediate image data to free memory
        del image_bgr, image_rgb, mask_gray, mask_bool

        if obj is None:
            return None

        return (obj, ref.mask_path)

    except (IOError, OSError) as io_err:
        logger.warning(f"I/O error loading object {ref.image_path}: {io_err}")
        return None
    except cv2.error as cv_err:
        logger.warning(f"OpenCV error loading object {ref.image_path}: {cv_err}")
        return None
    except Exception as e:
        logger.warning(f"Unexpected error loading object {ref.image_path}: {e}")
        return None


class ParallelSyntheticGenerator:
    """
    Parallel synthetic image generator using ProcessPoolExecutor.

    Distributes synthetic image generation across multiple CPU cores
    for faster data augmentation during competition day.
    """

    def __init__(self, config: CopyPasteConfig, num_workers: Optional[int] = None):
        """
        Initialize parallel generator.

        Args:
            config: CopyPasteConfig for augmentation settings
            num_workers: Number of worker processes (default: cpu_count // 2)

        Raises:
            ValueError: If num_workers is less than 1
        """
        self.config = config

        # Calculate default workers
        if num_workers is None:
            self.num_workers = max(1, os.cpu_count() // 2)
        else:
            # Validate num_workers
            if not isinstance(num_workers, int) or num_workers < 1:
                raise ValueError(
                    f"num_workers must be a positive integer, got {num_workers}"
                )
            # Cap at CPU count to prevent resource exhaustion
            max_workers = os.cpu_count() or 4
            if num_workers > max_workers:
                logger.warning(
                    f"num_workers ({num_workers}) exceeds CPU count ({max_workers}), "
                    f"capping at {max_workers}"
                )
                num_workers = max_workers
            self.num_workers = num_workers

        logger.info(f"ParallelSyntheticGenerator initialized with {self.num_workers} workers")

    def generate_batch_parallel(
        self,
        backgrounds: List[Path],
        object_refs: List[ObjectReference],
        output_dir: Path,
        num_synthetic: int,
        class_names: List[str],
        target_resolution: Optional[Tuple[int, int]] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> Dict:
        """
        Generate synthetic images in parallel.

        Args:
            backgrounds: List of background image paths
            object_refs: List of ObjectReference instances
            output_dir: Output directory for synthetic images
            num_synthetic: Number of synthetic images to generate
            class_names: List of class names
            target_resolution: Optional target (height, width) for backgrounds
            progress_callback: Optional callback(current, total, message)

        Returns:
            Statistics dictionary with generation results
        """
        if not backgrounds:
            return {"error": "No background images provided", "generated": 0}

        if not object_refs:
            return {"error": "No object references provided", "generated": 0}

        # Determine target resolution
        if target_resolution is None:
            target_resolution = self._get_target_resolution(object_refs)
            if target_resolution is None:
                return {"error": "Could not determine target resolution", "generated": 0}

        logger.info(f"Target resolution: {target_resolution}")

        # Serialize ObjectReference for pickle compatibility
        object_refs_data = [
            {
                "image_path": ref.image_path,
                "mask_path": ref.mask_path,
                "class_id": ref.class_id,
                "class_name": ref.class_name,
            }
            for ref in object_refs
        ]

        # Serialize config
        config_dict = asdict(self.config)

        # Create output directories
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / "labels").mkdir(parents=True, exist_ok=True)

        # Create generation tasks
        tasks = self._create_tasks(
            backgrounds=backgrounds,
            object_refs=object_refs,
            num_synthetic=num_synthetic,
            target_resolution=target_resolution,
            config_dict=config_dict,
        )

        logger.info(f"Created {len(tasks)} generation tasks")

        # Initialize statistics
        stats = {
            "generated": 0,
            "failed": 0,
            "per_class": {name: 0 for name in class_names},
            "total_objects_placed": 0,
            "avg_objects_per_image": 0.0,
        }

        # Execute tasks in parallel
        completed = 0
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(
                    worker_generate_single,
                    task,
                    object_refs_data,
                    str(output_dir),
                    class_names,
                ): task
                for task in tasks
            }

            # Process results as they complete
            for future in as_completed(future_to_task):
                completed += 1

                try:
                    result = future.result()

                    if result.success:
                        stats["generated"] += 1
                        stats["total_objects_placed"] += result.objects_placed

                        # Update per-class counts
                        for class_name, count in result.class_counts.items():
                            if class_name in stats["per_class"]:
                                stats["per_class"][class_name] += count
                    else:
                        stats["failed"] += 1
                        logger.warning(f"Task {result.task_id} failed: {result.error}")

                    # Progress callback
                    if progress_callback:
                        progress_callback(
                            completed,
                            num_synthetic,
                            f"Generated {stats['generated']}/{num_synthetic}"
                        )

                except Exception as e:
                    stats["failed"] += 1
                    logger.error(f"Future execution failed: {e}")

                    if progress_callback:
                        progress_callback(
                            completed,
                            num_synthetic,
                            f"Error: {str(e)}"
                        )

        # Calculate average objects per image
        if stats["generated"] > 0:
            stats["avg_objects_per_image"] = stats["total_objects_placed"] / stats["generated"]

        logger.info(
            f"Generation complete: {stats['generated']} succeeded, "
            f"{stats['failed']} failed"
        )

        return stats

    def _create_tasks(
        self,
        backgrounds: List[Path],
        object_refs: List[ObjectReference],
        num_synthetic: int,
        target_resolution: Tuple[int, int],
        config_dict: Dict,
    ) -> List[GenerationTask]:
        """
        Create generation tasks for parallel execution.

        Args:
            backgrounds: List of background image paths
            object_refs: List of ObjectReference instances
            num_synthetic: Number of synthetic images to generate
            target_resolution: Target (height, width) for backgrounds
            config_dict: Serialized CopyPasteConfig

        Returns:
            List of GenerationTask instances
        """
        tasks = []
        rng = np.random.RandomState(self.config.seed)

        for i in range(num_synthetic):
            # Select random background
            bg_path = backgrounds[rng.randint(0, len(backgrounds))]

            # Determine number of objects to place
            num_objects = rng.randint(
                self.config.min_objects_per_image,
                self.config.max_objects_per_image + 1,
            )
            num_objects = min(num_objects, len(object_refs))

            # Randomly select object indices
            if num_objects > 0:
                object_indices = rng.choice(
                    len(object_refs), size=num_objects, replace=False
                ).tolist()
            else:
                object_indices = []

            # Create unique seed for this task
            task_seed = self.config.seed + i

            tasks.append(
                GenerationTask(
                    task_id=i,
                    background_path=str(bg_path),
                    object_ref_indices=object_indices,
                    target_resolution=target_resolution,
                    config_dict=config_dict,
                    seed=task_seed,
                )
            )

        return tasks

    def _get_target_resolution(
        self,
        object_refs: List[ObjectReference],
    ) -> Optional[Tuple[int, int]]:
        """
        Get target resolution from the first valid object reference.

        Args:
            object_refs: List of object references

        Returns:
            Tuple of (height, width) or None if no valid images
        """
        if not object_refs:
            return None

        for ref in object_refs:
            img = cv2.imread(ref.image_path)
            if img is not None:
                resolution = img.shape[:2]  # (height, width)
                del img  # Immediately release memory
                return resolution

        return None


if __name__ == "__main__":
    # Test code
    print("ParallelSyntheticGenerator module loaded successfully")

    config = CopyPasteConfig(
        synthetic_to_real_ratio=2.0,
        scale_range=(0.5, 1.5),
        enable_white_balance=True,
        seed=42,
    )

    generator = ParallelSyntheticGenerator(config=config, num_workers=4)
    print(f"Initialized with {generator.num_workers} workers")
    print(f"Config: synthetic_ratio={config.synthetic_to_real_ratio}")
