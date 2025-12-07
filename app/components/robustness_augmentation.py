"""
Robustness Test Augmentation Utilities

Provides image transformations for robustness testing:
- Brightness/lighting variation
- Shadow injection
- Occlusion simulation
- Similar object generation (Hue rotation)
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from pathlib import Path


@dataclass
class AugmentationResult:
    """Result of an augmentation operation."""
    image: np.ndarray
    name: str
    description: str
    params: dict


class RobustnessAugmentor:
    """
    Generates augmented images for robustness testing.

    Supports:
    - Brightness variation (lighting changes)
    - Shadow injection (random shadows on objects)
    - Occlusion simulation (random overlapping objects)
    - Hue rotation (similar object generation)
    """

    def __init__(self, seed: int = 42):
        """Initialize augmentor with random seed for reproducibility."""
        self.rng = np.random.RandomState(seed)

    # =========================================================================
    # Brightness / Lighting Variation
    # =========================================================================

    def adjust_brightness(self, image: np.ndarray, factor: float) -> np.ndarray:
        """
        Adjust image brightness.

        Args:
            image: BGR image
            factor: Brightness factor (0.0 = black, 1.0 = original, 2.0 = double)

        Returns:
            Adjusted BGR image
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def generate_brightness_variants(
        self,
        image: np.ndarray,
        factors: List[float] = None
    ) -> List[AugmentationResult]:
        """
        Generate brightness variants of an image.

        Args:
            image: BGR image
            factors: List of brightness factors (default: [0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 1.8])

        Returns:
            List of AugmentationResult
        """
        if factors is None:
            factors = [0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 1.8]

        results = []
        for factor in factors:
            aug_image = self.adjust_brightness(image, factor)

            if factor < 1.0:
                name = f"Dark ({int(factor * 100)}%)"
                desc = f"Reduced brightness to {int(factor * 100)}%"
            elif factor > 1.0:
                name = f"Bright ({int(factor * 100)}%)"
                desc = f"Increased brightness to {int(factor * 100)}%"
            else:
                name = "Original"
                desc = "Original brightness"

            results.append(AugmentationResult(
                image=aug_image,
                name=name,
                description=desc,
                params={"brightness_factor": factor}
            ))

        return results

    # =========================================================================
    # Shadow Injection
    # =========================================================================

    def inject_shadow(
        self,
        image: np.ndarray,
        intensity: float = 0.5,
        num_shadows: int = 1,
        shadow_type: str = "random"
    ) -> np.ndarray:
        """
        Inject shadows onto an image.

        Args:
            image: BGR image
            intensity: Shadow darkness (0.0 = invisible, 1.0 = black)
            num_shadows: Number of shadow regions
            shadow_type: "random", "diagonal", "circular"

        Returns:
            Image with shadows
        """
        result = image.copy()
        h, w = image.shape[:2]

        for _ in range(num_shadows):
            if shadow_type == "diagonal":
                # Diagonal shadow across the image
                mask = self._create_diagonal_shadow_mask(h, w)
            elif shadow_type == "circular":
                # Circular shadow
                mask = self._create_circular_shadow_mask(h, w)
            else:  # random
                # Random polygon shadow
                mask = self._create_random_shadow_mask(h, w)

            # Apply shadow (darken the masked region)
            shadow_factor = 1.0 - intensity
            for c in range(3):
                result[:, :, c] = np.where(
                    mask > 0,
                    (result[:, :, c] * shadow_factor).astype(np.uint8),
                    result[:, :, c]
                )

        return result

    def _create_random_shadow_mask(self, h: int, w: int) -> np.ndarray:
        """Create a random polygon shadow mask."""
        mask = np.zeros((h, w), dtype=np.uint8)

        # Generate random polygon vertices
        num_vertices = self.rng.randint(3, 7)
        vertices = []

        # Start from a random edge
        start_x = self.rng.randint(0, w)
        start_y = 0
        vertices.append([start_x, start_y])

        # Add random intermediate points
        for _ in range(num_vertices - 2):
            x = self.rng.randint(0, w)
            y = self.rng.randint(0, h)
            vertices.append([x, y])

        # End at another edge
        end_x = self.rng.randint(0, w)
        end_y = h
        vertices.append([end_x, end_y])

        pts = np.array(vertices, np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 255)

        # Apply Gaussian blur for soft edges
        mask = cv2.GaussianBlur(mask, (51, 51), 0)

        return mask

    def _create_diagonal_shadow_mask(self, h: int, w: int) -> np.ndarray:
        """Create a diagonal shadow mask."""
        mask = np.zeros((h, w), dtype=np.uint8)

        # Random diagonal direction
        if self.rng.rand() > 0.5:
            pts = np.array([[0, 0], [w, 0], [w, h // 2], [0, h]], np.int32)
        else:
            pts = np.array([[0, h // 2], [w, 0], [w, h], [0, h]], np.int32)

        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 255)
        mask = cv2.GaussianBlur(mask, (101, 101), 0)

        return mask

    def _create_circular_shadow_mask(self, h: int, w: int) -> np.ndarray:
        """Create a circular shadow mask."""
        mask = np.zeros((h, w), dtype=np.uint8)

        # Random center and radius
        cx = self.rng.randint(w // 4, 3 * w // 4)
        cy = self.rng.randint(h // 4, 3 * h // 4)
        radius = self.rng.randint(min(h, w) // 4, min(h, w) // 2)

        cv2.circle(mask, (cx, cy), radius, 255, -1)
        mask = cv2.GaussianBlur(mask, (101, 101), 0)

        return mask

    def generate_shadow_variants(
        self,
        image: np.ndarray,
        intensities: List[float] = None
    ) -> List[AugmentationResult]:
        """
        Generate shadow variants of an image.

        Args:
            image: BGR image
            intensities: List of shadow intensities (default: [0.2, 0.4, 0.6])

        Returns:
            List of AugmentationResult
        """
        if intensities is None:
            intensities = [0.2, 0.4, 0.6]

        results = []
        shadow_types = ["random", "diagonal", "circular"]

        for intensity in intensities:
            for shadow_type in shadow_types:
                aug_image = self.inject_shadow(
                    image,
                    intensity=intensity,
                    shadow_type=shadow_type
                )

                level = "Light" if intensity < 0.4 else "Medium" if intensity < 0.6 else "Heavy"

                results.append(AugmentationResult(
                    image=aug_image,
                    name=f"{level} {shadow_type.capitalize()} Shadow",
                    description=f"{shadow_type.capitalize()} shadow with {int(intensity * 100)}% intensity",
                    params={"shadow_intensity": intensity, "shadow_type": shadow_type}
                ))

        return results

    # =========================================================================
    # Occlusion Simulation
    # =========================================================================

    def inject_occlusion(
        self,
        image: np.ndarray,
        occlusion_ratio: float = 0.3,
        occlusion_type: str = "rectangle",
        color: Tuple[int, int, int] = None
    ) -> np.ndarray:
        """
        Inject occlusion onto an image.

        Args:
            image: BGR image
            occlusion_ratio: Ratio of image to occlude (0.0 - 1.0)
            occlusion_type: "rectangle", "random", "edge"
            color: Occlusion color (default: random gray)

        Returns:
            Image with occlusion
        """
        result = image.copy()
        h, w = image.shape[:2]

        if color is None:
            # Random gray color (simulating another object)
            gray = self.rng.randint(50, 200)
            color = (gray, gray, gray)

        if occlusion_type == "rectangle":
            # Random rectangle occlusion
            occ_w = int(w * occlusion_ratio)
            occ_h = int(h * occlusion_ratio)
            x = self.rng.randint(0, max(1, w - occ_w))
            y = self.rng.randint(0, max(1, h - occ_h))
            cv2.rectangle(result, (x, y), (x + occ_w, y + occ_h), color, -1)

        elif occlusion_type == "edge":
            # Occlusion from edge (simulating object partially out of frame)
            edge = self.rng.choice(["top", "bottom", "left", "right"])
            occ_size = int(min(h, w) * occlusion_ratio)

            if edge == "top":
                cv2.rectangle(result, (0, 0), (w, occ_size), color, -1)
            elif edge == "bottom":
                cv2.rectangle(result, (0, h - occ_size), (w, h), color, -1)
            elif edge == "left":
                cv2.rectangle(result, (0, 0), (occ_size, h), color, -1)
            else:  # right
                cv2.rectangle(result, (w - occ_size, 0), (w, h), color, -1)

        else:  # random polygon
            num_vertices = self.rng.randint(4, 8)
            cx = self.rng.randint(w // 4, 3 * w // 4)
            cy = self.rng.randint(h // 4, 3 * h // 4)
            radius = int(min(h, w) * occlusion_ratio / 2)

            angles = np.linspace(0, 2 * np.pi, num_vertices, endpoint=False)
            angles += self.rng.rand(num_vertices) * 0.5  # Add randomness
            radii = radius * (0.7 + 0.6 * self.rng.rand(num_vertices))

            vertices = []
            for angle, r in zip(angles, radii):
                x = int(cx + r * np.cos(angle))
                y = int(cy + r * np.sin(angle))
                vertices.append([x, y])

            pts = np.array(vertices, np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(result, [pts], color)

        return result

    def generate_occlusion_variants(
        self,
        image: np.ndarray,
        ratios: List[float] = None
    ) -> List[AugmentationResult]:
        """
        Generate occlusion variants of an image.

        Args:
            image: BGR image
            ratios: List of occlusion ratios (default: [0.1, 0.2, 0.3, 0.4])

        Returns:
            List of AugmentationResult
        """
        if ratios is None:
            ratios = [0.1, 0.2, 0.3, 0.4]

        results = []
        occlusion_types = ["rectangle", "edge", "random"]

        for ratio in ratios:
            for occ_type in occlusion_types:
                aug_image = self.inject_occlusion(
                    image,
                    occlusion_ratio=ratio,
                    occlusion_type=occ_type
                )

                level = "Minor" if ratio < 0.2 else "Moderate" if ratio < 0.35 else "Severe"

                results.append(AugmentationResult(
                    image=aug_image,
                    name=f"{level} {occ_type.capitalize()} Occlusion",
                    description=f"{occ_type.capitalize()} occlusion covering {int(ratio * 100)}% of image",
                    params={"occlusion_ratio": ratio, "occlusion_type": occ_type}
                ))

        return results

    # =========================================================================
    # Hue Rotation (Similar Object Generation)
    # =========================================================================

    def rotate_hue(self, image: np.ndarray, angle: int) -> np.ndarray:
        """
        Rotate hue of an image in HSV color space.

        Args:
            image: BGR image
            angle: Hue rotation angle in degrees (0-360)

        Returns:
            Hue-rotated BGR image
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)

        # OpenCV uses 0-180 for Hue
        hsv[:, :, 0] = (hsv[:, :, 0] + angle / 2) % 180

        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def generate_hue_variants(
        self,
        image: np.ndarray,
        angles: List[int] = None
    ) -> List[AugmentationResult]:
        """
        Generate hue-rotated variants of an image (similar objects).

        Args:
            image: BGR image
            angles: List of hue rotation angles (default: [30, 60, 90, 120, 150, 180])

        Returns:
            List of AugmentationResult
        """
        if angles is None:
            angles = [0, 30, 60, 90, 120, 150, 180]

        results = []
        for angle in angles:
            aug_image = self.rotate_hue(image, angle)

            if angle == 0:
                name = "Original"
                desc = "Original hue"
            else:
                name = f"Hue +{angle}"
                desc = f"Hue rotated by {angle} degrees"

            results.append(AugmentationResult(
                image=aug_image,
                name=name,
                description=desc,
                params={"hue_angle": angle}
            ))

        return results

    def extract_object_from_bbox(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        padding: int = 10
    ) -> np.ndarray:
        """
        Extract object region from image using bounding box.

        Args:
            image: BGR image
            bbox: Bounding box (x1, y1, x2, y2)
            padding: Padding around bbox

        Returns:
            Cropped object image
        """
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox

        # Apply padding with bounds checking
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)

        return image[y1:y2, x1:x2].copy()

    # =========================================================================
    # Batch Processing
    # =========================================================================

    def run_full_robustness_test(
        self,
        image: np.ndarray,
        include_brightness: bool = True,
        include_shadow: bool = True,
        include_occlusion: bool = True,
        include_hue: bool = True
    ) -> dict:
        """
        Run all robustness tests on an image.

        Args:
            image: BGR image
            include_*: Flags to include specific test types

        Returns:
            Dictionary with test category -> List[AugmentationResult]
        """
        results = {}

        if include_brightness:
            results["brightness"] = self.generate_brightness_variants(image)

        if include_shadow:
            results["shadow"] = self.generate_shadow_variants(image)

        if include_occlusion:
            results["occlusion"] = self.generate_occlusion_variants(image)

        if include_hue:
            results["hue"] = self.generate_hue_variants(image)

        return results


def load_annotations_from_yolo(label_path: Path, image_shape: Tuple[int, int]) -> List[dict]:
    """
    Load YOLO format annotations.

    Args:
        label_path: Path to YOLO label file (.txt)
        image_shape: (height, width) of the image

    Returns:
        List of annotation dicts with 'class_id' and 'bbox' (x1, y1, x2, y2)
    """
    h, w = image_shape
    annotations = []

    if not label_path.exists():
        return annotations

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])

                # Convert YOLO format to pixel coordinates
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)

                annotations.append({
                    "class_id": class_id,
                    "bbox": (x1, y1, x2, y2)
                })

    return annotations
