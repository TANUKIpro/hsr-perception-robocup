"""
Robustness Test UI Component

Provides interactive UI for robustness testing of trained models.
Features:
- Real-time preview with sliders
- Batch testing across multiple conditions
- Detection comparison with confidence visualization
"""

import streamlit as st
import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass

from .robustness_augmentation import (
    RobustnessAugmentor,
    AugmentationResult,
    load_annotations_from_yolo,
)


@dataclass
class DetectionResult:
    """Detection result from model prediction."""
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]


@dataclass
class RobustnessTestResult:
    """Result of a robustness test on a single augmented image."""
    augmentation: AugmentationResult
    detections: List[DetectionResult]
    original_detections: List[DetectionResult]
    detection_count_diff: int
    avg_confidence_diff: float


def render_realtime_preview(
    model,
    image: np.ndarray,
    conf_threshold: float = 0.25
):
    """
    Render real-time preview with adjustable augmentation parameters.

    Args:
        model: YOLO model instance
        image: BGR image to test
        conf_threshold: Confidence threshold for detections
    """
    st.markdown("### Real-time Augmentation Preview")

    # Test type selection
    test_type = st.selectbox(
        "Test Type",
        ["Brightness", "Shadow", "Occlusion", "Hue Rotation (Similar Object)"],
        key="rt_test_type"
    )

    augmentor = RobustnessAugmentor()

    # Parameter controls based on test type
    aug_image = image.copy()

    if test_type == "Brightness":
        brightness = st.slider(
            "Brightness Factor",
            min_value=0.1,
            max_value=2.0,
            value=1.0,
            step=0.05,
            key="rt_brightness"
        )
        aug_image = augmentor.adjust_brightness(image, brightness)

    elif test_type == "Shadow":
        col1, col2 = st.columns(2)
        with col1:
            shadow_intensity = st.slider(
                "Shadow Intensity",
                min_value=0.1,
                max_value=0.8,
                value=0.4,
                step=0.1,
                key="rt_shadow_intensity"
            )
        with col2:
            shadow_type = st.selectbox(
                "Shadow Type",
                ["random", "diagonal", "circular"],
                key="rt_shadow_type"
            )

        if st.button("Regenerate Shadow", key="regen_shadow"):
            augmentor = RobustnessAugmentor(seed=np.random.randint(0, 10000))

        aug_image = augmentor.inject_shadow(image, shadow_intensity, shadow_type=shadow_type)

    elif test_type == "Occlusion":
        col1, col2 = st.columns(2)
        with col1:
            occlusion_ratio = st.slider(
                "Occlusion Ratio",
                min_value=0.05,
                max_value=0.5,
                value=0.2,
                step=0.05,
                key="rt_occlusion_ratio"
            )
        with col2:
            occlusion_type = st.selectbox(
                "Occlusion Type",
                ["rectangle", "edge", "random"],
                key="rt_occlusion_type"
            )

        if st.button("Regenerate Occlusion", key="regen_occlusion"):
            augmentor = RobustnessAugmentor(seed=np.random.randint(0, 10000))

        aug_image = augmentor.inject_occlusion(image, occlusion_ratio, occlusion_type)

    else:  # Hue Rotation
        hue_angle = st.slider(
            "Hue Rotation (degrees)",
            min_value=0,
            max_value=180,
            value=0,
            step=15,
            key="rt_hue_angle"
        )
        aug_image = augmentor.rotate_hue(image, hue_angle)

    # Run prediction on both images
    with st.spinner("Running predictions..."):
        orig_results = model(image, conf=conf_threshold, verbose=False)[0]
        aug_results = model(aug_image, conf=conf_threshold, verbose=False)[0]

    # Display side by side
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Original**")
        orig_annotated = orig_results.plot()
        st.image(orig_annotated, channels="BGR", use_container_width=True)
        st.caption(f"Detections: {len(orig_results.boxes)}")

    with col2:
        st.markdown("**Augmented**")
        aug_annotated = aug_results.plot()
        st.image(aug_annotated, channels="BGR", use_container_width=True)
        st.caption(f"Detections: {len(aug_results.boxes)}")

    # Comparison metrics
    _render_detection_comparison(model, orig_results, aug_results)


def render_batch_test(
    model,
    image: np.ndarray,
    conf_threshold: float = 0.25
):
    """
    Render batch test interface for comprehensive robustness evaluation.

    Args:
        model: YOLO model instance
        image: BGR image to test
        conf_threshold: Confidence threshold for detections
    """
    st.markdown("### Batch Robustness Test")

    # Test type selection (multi-select)
    test_types = st.multiselect(
        "Select Test Types",
        ["Brightness", "Shadow", "Occlusion", "Hue Rotation"],
        default=["Brightness", "Shadow"],
        key="batch_test_types"
    )

    if not test_types:
        st.info("Please select at least one test type.")
        return

    if st.button("Run Batch Test", type="primary", key="run_batch"):
        augmentor = RobustnessAugmentor()
        all_results = []

        # Get original detection count
        orig_results = model(image, conf=conf_threshold, verbose=False)[0]
        orig_count = len(orig_results.boxes)
        orig_conf = _get_avg_confidence(orig_results)

        with st.spinner("Running batch tests..."):
            progress = st.progress(0)

            if "Brightness" in test_types:
                brightness_augs = augmentor.generate_brightness_variants(image)
                for aug in brightness_augs:
                    result = _run_single_test(model, aug, orig_count, orig_conf, conf_threshold)
                    all_results.append(("Brightness", result))
                progress.progress(0.25)

            if "Shadow" in test_types:
                shadow_augs = augmentor.generate_shadow_variants(image)
                for aug in shadow_augs:
                    result = _run_single_test(model, aug, orig_count, orig_conf, conf_threshold)
                    all_results.append(("Shadow", result))
                progress.progress(0.5)

            if "Occlusion" in test_types:
                occlusion_augs = augmentor.generate_occlusion_variants(image)
                for aug in occlusion_augs:
                    result = _run_single_test(model, aug, orig_count, orig_conf, conf_threshold)
                    all_results.append(("Occlusion", result))
                progress.progress(0.75)

            if "Hue Rotation" in test_types:
                hue_augs = augmentor.generate_hue_variants(image)
                for aug in hue_augs:
                    result = _run_single_test(model, aug, orig_count, orig_conf, conf_threshold)
                    all_results.append(("Hue", result))
                progress.progress(1.0)

        # Display results
        _render_batch_results(model, all_results, orig_count, image)


def render_similar_object_test(
    model,
    dataset_path: Path,
    conf_threshold: float = 0.25
):
    """
    Render similar object test using dataset annotations.

    Args:
        model: YOLO model instance
        dataset_path: Path to dataset directory
        conf_threshold: Confidence threshold
    """
    st.markdown("### Similar Object Test (Hue Rotation)")

    # Get validation images
    val_dir = dataset_path / "images" / "val"
    label_dir = dataset_path / "labels" / "val"

    if not val_dir.exists():
        st.warning("Validation images not found.")
        return

    images = list(val_dir.glob("*.jpg")) + list(val_dir.glob("*.png"))
    if not images:
        st.warning("No images found in validation set.")
        return

    # Image selection
    selected_image = st.selectbox(
        "Select Image",
        images[:30],
        format_func=lambda x: x.name,
        key="similar_obj_image"
    )

    if not selected_image:
        return

    # Load image and annotations
    image = cv2.imread(str(selected_image))
    if image is None:
        st.error("Failed to load image.")
        return

    label_path = label_dir / (selected_image.stem + ".txt")
    annotations = load_annotations_from_yolo(label_path, image.shape[:2])

    if not annotations:
        st.info("No annotations found for this image. Select another image.")
        return

    # Get class names from model
    class_names = model.names

    # Select object to test
    object_options = []
    for i, ann in enumerate(annotations):
        class_id = ann["class_id"]
        class_label = class_names.get(class_id, f"Class {class_id}")
        object_options.append(f"Object {i+1}: {class_label}")

    selected_obj_idx = st.selectbox(
        "Select Object",
        range(len(annotations)),
        format_func=lambda x: object_options[x],
        key="selected_object"
    )

    if selected_obj_idx is None:
        return

    # Extract object region
    bbox = annotations[selected_obj_idx]["bbox"]
    augmentor = RobustnessAugmentor()
    obj_crop = augmentor.extract_object_from_bbox(image, bbox, padding=20)

    # Generate hue variants
    hue_angles = [0, 30, 60, 90, 120, 150, 180]
    st.markdown("**Hue-Rotated Variants (Similar Objects)**")

    cols = st.columns(len(hue_angles))
    for col, angle in zip(cols, hue_angles):
        with col:
            variant = augmentor.rotate_hue(obj_crop, angle)
            st.image(variant, channels="BGR", caption=f"+{angle}", use_container_width=True)

    # Run detection on full image variants
    st.markdown("---")
    st.markdown("**Detection Results on Hue-Rotated Full Images**")

    if st.button("Test All Hue Variants", key="test_hue_variants"):
        with st.spinner("Testing..."):
            orig_results = model(image, conf=conf_threshold, verbose=False)[0]
            orig_class_id = annotations[selected_obj_idx]["class_id"]
            orig_class_name = class_names.get(orig_class_id, f"Class {orig_class_id}")

            results_data = []
            for angle in hue_angles:
                aug_image = augmentor.rotate_hue(image, angle)
                aug_results = model(aug_image, conf=conf_threshold, verbose=False)[0]

                # Count detections of the same class
                same_class_count = sum(1 for box in aug_results.boxes if int(box.cls.item()) == orig_class_id)
                other_class_count = len(aug_results.boxes) - same_class_count

                # Get max confidence for the original class
                same_class_confs = [box.conf.item() for box in aug_results.boxes if int(box.cls.item()) == orig_class_id]
                max_conf = max(same_class_confs) if same_class_confs else 0

                results_data.append({
                    "Hue": f"+{angle}",
                    "Same Class": same_class_count,
                    "Other Classes": other_class_count,
                    "Max Conf": f"{max_conf:.1%}",
                    "Status": "Correct" if same_class_count > 0 else "Missed/Wrong"
                })

            # Display as table
            import pandas as pd
            df = pd.DataFrame(results_data)
            st.dataframe(df, use_container_width=True)

            # Summary
            correct_count = sum(1 for r in results_data if r["Status"] == "Correct")
            st.metric(
                "Robustness Score",
                f"{correct_count}/{len(hue_angles)}",
                delta=f"{(correct_count/len(hue_angles)*100):.0f}%",
                delta_color="normal" if correct_count >= len(hue_angles) * 0.7 else "inverse"
            )


def _run_single_test(
    model,
    augmentation: AugmentationResult,
    orig_count: int,
    orig_conf: float,
    conf_threshold: float
) -> RobustnessTestResult:
    """Run test on a single augmented image."""
    results = model(augmentation.image, conf=conf_threshold, verbose=False)[0]

    detections = []
    for box in results.boxes:
        class_id = int(box.cls.item())
        detections.append(DetectionResult(
            class_name=model.names.get(class_id, f"Class {class_id}"),
            confidence=box.conf.item(),
            bbox=tuple(map(int, box.xyxy[0].tolist()))
        ))

    aug_count = len(detections)
    aug_conf = _get_avg_confidence(results)

    return RobustnessTestResult(
        augmentation=augmentation,
        detections=detections,
        original_detections=[],  # Not needed for batch
        detection_count_diff=aug_count - orig_count,
        avg_confidence_diff=aug_conf - orig_conf if orig_conf > 0 else 0
    )


def _get_avg_confidence(results) -> float:
    """Get average confidence from results."""
    if len(results.boxes) == 0:
        return 0.0
    return sum(box.conf.item() for box in results.boxes) / len(results.boxes)


def _render_detection_comparison(model, orig_results, aug_results):
    """Render comparison between original and augmented detections."""
    st.markdown("---")
    st.markdown("**Detection Comparison**")

    col1, col2, col3 = st.columns(3)

    orig_count = len(orig_results.boxes)
    aug_count = len(aug_results.boxes)
    diff = aug_count - orig_count

    with col1:
        st.metric(
            "Detection Count",
            f"{aug_count}",
            delta=f"{diff:+d}" if diff != 0 else "Same",
            delta_color="normal" if diff >= 0 else "inverse"
        )

    orig_conf = _get_avg_confidence(orig_results)
    aug_conf = _get_avg_confidence(aug_results)
    conf_diff = aug_conf - orig_conf

    with col2:
        st.metric(
            "Avg Confidence",
            f"{aug_conf:.1%}",
            delta=f"{conf_diff:+.1%}" if conf_diff != 0 else "Same",
            delta_color="normal" if conf_diff >= 0 else "inverse"
        )

    # Robustness indicator
    with col3:
        if orig_count > 0:
            if aug_count >= orig_count and aug_conf >= orig_conf * 0.9:
                st.success("Robust")
            elif aug_count >= orig_count * 0.5:
                st.warning("Partially Robust")
            else:
                st.error("Not Robust")
        else:
            st.info("No original detections")


def _render_batch_results(model, all_results: List[Tuple[str, RobustnessTestResult]], orig_count: int, orig_image: np.ndarray):
    """Render batch test results as a summary table and charts."""
    st.markdown("---")
    st.markdown("### Batch Test Results")

    # Summary by category
    categories = {}
    for category, result in all_results:
        if category not in categories:
            categories[category] = []
        categories[category].append(result)

    # Overall metrics
    col1, col2, col3 = st.columns(3)

    total_tests = len(all_results)
    robust_count = sum(1 for _, r in all_results if r.detection_count_diff >= 0 and r.avg_confidence_diff >= -0.1)
    partial_count = sum(1 for _, r in all_results if r.detection_count_diff >= -1 and r.avg_confidence_diff >= -0.2) - robust_count
    failed_count = total_tests - robust_count - partial_count

    with col1:
        st.metric("Total Tests", total_tests)

    with col2:
        st.metric("Robust", robust_count, delta_color="off")

    with col3:
        robustness_score = robust_count / total_tests if total_tests > 0 else 0
        st.metric("Robustness Score", f"{robustness_score:.0%}")

    # Detailed results per category
    for category, results in categories.items():
        with st.expander(f"{category} Results ({len(results)} tests)", expanded=False):
            data = []
            for result in results:
                status = "Robust"
                if result.detection_count_diff < 0 or result.avg_confidence_diff < -0.1:
                    status = "Degraded"
                if result.detection_count_diff < -1 or result.avg_confidence_diff < -0.2:
                    status = "Failed"

                data.append({
                    "Augmentation": result.augmentation.name,
                    "Detections": len(result.detections),
                    "Diff": f"{result.detection_count_diff:+d}",
                    "Conf Change": f"{result.avg_confidence_diff:+.1%}",
                    "Status": status
                })

            import pandas as pd
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True)

            # Show worst cases
            worst = sorted(results, key=lambda r: r.detection_count_diff + r.avg_confidence_diff)[:3]
            if worst and worst[0].detection_count_diff < 0:
                st.markdown("**Worst Cases:**")
                cols = st.columns(len(worst))
                for col, result in zip(cols, worst):
                    with col:
                        st.image(result.augmentation.image, channels="BGR", caption=result.augmentation.name, use_container_width=True)
