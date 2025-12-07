"""
Configuration Preview Components

Settings summary cards, GPU status display, and validation UI
with Mission Control aesthetic styling.
"""

import streamlit as st
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any

from .training_styles import COLORS, ICONS, get_tier_class


@dataclass
class ValidationResult:
    """Result of configuration validation."""
    is_valid: bool
    warnings: List[str]
    errors: List[str]


def validate_training_config(
    dataset_yaml: str,
    model: str,
    batch_size: int,
    epochs: int,
    gpu_memory_gb: float,
    auto_scale: bool = False,
) -> ValidationResult:
    """
    Validate training configuration with helpful messages.

    Args:
        dataset_yaml: Path to dataset YAML
        model: Model name (e.g., 'yolov8m.pt')
        batch_size: Batch size
        epochs: Number of epochs
        gpu_memory_gb: Available GPU memory in GB
        auto_scale: Whether auto-scaling is enabled

    Returns:
        ValidationResult with is_valid, warnings, and errors
    """
    errors = []
    warnings = []

    # Dataset validation
    if dataset_yaml and not Path(dataset_yaml).exists():
        errors.append(f"Dataset file not found: {dataset_yaml}")

    # Skip VRAM checks if auto-scaling is enabled
    if not auto_scale and gpu_memory_gb > 0:
        # Model VRAM overhead estimates (GB)
        model_overhead = {
            "yolov8n.pt": 1.2,
            "yolov8s.pt": 1.8,
            "yolov8m.pt": 2.5,
            "yolov8l.pt": 3.5,
            "yolov8x.pt": 5.0,
        }.get(model, 2.5)

        # Per-sample memory at 640x640 (MB)
        per_sample_mb = {
            "yolov8n.pt": 180,
            "yolov8s.pt": 250,
            "yolov8m.pt": 380,
            "yolov8l.pt": 550,
            "yolov8x.pt": 750,
        }.get(model, 380)

        # Estimate VRAM usage
        estimated_vram = model_overhead + (batch_size * per_sample_mb / 1024)

        if estimated_vram > gpu_memory_gb * 0.9:
            warnings.append(
                f"Estimated VRAM usage ({estimated_vram:.1f}GB) may exceed "
                f"available memory ({gpu_memory_gb:.1f}GB). Consider reducing batch size."
            )

    # Epochs validation
    if epochs < 20:
        warnings.append(
            "Low epoch count (<20) may result in underfitting. "
            "Consider at least 30-50 epochs for better results."
        )
    elif epochs > 100:
        warnings.append(
            "High epoch count (>100) may lead to overfitting. "
            "Monitor validation metrics carefully."
        )

    # Batch size validation
    if batch_size < 4:
        warnings.append(
            "Very small batch size (<4) may cause unstable training. "
            "Consider increasing if GPU memory allows."
        )

    return ValidationResult(
        is_valid=len(errors) == 0,
        warnings=warnings,
        errors=errors,
    )


def render_validation_messages(result: ValidationResult):
    """
    Render validation warnings and errors.

    Args:
        result: ValidationResult from validate_training_config
    """
    for error in result.errors:
        st.error(f"✕ {error}")

    for warning in result.warnings:
        st.warning(f"⚠ {warning}")


def render_gpu_status_card(
    gpu_name: str,
    gpu_memory: float,
    gpu_tier: str,
    auto_scale_enabled: bool = False,
):
    """
    Render GPU status card with tier indicator.

    Args:
        gpu_name: GPU device name
        gpu_memory: Total GPU memory in GB
        gpu_tier: GPU tier classification
        auto_scale_enabled: Whether auto-scaling is enabled
    """
    tier_class = get_tier_class(gpu_tier)
    tier_icon = ICONS.get(f"gpu_{tier_class}", ICONS["gpu_medium"])

    with st.container(border=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**{tier_icon} {gpu_name}**")
            st.caption(f"{gpu_memory:.1f} GB VRAM")
        with col2:
            st.markdown(f"`{gpu_tier}`")

        if auto_scale_enabled:
            st.caption("✓ Auto-scaling enabled")


def render_gpu_not_available():
    """Render warning when GPU is not available."""
    st.warning("⚠ **CUDA not available** - Training will run on CPU and be significantly slower. Consider using a machine with GPU support.")


def render_config_summary(
    dataset_name: str,
    train_count: int,
    val_count: int,
    model: str,
    batch_size: int,
    epochs: int,
    gpu_tier: str,
    estimated_time: float,
    auto_scale: bool = False,
):
    """
    Render configuration summary card before training starts.

    Args:
        dataset_name: Name of the dataset
        train_count: Number of training images
        val_count: Number of validation images
        model: Model name
        batch_size: Batch size
        epochs: Number of epochs
        gpu_tier: GPU tier
        estimated_time: Estimated training time in minutes
        auto_scale: Whether auto-scaling is enabled
    """
    batch_display = f"{batch_size}" if not auto_scale else f"{batch_size} (auto)"

    with st.container(border=True):
        st.caption(f"{ICONS['model']} TRAINING CONFIGURATION")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Dataset", dataset_name)
            st.metric("Model", model)
        with col2:
            st.metric("Images", f"{train_count} / {val_count}")
            st.metric("Batch Size", batch_display)

        st.divider()

        col3, col4 = st.columns(2)
        with col3:
            st.metric("Epochs", epochs)
        with col4:
            st.metric("Est. Time", f"~{estimated_time:.0f} min")


def render_model_recommendation(
    recommended_model: str,
    recommended_batch: int,
    gpu_tier: str,
):
    """
    Render model recommendation from auto-scaling.

    Args:
        recommended_model: Recommended model name
        recommended_batch: Recommended batch size
        gpu_tier: GPU tier
    """
    tier_class = get_tier_class(gpu_tier)
    gpu_icon = ICONS.get("gpu_" + tier_class, "▫")

    with st.container(border=True):
        st.caption(f"{gpu_icon} AUTO-SCALING RECOMMENDATION")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Model", recommended_model)
        with col2:
            st.metric("Batch Size", recommended_batch)


def render_target_metrics_info():
    """Render target metrics information for competition."""
    with st.container(border=True):
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.markdown("**Competition Targets**")
        with col2:
            st.metric("mAP@50", "≥85%")
        with col3:
            st.metric("Inference", "≤100ms")
