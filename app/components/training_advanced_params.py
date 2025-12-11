"""
Training Advanced Parameters Component

Provides UI for configuring detailed YOLOv8 training parameters.
Features:
- Preset selection (Competition, Fast Test, High Accuracy)
- Tabbed interface for parameter categories
- Compact multi-column layout
- Mission Control aesthetic
"""

import streamlit as st
from typing import Any, Dict

from .training_styles import COLORS, ICONS

# =============================================================================
# Preset Definitions
# =============================================================================

PRESETS = {
    "Competition": {
        # Competition-optimized: Balanced settings (~45 min training)
        "imgsz": 640,
        "patience": 10,
        "close_mosaic": 10,
        "freeze": 10,  # Freeze first 10 backbone layers to prevent overfitting
        "warmup_epochs": 3,  # Warmup for stable fine-tuning
        # Augmentation
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 10.0,
        "translate": 0.1,
        "scale": 0.5,
        "shear": 2.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "mixup": 0.1,
        # Optimizer
        "optimizer": "AdamW",
        "lr0": 0.001,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.001,  # Increased for regularization
        # Performance
        "workers": 8,
        "cache": True,
        "amp": True,
        # Checkpoint
        "save": True,
        "save_period": 5,
        "exist_ok": True,
    },
    "Fast Test": {
        # Fast testing: Smaller model, fewer epochs
        "imgsz": 480,
        "patience": 5,
        "close_mosaic": 5,
        "freeze": 10,  # Freeze first 10 backbone layers to prevent overfitting
        "warmup_epochs": 2,  # Shorter warmup for fast testing
        # Augmentation (reduced)
        "hsv_h": 0.015,
        "hsv_s": 0.5,
        "hsv_v": 0.3,
        "degrees": 5.0,
        "translate": 0.1,
        "scale": 0.3,
        "shear": 1.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "mosaic": 0.8,
        "mixup": 0.0,
        # Optimizer
        "optimizer": "AdamW",
        "lr0": 0.001,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        # Performance
        "workers": 4,
        "cache": True,
        "amp": True,
        # Checkpoint
        "save": True,
        "save_period": 10,
        "exist_ok": True,
    },
    "High Accuracy": {
        # High accuracy: Stronger augmentation, optimized to prevent overfitting
        "imgsz": 640,
        "patience": 10,  # Reduced from 15 to prevent overfitting
        "close_mosaic": 10,  # Match patience
        "freeze": 10,  # Freeze first 10 backbone layers to prevent overfitting
        "warmup_epochs": 3,  # Warmup for stable fine-tuning
        # Augmentation (stronger)
        "hsv_h": 0.02,
        "hsv_s": 0.8,
        "hsv_v": 0.5,
        "degrees": 15.0,
        "translate": 0.15,
        "scale": 0.6,
        "shear": 3.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "mixup": 0.2,
        # Optimizer (optimized for fine-tuning)
        "optimizer": "AdamW",
        "lr0": 0.0005,  # Reduced from 0.0008 for fine-tuning
        "lrf": 0.01,  # Adjusted final LR scale
        "momentum": 0.937,
        "weight_decay": 0.001,  # Increased from 0.0005 for regularization
        # Performance
        "workers": 8,
        "cache": True,
        "amp": True,
        # Checkpoint
        "save": True,
        "save_period": 5,
        "exist_ok": True,
    },
}


# =============================================================================
# Session State Management
# =============================================================================

def _ensure_session_state():
    """Ensure all advanced parameters are initialized in session state."""
    if "adv_params_initialized" not in st.session_state:
        # Load Competition preset as default
        _load_preset("Competition")
        st.session_state.adv_params_initialized = True
        st.session_state.current_preset = "Competition"


def _load_preset(preset_name: str) -> None:
    """Load preset values into session state."""
    if preset_name not in PRESETS:
        return

    preset = PRESETS[preset_name]
    for key, value in preset.items():
        st.session_state[f"adv_{key}"] = value


def _get_param(key: str, default: Any) -> Any:
    """Get parameter value from session state."""
    return st.session_state.get(f"adv_{key}", default)


# =============================================================================
# UI Components
# =============================================================================

def _render_preset_selector() -> str:
    """Render preset selection buttons."""
    st.html(f"""
    <div style="
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        margin-bottom: 12px;
        opacity: 0.7;
    ">Presets</div>
    """)

    col1, col2, col3 = st.columns(3)

    current_preset = st.session_state.get("current_preset", "Competition")

    with col1:
        if st.button(
            "Competition",
            use_container_width=True,
            type="primary" if current_preset == "Competition" else "secondary",
            help="Balanced settings optimized for competition day (~45 min)",
        ):
            _load_preset("Competition")
            st.session_state.current_preset = "Competition"
            st.rerun()

    with col2:
        if st.button(
            "Fast Test",
            use_container_width=True,
            type="primary" if current_preset == "Fast Test" else "secondary",
            help="Quick validation with reduced settings (~15 min)",
        ):
            _load_preset("Fast Test")
            st.session_state.current_preset = "Fast Test"
            st.rerun()

    with col3:
        if st.button(
            "High Accuracy",
            use_container_width=True,
            type="primary" if current_preset == "High Accuracy" else "secondary",
            help="Maximum accuracy with stronger augmentation (~90 min)",
        ):
            _load_preset("High Accuracy")
            st.session_state.current_preset = "High Accuracy"
            st.rerun()

    return current_preset


def _render_augmentation_tab() -> Dict[str, float]:
    """Render Augmentation tab with HSV, geometric, and flip parameters."""
    params = {}

    # HSV Color Space
    st.html(f"""
    <div style="
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        opacity: 0.6;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    ">Color Space (HSV)</div>
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        params["hsv_h"] = st.slider(
            "Hue",
            min_value=0.0,
            max_value=0.1,
            value=_get_param("hsv_h", 0.015),
            step=0.005,
            help="Hue shift range. Helps adapt to lighting changes.",
            key="adv_hsv_h",
        )
    with col2:
        params["hsv_s"] = st.slider(
            "Saturation",
            min_value=0.0,
            max_value=1.0,
            value=_get_param("hsv_s", 0.7),
            step=0.05,
            help="Saturation shift range. Adjusts color vividness.",
            key="adv_hsv_s",
        )
    with col3:
        params["hsv_v"] = st.slider(
            "Value",
            min_value=0.0,
            max_value=1.0,
            value=_get_param("hsv_v", 0.4),
            step=0.05,
            help="Value (brightness) shift range.",
            key="adv_hsv_v",
        )

    st.markdown("<div style='height: 16px'></div>", unsafe_allow_html=True)

    # Geometric Transforms
    st.html(f"""
    <div style="
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        opacity: 0.6;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    ">Geometric Transforms</div>
    """)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        params["degrees"] = st.slider(
            "Rotation",
            min_value=0.0,
            max_value=45.0,
            value=_get_param("degrees", 10.0),
            step=1.0,
            help="Rotation angle range in degrees.",
            key="adv_degrees",
        )
    with col2:
        params["translate"] = st.slider(
            "Translation",
            min_value=0.0,
            max_value=0.5,
            value=_get_param("translate", 0.1),
            step=0.05,
            help="Translation range as fraction of image size.",
            key="adv_translate",
        )
    with col3:
        params["scale"] = st.slider(
            "Scale",
            min_value=0.0,
            max_value=1.0,
            value=_get_param("scale", 0.5),
            step=0.05,
            help="Scale range (0.5 means 0.5x-1.5x).",
            key="adv_scale",
        )
    with col4:
        params["shear"] = st.slider(
            "Shear",
            min_value=0.0,
            max_value=10.0,
            value=_get_param("shear", 2.0),
            step=0.5,
            help="Shear angle range in degrees.",
            key="adv_shear",
        )

    st.markdown("<div style='height: 16px'></div>", unsafe_allow_html=True)

    # Flip & Advanced
    st.html(f"""
    <div style="
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        opacity: 0.6;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    ">Flip & Advanced</div>
    """)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        params["flipud"] = st.slider(
            "Flip Vertical",
            min_value=0.0,
            max_value=1.0,
            value=_get_param("flipud", 0.0),
            step=0.1,
            help="Vertical flip probability. Usually 0 for object detection.",
            key="adv_flipud",
        )
    with col2:
        params["fliplr"] = st.slider(
            "Flip Horizontal",
            min_value=0.0,
            max_value=1.0,
            value=_get_param("fliplr", 0.5),
            step=0.1,
            help="Horizontal flip probability.",
            key="adv_fliplr",
        )
    with col3:
        params["mosaic"] = st.slider(
            "Mosaic",
            min_value=0.0,
            max_value=1.0,
            value=_get_param("mosaic", 1.0),
            step=0.1,
            help="Mosaic augmentation probability (combines 4 images).",
            key="adv_mosaic",
        )
    with col4:
        params["mixup"] = st.slider(
            "MixUp",
            min_value=0.0,
            max_value=1.0,
            value=_get_param("mixup", 0.1),
            step=0.05,
            help="MixUp augmentation probability (blends 2 images).",
            key="adv_mixup",
        )

    return params


def _render_optimizer_tab() -> Dict[str, Any]:
    """Render Optimizer tab with learning rate and regularization parameters."""
    params = {}

    # Learning Rate
    st.html(f"""
    <div style="
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        opacity: 0.6;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    ">Learning Rate</div>
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        params["optimizer"] = st.selectbox(
            "Optimizer",
            options=["AdamW", "Adam", "SGD", "RMSProp"],
            index=["AdamW", "Adam", "SGD", "RMSProp"].index(
                _get_param("optimizer", "AdamW")
            ),
            help="Optimization algorithm. AdamW recommended for fine-tuning.",
            key="adv_optimizer",
        )
    with col2:
        params["lr0"] = st.number_input(
            "Initial LR",
            min_value=0.0001,
            max_value=0.1,
            value=_get_param("lr0", 0.001),
            step=0.0001,
            format="%.4f",
            help="Initial learning rate. Lower for fine-tuning.",
            key="adv_lr0",
        )
    with col3:
        params["lrf"] = st.number_input(
            "Final LR Scale",
            min_value=0.001,
            max_value=1.0,
            value=_get_param("lrf", 0.01),
            step=0.001,
            format="%.3f",
            help="Final learning rate = lr0 * lrf",
            key="adv_lrf",
        )

    st.markdown("<div style='height: 16px'></div>", unsafe_allow_html=True)

    # Regularization
    st.html(f"""
    <div style="
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        opacity: 0.6;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    ">Regularization</div>
    """)

    col1, col2 = st.columns(2)
    with col1:
        params["momentum"] = st.slider(
            "Momentum",
            min_value=0.8,
            max_value=0.99,
            value=_get_param("momentum", 0.937),
            step=0.001,
            format="%.3f",
            help="Momentum coefficient for SGD/AdamW.",
            key="adv_momentum",
        )
    with col2:
        params["weight_decay"] = st.number_input(
            "Weight Decay",
            min_value=0.0,
            max_value=0.01,
            value=_get_param("weight_decay", 0.0005),
            step=0.0001,
            format="%.4f",
            help="L2 regularization strength.",
            key="adv_weight_decay",
        )

    return params


def _render_performance_tab() -> Dict[str, Any]:
    """Render Performance tab with workers, cache, and AMP settings."""
    params = {}

    st.html(f"""
    <div style="
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        opacity: 0.6;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    ">Performance Settings</div>
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        params["workers"] = st.number_input(
            "Dataloader Workers",
            min_value=0,
            max_value=16,
            value=_get_param("workers", 8),
            step=1,
            help="Number of parallel data loading workers.",
            key="adv_workers",
        )
    with col2:
        params["cache"] = st.checkbox(
            "Cache Images in RAM",
            value=_get_param("cache", True),
            help="Cache images in memory for faster training.",
            key="adv_cache",
        )
    with col3:
        params["amp"] = st.checkbox(
            "Auto Mixed Precision",
            value=_get_param("amp", True),
            help="Use FP16+FP32 mixed precision for faster training.",
            key="adv_amp",
        )

    st.markdown("<div style='height: 16px'></div>", unsafe_allow_html=True)

    # Additional settings
    st.html(f"""
    <div style="
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        opacity: 0.6;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    ">Training Settings</div>
    """)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        params["imgsz"] = st.selectbox(
            "Image Size",
            options=[320, 416, 480, 512, 640, 800, 1024],
            index=[320, 416, 480, 512, 640, 800, 1024].index(
                _get_param("imgsz", 640)
            ),
            help="Input image size. Larger = more accurate but slower.",
            key="adv_imgsz",
        )
    with col2:
        params["patience"] = st.number_input(
            "Early Stop Patience",
            min_value=1,
            max_value=50,
            value=_get_param("patience", 10),
            step=1,
            help="Stop training if no improvement for N epochs.",
            key="adv_patience",
        )
    with col3:
        params["close_mosaic"] = st.number_input(
            "Close Mosaic",
            min_value=0,
            max_value=30,
            value=_get_param("close_mosaic", 10),
            step=1,
            help="Disable mosaic for last N epochs.",
            key="adv_close_mosaic",
        )
    with col4:
        params["freeze"] = st.number_input(
            "Freeze Layers",
            min_value=0,
            max_value=20,
            value=_get_param("freeze", 10),
            step=1,
            help="Freeze first N backbone layers. Prevents overfitting on small datasets.",
            key="adv_freeze",
        )

    return params


def _render_checkpoint_tab() -> Dict[str, Any]:
    """Render Checkpoint tab with save settings."""
    params = {}

    st.html(f"""
    <div style="
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        opacity: 0.6;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    ">Checkpoint Settings</div>
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        params["save"] = st.checkbox(
            "Save Checkpoints",
            value=_get_param("save", True),
            help="Save model checkpoints during training.",
            key="adv_save",
        )
    with col2:
        params["save_period"] = st.number_input(
            "Save Every N Epochs",
            min_value=1,
            max_value=20,
            value=_get_param("save_period", 5),
            step=1,
            disabled=not _get_param("save", True),
            help="Save checkpoint every N epochs.",
            key="adv_save_period",
        )
    with col3:
        params["exist_ok"] = st.checkbox(
            "Overwrite Existing",
            value=_get_param("exist_ok", True),
            help="Allow overwriting existing run directory.",
            key="adv_exist_ok",
        )

    return params


# =============================================================================
# Main Entry Point
# =============================================================================

def render_advanced_parameters_section(
    auto_scale: bool,
    gpu_tier: str,
) -> Dict[str, Any]:
    """
    Render the advanced parameters section.

    Args:
        auto_scale: Whether GPU auto-scaling is enabled
        gpu_tier: Current GPU tier (low/medium/high/workstation)

    Returns:
        Dictionary of all advanced parameters from UI
    """
    # Ensure session state is initialized
    _ensure_session_state()

    # Section header
    st.html(f"""
    <div style="
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.9rem;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    ">
        <span>{"" if "settings" not in ICONS else ICONS.get("settings", "")}</span>
        <span>Advanced Parameters</span>
    </div>
    """)

    with st.expander("Configure detailed training parameters", expanded=False):
        # Note about parameter priority
        if auto_scale:
            st.info(
                "Advanced parameters will override GPU auto-scaling settings."
            )

        # Preset selector
        _render_preset_selector()

        st.markdown("<div style='height: 16px'></div>", unsafe_allow_html=True)

        # Parameter tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "Data Augmentation",
            "Optimizer",
            "Performance",
            "Checkpoint"
        ])

        with tab1:
            aug_params = _render_augmentation_tab()

        with tab2:
            opt_params = _render_optimizer_tab()

        with tab3:
            perf_params = _render_performance_tab()

        with tab4:
            ckpt_params = _render_checkpoint_tab()

    # Combine all parameters
    all_params = {
        **aug_params,
        **opt_params,
        **perf_params,
        **ckpt_params,
    }

    return all_params
