"""
Dataset Status Components

Visual components for displaying class-by-class data status
and dataset preparation UI in the Annotation page.
"""

import streamlit as st
from typing import List, Optional
from datetime import datetime

from .training_styles import COLORS, ICONS

# Status icons and colors
STATUS_CONFIG = {
    "ready": {
        "icon": "✓",
        "color": COLORS["success"],
        "bg": COLORS["success_bg"],
        "label": "Ready",
    },
    "partial": {
        "icon": "◐",
        "color": COLORS["warning"],
        "bg": COLORS["warning_bg"],
        "label": "Partial",
    },
    "insufficient": {
        "icon": "⚠",
        "color": COLORS["warning"],
        "bg": COLORS["warning_bg"],
        "label": "Insufficient",
    },
    "no_data": {
        "icon": "✕",
        "color": COLORS["error"],
        "bg": COLORS["error_bg"],
        "label": "No Data",
    },
}


def render_class_status_card(class_info):
    """
    Render a single class status card using Streamlit components.

    Args:
        class_info: ClassInfo object from DatasetPreparer
    """
    status = class_info.status
    config = STATUS_CONFIG.get(status, STATUS_CONFIG["no_data"])
    match_percent = int(class_info.match_ratio * 100)

    with st.container(border=True):
        # Header: class name and status
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**{class_info.name}**")
        with col2:
            st.markdown(f"{config['icon']} {config['label']}")

        # Stats
        st.caption(f"📷 {class_info.image_count} images | 🏷️ {class_info.label_count} labels | ⚡ {match_percent}% matched")

        # Progress bar
        st.progress(class_info.match_ratio)


def render_class_status_grid(classes: List, columns: int = 2):
    """
    Render a grid of class status cards.

    Args:
        classes: List of ClassInfo objects
        columns: Number of columns in grid
    """
    # Extract accent colors only (these are intentional)
    info_color = COLORS["info"]
    accent_primary = COLORS["accent_primary"]
    success = COLORS["success"]

    if not classes:
        html = """
        <div style="
            border: 1px dashed currentColor;
            border-radius: 12px;
            padding: 40px;
            text-align: center;
            opacity: 0.6;
        ">
            <div style="
                font-size: 2rem;
                margin-bottom: 12px;
            ">📁</div>
            <div style="
                font-family: 'Inter', sans-serif;
                font-size: 0.9rem;
                opacity: 0.8;
            ">No classes found</div>
            <div style="
                font-family: 'Inter', sans-serif;
                font-size: 0.8rem;
                opacity: 0.5;
                margin-top: 4px;
            ">Capture some images first using the app</div>
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)
        return

    # Summary stats
    total_images = sum(c.image_count for c in classes)
    total_matched = sum(c.matched_count for c in classes)
    ready_count = sum(1 for c in classes if c.is_ready)
    num_classes = len(classes)

    html = f"""
    <div style="
        display: flex;
        gap: 16px;
        margin-bottom: 16px;
        padding: 12px 16px;
        border-radius: 8px;
    ">
        <div style="text-align: center; flex: 1;">
            <div style="
                font-family: 'JetBrains Mono', monospace;
                font-size: 1.25rem;
                font-weight: 600;
            ">{num_classes}</div>
            <div style="font-size: 0.7rem; opacity: 0.5;">Classes</div>
        </div>
        <div style="text-align: center; flex: 1;">
            <div style="
                font-family: 'JetBrains Mono', monospace;
                font-size: 1.25rem;
                font-weight: 600;
                color: {info_color};
            ">{total_images}</div>
            <div style="font-size: 0.7rem; opacity: 0.5;">Images</div>
        </div>
        <div style="text-align: center; flex: 1;">
            <div style="
                font-family: 'JetBrains Mono', monospace;
                font-size: 1.25rem;
                font-weight: 600;
                color: {accent_primary};
            ">{total_matched}</div>
            <div style="font-size: 0.7rem; opacity: 0.5;">Matched</div>
        </div>
        <div style="text-align: center; flex: 1;">
            <div style="
                font-family: 'JetBrains Mono', monospace;
                font-size: 1.25rem;
                font-weight: 600;
                color: {success};
            ">{ready_count}/{num_classes}</div>
            <div style="font-size: 0.7rem; opacity: 0.5;">Ready</div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

    # Render cards in columns
    cols = st.columns(columns)
    for i, cls in enumerate(classes):
        with cols[i % columns]:
            render_class_status_card(cls)


def render_dataset_preparation_panel(
    classes: List,
    on_generate: callable = None,
):
    """
    Render dataset preparation panel with class selection and options.

    Args:
        classes: List of ClassInfo objects
        on_generate: Callback when generate button is clicked
    """
    ready_classes = [c for c in classes if c.is_ready]

    if not ready_classes:
        st.warning("No classes are ready for training. Annotate at least 10 samples per class.")
        return None

    # Extract accent colors only
    info_color = COLORS["info"]
    accent_primary = COLORS["accent_primary"]

    html = """
    <div style="
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.9rem;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    ">
        <span>📦</span>
        <span>Generate Training Dataset</span>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

    # Class selection
    html2 = """
    <div style="
        font-size: 0.8rem;
        opacity: 0.7;
        margin-bottom: 8px;
    ">Select classes to include:</div>
    """
    st.markdown(html2, unsafe_allow_html=True)

    selected_classes = []
    cols = st.columns(min(len(ready_classes), 4))

    for i, cls in enumerate(ready_classes):
        with cols[i % len(cols)]:
            if st.checkbox(
                f"{cls.name} ({cls.matched_count})",
                value=True,
                key=f"class_select_{cls.name}"
            ):
                selected_classes.append(cls.name)

    st.markdown("<div style='height: 16px'></div>", unsafe_allow_html=True)

    # Options
    col1, col2 = st.columns(2)

    with col1:
        dataset_name = st.text_input(
            "Dataset Name",
            value=f"dataset_{datetime.now().strftime('%Y%m%d_%H%M')}",
            help="Name for the output dataset directory"
        )

    with col2:
        val_ratio = st.slider(
            "Validation Ratio",
            min_value=0.1,
            max_value=0.4,
            value=0.2,
            step=0.05,
            help="Proportion of data to use for validation"
        )

    st.markdown("<div style='height: 16px'></div>", unsafe_allow_html=True)

    # Preview
    if selected_classes:
        total_samples = sum(
            c.matched_count for c in ready_classes if c.name in selected_classes
        )
        train_count = int(total_samples * (1 - val_ratio))
        val_count = total_samples - train_count
        num_selected = len(selected_classes)

        html3 = f"""
        <div style="
            border-radius: 8px;
            padding: 12px 16px;
            margin-bottom: 16px;
        ">
            <div style="
                font-family: 'JetBrains Mono', monospace;
                font-size: 0.75rem;
                opacity: 0.5;
                text-transform: uppercase;
                letter-spacing: 0.1em;
                margin-bottom: 8px;
            ">Preview</div>
            <div style="display: flex; gap: 24px;">
                <div>
                    <div style="font-size: 0.7rem; opacity: 0.5;">Classes</div>
                    <div style="
                        font-family: 'JetBrains Mono', monospace;
                    ">{num_selected}</div>
                </div>
                <div>
                    <div style="font-size: 0.7rem; opacity: 0.5;">Train</div>
                    <div style="
                        font-family: 'JetBrains Mono', monospace;
                        color: {info_color};
                    ">{train_count}</div>
                </div>
                <div>
                    <div style="font-size: 0.7rem; opacity: 0.5;">Val</div>
                    <div style="
                        font-family: 'JetBrains Mono', monospace;
                        color: {accent_primary};
                    ">{val_count}</div>
                </div>
                <div>
                    <div style="font-size: 0.7rem; opacity: 0.5;">Total</div>
                    <div style="
                        font-family: 'JetBrains Mono', monospace;
                    ">{total_samples}</div>
                </div>
            </div>
        </div>
        """
        st.markdown(html3, unsafe_allow_html=True)

    # Generate button
    if st.button(
        "🚀 Generate Dataset",
        type="primary",
        use_container_width=True,
        disabled=len(selected_classes) == 0
    ):
        return {
            "classes": selected_classes,
            "dataset_name": dataset_name,
            "val_ratio": val_ratio,
        }

    return None


def render_dataset_result(result, dataset_name: str):
    """
    Render dataset generation result.

    Args:
        result: DatasetResult from DatasetPreparer
        dataset_name: Name of the generated dataset
    """
    # Extract accent colors only
    success_color = COLORS["success"]
    success_bg = COLORS["success_bg"]
    info_color = COLORS["info"]
    accent_primary = COLORS["accent_primary"]

    if result.success:
        num_classes = len(result.class_names)
        train_count = result.train_count
        val_count = result.val_count
        total_count = train_count + val_count
        class_names_str = ', '.join(result.class_names)

        html = f"""
        <div style="
            background: {success_bg};
            border: 1px solid rgba(63, 185, 80, 0.3);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 16px;
        ">
            <div style="
                display: flex;
                align-items: center;
                gap: 12px;
                margin-bottom: 12px;
            ">
                <span style="
                    font-size: 1.5rem;
                    color: {success_color};
                ">✓</span>
                <span style="
                    font-family: 'JetBrains Mono', monospace;
                    font-size: 1rem;
                    font-weight: 600;
                    color: {success_color};
                ">Dataset Created Successfully!</span>
            </div>

            <div style="
                display: grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 16px;
                margin-top: 16px;
            ">
                <div style="text-align: center;">
                    <div style="
                        font-family: 'JetBrains Mono', monospace;
                        font-size: 1.25rem;
                        font-weight: 600;
                    ">{num_classes}</div>
                    <div style="font-size: 0.7rem; opacity: 0.5;">Classes</div>
                </div>
                <div style="text-align: center;">
                    <div style="
                        font-family: 'JetBrains Mono', monospace;
                        font-size: 1.25rem;
                        font-weight: 600;
                        color: {info_color};
                    ">{train_count}</div>
                    <div style="font-size: 0.7rem; opacity: 0.5;">Train</div>
                </div>
                <div style="text-align: center;">
                    <div style="
                        font-family: 'JetBrains Mono', monospace;
                        font-size: 1.25rem;
                        font-weight: 600;
                        color: {accent_primary};
                    ">{val_count}</div>
                    <div style="font-size: 0.7rem; opacity: 0.5;">Val</div>
                </div>
                <div style="text-align: center;">
                    <div style="
                        font-family: 'JetBrains Mono', monospace;
                        font-size: 1.25rem;
                        font-weight: 600;
                    ">{total_count}</div>
                    <div style="font-size: 0.7rem; opacity: 0.5;">Total</div>
                </div>
            </div>

            <div style="
                margin-top: 16px;
                padding-top: 12px;
                border-top: 1px solid rgba(63, 185, 80, 0.2);
            ">
                <div style="
                    font-size: 0.8rem;
                    opacity: 0.7;
                    margin-bottom: 4px;
                ">Classes:</div>
                <div style="
                    font-family: 'JetBrains Mono', monospace;
                    font-size: 0.85rem;
                ">{class_names_str}</div>
            </div>
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)

        # Show data.yaml preview
        if result.output_dir:
            yaml_path = result.output_dir / "data.yaml"
            if yaml_path.exists():
                with st.expander("📄 data.yaml", expanded=False):
                    st.code(yaml_path.read_text(), language="yaml")

    else:
        st.error(f"Dataset generation failed: {result.error_message}")
