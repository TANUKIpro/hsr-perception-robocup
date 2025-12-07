"""
Training Charts - Plotly-based visualization components

Real-time training metrics visualization with Mission Control aesthetic.
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Optional, Any

from .training_styles import COLORS

# Plotly theme configuration matching Mission Control aesthetic
PLOTLY_THEME = {
    "paper_bgcolor": COLORS["surface"],
    "plot_bgcolor": COLORS["surface"],
    "font_family": "'JetBrains Mono', 'SF Mono', monospace",
    "font_color": COLORS["text_secondary"],
    "gridcolor": COLORS["surface_hover"],
    "linecolor": COLORS["surface_hover"],
}


def render_training_chart(
    training_history: List[Dict[str, Any]],
    target_map50: float = 0.85,
    height: int = 350,
    show_title: bool = True,
):
    """
    Render live training metrics chart using Plotly.

    Shows:
    - mAP50 vs epoch (with target line)
    - Loss vs epoch (secondary y-axis)

    Args:
        training_history: List of dicts with 'epoch', 'mAP50', 'loss' keys
        target_map50: Target mAP50 value for reference line
        height: Chart height in pixels
        show_title: Whether to show chart title
    """
    if not training_history:
        _render_empty_chart_placeholder()
        return

    # Extract data
    epochs = [h.get("epoch", i + 1) for i, h in enumerate(training_history)]
    map50_values = [h.get("mAP50", 0) for h in training_history]
    loss_values = [h.get("loss", 0) for h in training_history]

    # Create figure with secondary y-axis
    fig = make_subplots(
        rows=1, cols=1,
        specs=[[{"secondary_y": True}]]
    )

    # mAP50 line with gradient fill
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=map50_values,
            name="mAP@50",
            mode="lines+markers",
            line=dict(
                color=COLORS["accent_primary"],
                width=2.5,
            ),
            marker=dict(
                size=6,
                color=COLORS["accent_primary"],
                line=dict(width=1, color=COLORS["surface"]),
            ),
            fill="tozeroy",
            fillcolor="rgba(0, 212, 170, 0.1)",
            hovertemplate="Epoch %{x}<br>mAP@50: %{y:.3f}<extra></extra>",
        ),
        secondary_y=False
    )

    # Target line
    if target_map50 > 0:
        fig.add_hline(
            y=target_map50,
            line_dash="dash",
            line_color=COLORS["success"],
            line_width=1.5,
            annotation=dict(
                text=f"Target {target_map50:.0%}",
                font=dict(
                    size=10,
                    color=COLORS["success"],
                    family="JetBrains Mono",
                ),
                showarrow=False,
                xanchor="right",
                x=1,
                yshift=10,
            ),
            secondary_y=False
        )

    # Loss line
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=loss_values,
            name="Loss",
            mode="lines",
            line=dict(
                color=COLORS["error"],
                width=2,
                dash="dot",
            ),
            hovertemplate="Epoch %{x}<br>Loss: %{y:.4f}<extra></extra>",
        ),
        secondary_y=True
    )

    # Update layout with Mission Control theme
    fig.update_layout(
        title=dict(
            text="Training Progress" if show_title else None,
            font=dict(
                size=14,
                color=COLORS["text_primary"],
                family="JetBrains Mono",
            ),
            x=0,
            xanchor="left",
        ),
        height=height,
        margin=dict(l=10, r=10, t=40 if show_title else 10, b=10),
        paper_bgcolor=PLOTLY_THEME["paper_bgcolor"],
        plot_bgcolor=PLOTLY_THEME["plot_bgcolor"],
        font=dict(
            family=PLOTLY_THEME["font_family"],
            size=11,
            color=PLOTLY_THEME["font_color"],
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(size=10),
            bgcolor="rgba(0,0,0,0)",
        ),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor=COLORS["surface_elevated"],
            font_size=11,
            font_family="JetBrains Mono",
        ),
    )

    # Update axes
    fig.update_xaxes(
        title=dict(
            text="Epoch",
            font=dict(size=10, color=COLORS["text_muted"]),
        ),
        tickfont=dict(size=10),
        gridcolor=PLOTLY_THEME["gridcolor"],
        linecolor=PLOTLY_THEME["linecolor"],
        zeroline=False,
        showgrid=True,
        gridwidth=1,
        griddash="dot",
    )

    fig.update_yaxes(
        title=dict(
            text="mAP@50",
            font=dict(size=10, color=COLORS["accent_primary"]),
        ),
        tickfont=dict(size=10, color=COLORS["accent_primary"]),
        range=[0, 1],
        tickformat=".0%",
        gridcolor=PLOTLY_THEME["gridcolor"],
        linecolor=PLOTLY_THEME["linecolor"],
        zeroline=False,
        showgrid=True,
        gridwidth=1,
        griddash="dot",
        secondary_y=False,
    )

    fig.update_yaxes(
        title=dict(
            text="Loss",
            font=dict(size=10, color=COLORS["error"]),
        ),
        tickfont=dict(size=10, color=COLORS["error"]),
        gridcolor="rgba(0,0,0,0)",  # Hide grid for secondary axis
        linecolor=PLOTLY_THEME["linecolor"],
        zeroline=False,
        showgrid=False,
        secondary_y=True,
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_epoch_metrics_chart(
    training_history: List[Dict[str, Any]],
    metrics: List[str] = None,
    height: int = 280,
):
    """
    Render multi-metric chart for detailed epoch analysis.

    Args:
        training_history: List of dicts with metric values
        metrics: List of metric keys to display
        height: Chart height in pixels
    """
    if not training_history:
        _render_empty_chart_placeholder()
        return

    metrics = metrics or ["precision", "recall", "mAP50", "mAP50-95"]

    # Color palette for metrics
    metric_colors = {
        "precision": COLORS["info"],
        "recall": COLORS["tier_high"],
        "mAP50": COLORS["accent_primary"],
        "mAP50-95": COLORS["accent_secondary"],
    }

    epochs = [h.get("epoch", i + 1) for i, h in enumerate(training_history)]

    fig = go.Figure()

    for metric in metrics:
        values = [h.get(metric, 0) for h in training_history]
        if any(v > 0 for v in values):  # Only show metrics with data
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=values,
                    name=metric,
                    mode="lines",
                    line=dict(
                        color=metric_colors.get(metric, COLORS["text_secondary"]),
                        width=2,
                    ),
                    hovertemplate=f"{metric}: %{{y:.3f}}<extra></extra>",
                )
            )

    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor=PLOTLY_THEME["paper_bgcolor"],
        plot_bgcolor=PLOTLY_THEME["plot_bgcolor"],
        font=dict(
            family=PLOTLY_THEME["font_family"],
            size=10,
            color=PLOTLY_THEME["font_color"],
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(size=9),
            bgcolor="rgba(0,0,0,0)",
        ),
        hovermode="x unified",
    )

    fig.update_xaxes(
        gridcolor=PLOTLY_THEME["gridcolor"],
        linecolor=PLOTLY_THEME["linecolor"],
        zeroline=False,
        griddash="dot",
    )

    fig.update_yaxes(
        range=[0, 1],
        tickformat=".0%",
        gridcolor=PLOTLY_THEME["gridcolor"],
        linecolor=PLOTLY_THEME["linecolor"],
        zeroline=False,
        griddash="dot",
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_loss_breakdown_chart(
    training_history: List[Dict[str, Any]],
    height: int = 200,
):
    """
    Render loss breakdown (box, cls, dfl) as stacked area chart.

    Args:
        training_history: List of dicts with loss values
        height: Chart height in pixels
    """
    if not training_history:
        return

    epochs = [h.get("epoch", i + 1) for i, h in enumerate(training_history)]

    # Loss components (if available)
    loss_types = [
        ("box_loss", "Box Loss", COLORS["info"]),
        ("cls_loss", "Class Loss", COLORS["accent_primary"]),
        ("dfl_loss", "DFL Loss", COLORS["tier_high"]),
    ]

    fig = go.Figure()

    for loss_key, loss_name, color in loss_types:
        values = [h.get(loss_key, 0) for h in training_history]
        if any(v > 0 for v in values):
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=values,
                    name=loss_name,
                    mode="lines",
                    stackgroup="one",
                    line=dict(width=0.5, color=color),
                    fillcolor=color.replace(")", ", 0.3)").replace("rgb", "rgba"),
                )
            )

    fig.update_layout(
        height=height,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor=PLOTLY_THEME["paper_bgcolor"],
        plot_bgcolor=PLOTLY_THEME["plot_bgcolor"],
        font=dict(
            family=PLOTLY_THEME["font_family"],
            size=9,
            color=PLOTLY_THEME["font_color"],
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1,
            font=dict(size=8),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=True,
    )

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _render_empty_chart_placeholder():
    """Render placeholder when no training data is available."""
    st.markdown(f"""
    <div style="
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 60px 20px;
        background: {COLORS["surface"]};
        border: 1px dashed {COLORS["surface_hover"]};
        border-radius: 8px;
        text-align: center;
    ">
        <div style="
            font-size: 2rem;
            color: {COLORS["text_muted"]};
            margin-bottom: 12px;
        ">◐</div>
        <div style="
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            color: {COLORS["text_secondary"]};
        ">Waiting for training data...</div>
        <div style="
            font-family: 'Inter', sans-serif;
            font-size: 0.75rem;
            color: {COLORS["text_muted"]};
            margin-top: 4px;
        ">Chart will update as epochs complete</div>
    </div>
    """, unsafe_allow_html=True)
