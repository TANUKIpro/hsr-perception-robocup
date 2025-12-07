"""
Training Page Styles - Mission Control Aesthetic

Industrial tech design system for the training dashboard.
Dark theme with cyan/teal accents for a control room feel.
"""

import streamlit as st
from typing import Optional

# =============================================================================
# Color Palette - Mission Control Theme
# =============================================================================

COLORS = {
    # Base colors
    "background": "#0A0E14",
    "surface": "#12171E",
    "surface_elevated": "#1A2029",
    "surface_hover": "#232B36",

    # Text
    "text_primary": "#E6EDF3",
    "text_secondary": "#8B949E",
    "text_muted": "#6E7681",

    # Accent - Cyan/Teal gradient
    "accent_primary": "#00D4AA",
    "accent_secondary": "#00B4D8",
    "accent_gradient": "linear-gradient(135deg, #00D4AA 0%, #00B4D8 100%)",

    # Status colors
    "success": "#3FB950",
    "success_bg": "rgba(63, 185, 80, 0.15)",
    "warning": "#D29922",
    "warning_bg": "rgba(210, 153, 34, 0.15)",
    "error": "#F85149",
    "error_bg": "rgba(248, 81, 73, 0.15)",
    "info": "#58A6FF",
    "info_bg": "rgba(88, 166, 255, 0.15)",

    # GPU Tier colors
    "tier_low": "#6E7681",
    "tier_medium": "#58A6FF",
    "tier_high": "#A371F7",
    "tier_workstation": "#F0883E",

    # Progress
    "progress_bg": "#21262D",
    "progress_track": "#30363D",
}

# =============================================================================
# Icon System
# =============================================================================

ICONS = {
    # Task status
    "pending": "◐",
    "running": "◉",
    "completed": "✓",
    "failed": "✕",
    "cancelled": "◌",

    # GPU tiers
    "gpu_low": "▪",
    "gpu_medium": "▫",
    "gpu_high": "◆",
    "gpu_workstation": "◈",

    # UI elements
    "tensorboard": "◫",
    "model": "◎",
    "dataset": "⊞",
    "accuracy": "◉",
    "time": "◷",
    "memory": "▥",
    "speed": "⚡",
    "expand": "▾",
    "collapse": "▴",
    "external": "↗",
    "refresh": "↻",
}

# =============================================================================
# Typography
# =============================================================================

FONTS = {
    "display": "'JetBrains Mono', 'SF Mono', 'Consolas', monospace",
    "body": "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
    "mono": "'JetBrains Mono', 'SF Mono', monospace",
}

# =============================================================================
# CSS Injection
# =============================================================================

def inject_training_styles():
    """Inject custom CSS for training page - theme-aware version."""

    css = f"""
    <style>
    /* Import fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    /* =========================================
       Custom Component Styles (Theme-aware)
       ========================================= */

    /* Mission Control Header */
    .mc-header {{
        font-family: {FONTS["display"]};
        font-size: 1.75rem;
        font-weight: 600;
        letter-spacing: -0.02em;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 12px;
    }}

    .mc-header::before {{
        content: '';
        display: inline-block;
        width: 4px;
        height: 24px;
        background: {COLORS["accent_gradient"]};
        border-radius: 2px;
    }}

    /* Status Badge */
    .mc-status-badge {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 4px;
        font-family: {FONTS["mono"]};
        font-size: 0.75rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}

    .mc-status-badge.running {{
        background: {COLORS["info_bg"]};
        color: {COLORS["info"]};
    }}

    .mc-status-badge.running::before {{
        content: '';
        width: 6px;
        height: 6px;
        background: {COLORS["info"]};
        border-radius: 50%;
        animation: pulse-dot 1.5s ease-in-out infinite;
    }}

    .mc-status-badge.completed {{
        background: {COLORS["success_bg"]};
        color: {COLORS["success"]};
    }}

    .mc-status-badge.failed {{
        background: {COLORS["error_bg"]};
        color: {COLORS["error"]};
    }}

    @keyframes pulse-dot {{
        0%, 100% {{ opacity: 1; transform: scale(1); }}
        50% {{ opacity: 0.5; transform: scale(0.8); }}
    }}

    /* =========================================
       Metric Card
       ========================================= */

    .mc-metric-card {{
        border-radius: 8px;
        padding: 20px;
        position: relative;
        overflow: hidden;
    }}

    .mc-metric-card::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: {COLORS["accent_gradient"]};
        opacity: 0;
        transition: opacity 0.2s ease;
    }}

    .mc-metric-card:hover::before {{
        opacity: 1;
    }}

    .mc-metric-label {{
        font-family: {FONTS["mono"]};
        font-size: 0.7rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 8px;
        opacity: 0.7;
    }}

    .mc-metric-value {{
        font-family: {FONTS["display"]};
        font-size: 1.75rem;
        font-weight: 600;
        line-height: 1.1;
    }}

    .mc-metric-value.accent {{
        background: {COLORS["accent_gradient"]};
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }}

    .mc-metric-value.success {{
        color: {COLORS["success"]};
    }}

    .mc-metric-value.warning {{
        color: {COLORS["warning"]};
    }}

    .mc-metric-value.error {{
        color: {COLORS["error"]};
    }}

    .mc-metric-subtitle {{
        font-family: {FONTS["body"]};
        font-size: 0.75rem;
        margin-top: 4px;
        opacity: 0.7;
    }}

    /* =========================================
       Circular Progress
       ========================================= */

    .mc-circular-progress {{
        position: relative;
        display: flex;
        align-items: center;
        justify-content: center;
    }}

    .mc-circular-progress svg {{
        transform: rotate(-90deg);
    }}

    .mc-circular-progress .progress-text {{
        position: absolute;
        font-family: {FONTS["display"]};
        font-size: 1.5rem;
        font-weight: 700;
    }}

    .mc-circular-progress .progress-label {{
        position: absolute;
        bottom: -24px;
        font-family: {FONTS["mono"]};
        font-size: 0.65rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        opacity: 0.7;
    }}

    /* =========================================
       GPU Status Card
       ========================================= */

    .mc-gpu-card {{
        border-radius: 12px;
        padding: 24px;
        position: relative;
    }}

    .mc-gpu-card.tier-low {{ border-left: 3px solid {COLORS["tier_low"]}; }}
    .mc-gpu-card.tier-medium {{ border-left: 3px solid {COLORS["tier_medium"]}; }}
    .mc-gpu-card.tier-high {{ border-left: 3px solid {COLORS["tier_high"]}; }}
    .mc-gpu-card.tier-workstation {{ border-left: 3px solid {COLORS["tier_workstation"]}; }}

    .mc-gpu-name {{
        font-family: {FONTS["display"]};
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 4px;
    }}

    .mc-gpu-memory {{
        font-family: {FONTS["mono"]};
        font-size: 0.85rem;
        opacity: 0.7;
    }}

    .mc-gpu-tier-badge {{
        display: inline-block;
        padding: 4px 10px;
        border-radius: 4px;
        font-family: {FONTS["mono"]};
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}

    .mc-gpu-tier-badge.low {{
        background: rgba(110, 118, 129, 0.2);
        color: {COLORS["tier_low"]};
    }}

    .mc-gpu-tier-badge.medium {{
        background: rgba(88, 166, 255, 0.15);
        color: {COLORS["tier_medium"]};
    }}

    .mc-gpu-tier-badge.high {{
        background: rgba(163, 113, 247, 0.15);
        color: {COLORS["tier_high"]};
    }}

    .mc-gpu-tier-badge.workstation {{
        background: rgba(240, 136, 62, 0.15);
        color: {COLORS["tier_workstation"]};
    }}

    /* =========================================
       Config Summary Card
       ========================================= */

    .mc-config-card {{
        background: linear-gradient(135deg,
            rgba(0, 212, 170, 0.08) 0%,
            rgba(0, 180, 216, 0.08) 100%);
        border: 1px solid rgba(0, 212, 170, 0.2);
        border-radius: 12px;
        padding: 24px;
        position: relative;
        overflow: hidden;
    }}

    .mc-config-card::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: {COLORS["accent_gradient"]};
    }}

    .mc-config-title {{
        font-family: {FONTS["display"]};
        font-size: 0.9rem;
        font-weight: 600;
        color: {COLORS["accent_primary"]};
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 16px;
    }}

    .mc-config-grid {{
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 16px;
    }}

    .mc-config-item {{
        display: flex;
        flex-direction: column;
        gap: 2px;
    }}

    .mc-config-item-label {{
        font-family: {FONTS["body"]};
        font-size: 0.75rem;
        opacity: 0.7;
    }}

    .mc-config-item-value {{
        font-family: {FONTS["display"]};
        font-size: 1rem;
        font-weight: 600;
    }}

    .mc-config-divider {{
        grid-column: 1 / -1;
        height: 1px;
        background: rgba(0, 212, 170, 0.2);
        margin: 8px 0;
    }}

    .mc-config-footer {{
        grid-column: 1 / -1;
        display: flex;
        justify-content: space-between;
        align-items: flex-end;
        margin-top: 8px;
    }}

    .mc-config-eta {{
        font-family: {FONTS["display"]};
        font-size: 1.5rem;
        font-weight: 700;
        background: {COLORS["accent_gradient"]};
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }}

    /* =========================================
       TensorBoard Panel
       ========================================= */

    .mc-tensorboard-panel {{
        border-radius: 12px;
        overflow: hidden;
    }}

    .mc-tensorboard-header {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 16px 20px;
    }}

    .mc-tensorboard-title {{
        display: flex;
        align-items: center;
        gap: 8px;
        font-family: {FONTS["display"]};
        font-size: 0.9rem;
        font-weight: 600;
    }}

    .mc-tensorboard-title .icon {{
        color: {COLORS["accent_primary"]};
    }}

    .mc-tensorboard-actions {{
        display: flex;
        gap: 8px;
    }}

    .mc-btn {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 8px 14px;
        border-radius: 6px;
        font-family: {FONTS["body"]};
        font-size: 0.8rem;
        font-weight: 500;
        text-decoration: none;
        cursor: pointer;
        transition: all 0.2s ease;
        border: none;
    }}

    .mc-btn-primary {{
        background: {COLORS["accent_gradient"]};
        color: white;
    }}

    .mc-btn-primary:hover {{
        filter: brightness(1.1);
        transform: translateY(-1px);
    }}

    .mc-tensorboard-placeholder {{
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 60px 20px;
        text-align: center;
        opacity: 0.7;
    }}

    .mc-tensorboard-placeholder .icon {{
        font-size: 2rem;
        margin-bottom: 12px;
    }}

    /* Quick links */
    .mc-quick-links {{
        display: flex;
        gap: 8px;
        margin-top: 16px;
    }}

    .mc-quick-link {{
        padding: 6px 12px;
        border-radius: 4px;
        font-family: {FONTS["mono"]};
        font-size: 0.75rem;
        text-decoration: none;
        transition: all 0.2s ease;
    }}

    .mc-quick-link.scalars {{
        background: rgba(88, 166, 255, 0.15);
        color: {COLORS["info"]};
    }}

    .mc-quick-link.images {{
        background: rgba(163, 113, 247, 0.15);
        color: {COLORS["tier_high"]};
    }}

    .mc-quick-link.graphs {{
        background: rgba(63, 185, 80, 0.15);
        color: {COLORS["success"]};
    }}

    .mc-quick-link:hover {{
        filter: brightness(1.2);
        transform: translateY(-1px);
    }}

    /* =========================================
       Progress Bar
       ========================================= */

    .mc-progress-container {{
        border-radius: 4px;
        height: 8px;
        overflow: hidden;
        position: relative;
        opacity: 0.3;
    }}

    .mc-progress-bar {{
        height: 100%;
        background: {COLORS["accent_gradient"]};
        border-radius: 4px;
        transition: width 0.5s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
    }}

    .mc-progress-bar.animated::after {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(
            90deg,
            transparent,
            rgba(255, 255, 255, 0.2),
            transparent
        );
        animation: shimmer 2s infinite;
    }}

    @keyframes shimmer {{
        0% {{ transform: translateX(-100%); }}
        100% {{ transform: translateX(100%); }}
    }}

    .mc-progress-text {{
        font-family: {FONTS["mono"]};
        font-size: 0.75rem;
        margin-top: 6px;
        display: flex;
        justify-content: space-between;
        opacity: 0.7;
    }}

    /* =========================================
       Training History / Task List
       ========================================= */

    .mc-task-item {{
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 8px;
        transition: all 0.2s ease;
    }}

    .mc-task-item:hover {{
        transform: translateX(4px);
    }}

    .mc-task-header {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 12px;
    }}

    .mc-task-id {{
        font-family: {FONTS["mono"]};
        font-size: 0.8rem;
        opacity: 0.7;
    }}

    .mc-task-time {{
        font-family: {FONTS["mono"]};
        font-size: 0.75rem;
        opacity: 0.5;
    }}

    /* =========================================
       Validation Messages
       ========================================= */

    .mc-validation {{
        display: flex;
        align-items: flex-start;
        gap: 10px;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 8px;
        font-family: {FONTS["body"]};
        font-size: 0.85rem;
    }}

    .mc-validation.warning {{
        background: {COLORS["warning_bg"]};
        color: {COLORS["warning"]};
    }}

    .mc-validation.error {{
        background: {COLORS["error_bg"]};
        color: {COLORS["error"]};
    }}

    .mc-validation .icon {{
        font-size: 1rem;
        flex-shrink: 0;
    }}

    /* =========================================
       Animations
       ========================================= */

    @keyframes fade-in {{
        from {{ opacity: 0; transform: translateY(10px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

    @keyframes slide-in {{
        from {{ opacity: 0; transform: translateX(-20px); }}
        to {{ opacity: 1; transform: translateX(0); }}
    }}

    .mc-animate-fade {{
        animation: fade-in 0.4s ease-out;
    }}

    .mc-animate-slide {{
        animation: slide-in 0.3s ease-out;
    }}

    /* Staggered animation delays */
    .mc-stagger-1 {{ animation-delay: 0.05s; }}
    .mc-stagger-2 {{ animation-delay: 0.1s; }}
    .mc-stagger-3 {{ animation-delay: 0.15s; }}
    .mc-stagger-4 {{ animation-delay: 0.2s; }}

    </style>
    """

    st.markdown(css, unsafe_allow_html=True)


def render_header(title: str, subtitle: Optional[str] = None):
    """Render Mission Control style header."""
    html = f'<div class="mc-header">{title}</div>'
    if subtitle:
        html += f'<p style="font-size: 0.9rem; margin-top: -8px; opacity: 0.7;">{subtitle}</p>'
    st.markdown(html, unsafe_allow_html=True)


def render_status_badge(status: str, text: Optional[str] = None):
    """Render status badge with appropriate styling."""
    display_text = text or status.upper()
    icon = ICONS.get(status.lower(), "◉")

    html = f'''
    <span class="mc-status-badge {status.lower()}">
        {icon} {display_text}
    </span>
    '''
    return html


def get_tier_class(tier: str) -> str:
    """Get CSS class for GPU tier."""
    return tier.lower() if tier.lower() in ["low", "medium", "high", "workstation"] else "medium"
