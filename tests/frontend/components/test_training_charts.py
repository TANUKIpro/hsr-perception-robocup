"""
Training Charts テスト

app/components/training_charts.py のテスト
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# アプリパスを追加（インポート前に）
APP_PATH = str(Path(__file__).parent.parent.parent.parent / "app")
sys.path.insert(0, APP_PATH)

# Streamlitをモック（インポート前に）
mock_st = MagicMock()
mock_st.html = MagicMock()
mock_st.plotly_chart = MagicMock()
sys.modules['streamlit'] = mock_st

# Plotlyをモック
mock_go = MagicMock()
mock_figure = MagicMock()
mock_go.Figure.return_value = mock_figure
mock_go.Scatter.return_value = MagicMock()
mock_make_subplots = MagicMock()
mock_make_subplots.return_value = MagicMock()
sys.modules['plotly'] = MagicMock()
sys.modules['plotly.graph_objects'] = mock_go
mock_subplots_module = MagicMock()
mock_subplots_module.make_subplots = mock_make_subplots
sys.modules['plotly.subplots'] = mock_subplots_module

# COLORS定義（training_styles.pyから）
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

# training_stylesモジュールをモック
mock_training_styles = MagicMock()
mock_training_styles.COLORS = COLORS
sys.modules['components'] = MagicMock()
sys.modules['components.training_styles'] = mock_training_styles

# componentsパッケージの__init__をセットアップ
components_init = MagicMock()
components_init.training_styles = mock_training_styles
sys.modules['components'] = components_init

# training_chartsモジュールをインポート
# ファイルを直接読み込んでexec
import types

training_charts_path = Path(__file__).parent.parent.parent.parent / "app" / "components" / "training_charts.py"
with open(training_charts_path, 'r') as f:
    source = f.read()

# 相対インポートを絶対インポートに変更
source = source.replace("from .training_styles import COLORS", "from components.training_styles import COLORS")

# モジュールを作成して実行
training_charts_module = types.ModuleType("training_charts")
training_charts_module.__file__ = str(training_charts_path)
exec(compile(source, training_charts_path, 'exec'), training_charts_module.__dict__)

# テスト対象関数を取得
render_training_chart = training_charts_module.render_training_chart
render_epoch_metrics_chart = training_charts_module.render_epoch_metrics_chart
render_loss_breakdown_chart = training_charts_module.render_loss_breakdown_chart
_render_empty_chart_placeholder = training_charts_module._render_empty_chart_placeholder
PLOTLY_THEME = training_charts_module.PLOTLY_THEME


# =============================================================================
# テストデータ
# =============================================================================

SAMPLE_TRAINING_HISTORY = [
    {"epoch": 1, "mAP50": 0.3, "loss": 1.5},
    {"epoch": 2, "mAP50": 0.5, "loss": 1.0},
    {"epoch": 3, "mAP50": 0.7, "loss": 0.6},
]

SAMPLE_SINGLE_EPOCH = [
    {"epoch": 1, "mAP50": 0.3, "loss": 1.5},
]

SAMPLE_METRICS_HISTORY = [
    {"epoch": 1, "precision": 0.5, "recall": 0.4, "mAP50": 0.3, "mAP50-95": 0.2},
    {"epoch": 2, "precision": 0.7, "recall": 0.6, "mAP50": 0.5, "mAP50-95": 0.4},
]

SAMPLE_LOSS_HISTORY = [
    {"epoch": 1, "box_loss": 0.5, "cls_loss": 0.4, "dfl_loss": 0.3},
    {"epoch": 2, "box_loss": 0.3, "cls_loss": 0.2, "dfl_loss": 0.2},
]

SAMPLE_HISTORY_MISSING_KEYS = [
    {"epoch": 1, "mAP50": 0.3},  # loss キーなし
    {"mAP50": 0.5, "loss": 1.0},  # epoch キーなし
]


# =============================================================================
# TestRenderTrainingChart
# =============================================================================


class TestRenderTrainingChart:
    """render_training_chart() のテスト"""

    def setup_method(self):
        """各テスト前にモックをリセット"""
        mock_st.reset_mock()
        mock_go.reset_mock()
        mock_make_subplots.reset_mock()

    def test_empty_history_shows_placeholder(self):
        """空の履歴の場合、プレースホルダーが表示される"""
        render_training_chart([])
        # st.htmlが呼ばれる（プレースホルダー表示）
        mock_st.html.assert_called()

    def test_single_epoch_data(self):
        """単一エポックのデータでチャートが描画される"""
        mock_fig = MagicMock()
        mock_make_subplots.return_value = mock_fig

        render_training_chart(SAMPLE_SINGLE_EPOCH)

        # make_subplotsが呼ばれる
        mock_make_subplots.assert_called_once()
        # add_traceが呼ばれる（mAP50とLoss）
        assert mock_fig.add_trace.call_count == 2

    def test_multiple_epochs_data(self):
        """複数エポックのデータでチャートが描画される"""
        mock_fig = MagicMock()
        mock_make_subplots.return_value = mock_fig

        render_training_chart(SAMPLE_TRAINING_HISTORY)

        mock_make_subplots.assert_called_once()
        # add_traceが2回（mAP50とLoss）
        assert mock_fig.add_trace.call_count == 2

    def test_data_extraction_with_missing_keys(self):
        """キーが欠けているデータでもデフォルト値で処理される"""
        mock_fig = MagicMock()
        mock_make_subplots.return_value = mock_fig

        # 例外が発生しないこと
        render_training_chart(SAMPLE_HISTORY_MISSING_KEYS)

        mock_make_subplots.assert_called_once()

    def test_target_line_rendered(self):
        """ターゲットライン（mAP50目標）が描画される"""
        mock_fig = MagicMock()
        mock_make_subplots.return_value = mock_fig

        render_training_chart(SAMPLE_TRAINING_HISTORY, target_map50=0.85)

        # add_hlineが呼ばれる
        mock_fig.add_hline.assert_called()

    def test_target_line_not_rendered_when_zero(self):
        """target_map50が0の場合、ターゲットラインは描画されない"""
        mock_fig = MagicMock()
        mock_make_subplots.return_value = mock_fig

        render_training_chart(SAMPLE_TRAINING_HISTORY, target_map50=0)

        # add_hlineが呼ばれない
        mock_fig.add_hline.assert_not_called()

    def test_secondary_yaxis_configuration(self):
        """二軸チャート設定が正しく適用される"""
        mock_fig = MagicMock()
        mock_make_subplots.return_value = mock_fig

        render_training_chart(SAMPLE_TRAINING_HISTORY)

        # make_subplotsにsecondary_y指定がある
        call_kwargs = mock_make_subplots.call_args
        assert call_kwargs is not None
        specs = call_kwargs[1].get('specs', [[]])
        assert specs[0][0].get('secondary_y') is True

    def test_custom_height(self):
        """カスタム高さパラメータが適用される"""
        mock_fig = MagicMock()
        mock_make_subplots.return_value = mock_fig

        render_training_chart(SAMPLE_TRAINING_HISTORY, height=500)

        # update_layoutにheightが渡される
        mock_fig.update_layout.assert_called()
        call_kwargs = mock_fig.update_layout.call_args[1]
        assert call_kwargs.get('height') == 500

    def test_show_title_true(self):
        """show_title=Trueでタイトルが表示される"""
        mock_fig = MagicMock()
        mock_make_subplots.return_value = mock_fig

        render_training_chart(SAMPLE_TRAINING_HISTORY, show_title=True)

        mock_fig.update_layout.assert_called()
        call_kwargs = mock_fig.update_layout.call_args[1]
        assert call_kwargs.get('title') is not None
        assert call_kwargs['title']['text'] == "Training Progress"

    def test_show_title_false(self):
        """show_title=Falseでタイトルがない"""
        mock_fig = MagicMock()
        mock_make_subplots.return_value = mock_fig

        render_training_chart(SAMPLE_TRAINING_HISTORY, show_title=False)

        mock_fig.update_layout.assert_called()
        call_kwargs = mock_fig.update_layout.call_args[1]
        assert call_kwargs['title']['text'] is None

    def test_plotly_chart_called(self):
        """st.plotly_chartが呼ばれる"""
        mock_fig = MagicMock()
        mock_make_subplots.return_value = mock_fig

        render_training_chart(SAMPLE_TRAINING_HISTORY)

        mock_st.plotly_chart.assert_called_once()


# =============================================================================
# TestRenderEpochMetricsChart
# =============================================================================


class TestRenderEpochMetricsChart:
    """render_epoch_metrics_chart() のテスト"""

    def setup_method(self):
        """各テスト前にモックをリセット"""
        mock_st.reset_mock()

    def test_empty_history_shows_placeholder(self):
        """空の履歴の場合、プレースホルダーが表示される"""
        render_epoch_metrics_chart([])
        mock_st.html.assert_called()

    def test_with_valid_history_renders_chart(self):
        """有効な履歴がある場合、チャートが描画される"""
        render_epoch_metrics_chart(SAMPLE_METRICS_HISTORY)

        # st.plotly_chartが呼ばれる
        mock_st.plotly_chart.assert_called_once()

    def test_with_custom_metrics_renders_chart(self):
        """カスタムメトリクスでもチャートが描画される"""
        render_epoch_metrics_chart(SAMPLE_METRICS_HISTORY, metrics=["precision", "recall"])

        # st.plotly_chartが呼ばれる
        mock_st.plotly_chart.assert_called_once()

    def test_custom_height(self):
        """カスタム高さパラメータで描画される"""
        # 例外なく実行されること
        render_epoch_metrics_chart(SAMPLE_METRICS_HISTORY, height=400)
        mock_st.plotly_chart.assert_called_once()

    def test_plotly_chart_called(self):
        """st.plotly_chartが呼ばれる"""
        render_epoch_metrics_chart(SAMPLE_METRICS_HISTORY)

        mock_st.plotly_chart.assert_called_once()


# =============================================================================
# TestRenderLossBreakdownChart
# =============================================================================


class TestRenderLossBreakdownChart:
    """render_loss_breakdown_chart() のテスト"""

    def setup_method(self):
        """各テスト前にモックをリセット"""
        mock_st.reset_mock()

    def test_empty_history_returns_early(self):
        """空の履歴の場合、早期リターン（プレースホルダーなし）"""
        render_loss_breakdown_chart([])

        # plotly_chartが呼ばれない
        mock_st.plotly_chart.assert_not_called()

    def test_with_valid_history_renders_chart(self):
        """有効な履歴がある場合、チャートが描画される"""
        render_loss_breakdown_chart(SAMPLE_LOSS_HISTORY)

        # st.plotly_chartが呼ばれる
        mock_st.plotly_chart.assert_called_once()

    def test_with_partial_zero_losses(self):
        """一部が0のロスでもチャートが描画される"""
        history = [
            {"epoch": 1, "box_loss": 0.5, "cls_loss": 0.0, "dfl_loss": 0.3},
            {"epoch": 2, "box_loss": 0.3, "cls_loss": 0.0, "dfl_loss": 0.2},
        ]

        render_loss_breakdown_chart(history)

        # st.plotly_chartが呼ばれる
        mock_st.plotly_chart.assert_called_once()

    def test_custom_height(self):
        """カスタム高さパラメータで描画される"""
        render_loss_breakdown_chart(SAMPLE_LOSS_HISTORY, height=300)
        mock_st.plotly_chart.assert_called_once()

    def test_plotly_chart_called(self):
        """st.plotly_chartが呼ばれる"""
        render_loss_breakdown_chart(SAMPLE_LOSS_HISTORY)

        mock_st.plotly_chart.assert_called_once()


# =============================================================================
# TestRenderEmptyChartPlaceholder
# =============================================================================


class TestRenderEmptyChartPlaceholder:
    """_render_empty_chart_placeholder() のテスト"""

    def setup_method(self):
        """各テスト前にモックをリセット"""
        mock_st.reset_mock()

    def test_html_rendered(self):
        """st.htmlが呼ばれてHTMLが描画される"""
        _render_empty_chart_placeholder()

        mock_st.html.assert_called_once()
        # HTMLにプレースホルダーテキストが含まれる
        html_content = mock_st.html.call_args[0][0]
        assert "Waiting for training data" in html_content


# =============================================================================
# TestPlotlyTheme
# =============================================================================


class TestPlotlyTheme:
    """PLOTLY_THEME 定数のテスト"""

    def test_theme_keys(self):
        """テーマに必要なキーが含まれている"""
        required_keys = [
            "paper_bgcolor",
            "plot_bgcolor",
            "font_family",
            "font_color",
            "gridcolor",
            "linecolor",
        ]
        for key in required_keys:
            assert key in PLOTLY_THEME

    def test_transparent_backgrounds(self):
        """背景が透明に設定されている"""
        assert PLOTLY_THEME["paper_bgcolor"] == "rgba(0,0,0,0)"
        assert PLOTLY_THEME["plot_bgcolor"] == "rgba(0,0,0,0)"


# =============================================================================
# TestColors
# =============================================================================


class TestColors:
    """COLORS 定数のテスト"""

    def test_required_colors(self):
        """必要な色キーが含まれている"""
        required_colors = [
            "accent_primary",
            "accent_secondary",
            "success",
            "error",
            "info",
            "tier_high",
            "text_secondary",
        ]
        for color_key in required_colors:
            assert color_key in COLORS
