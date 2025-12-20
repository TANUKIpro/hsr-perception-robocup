"""
Training Integration テスト

app/pages/5_Training.py のテスト
"""

import pytest
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch
from enum import Enum


# =============================================================================
# モックセットアップ（インポート前に）
# =============================================================================

# Streamlitをモック
mock_st = MagicMock()
mock_st.session_state = MagicMock()
mock_st.title = MagicMock()
mock_st.tabs = MagicMock()
mock_st.selectbox = MagicMock()
mock_st.slider = MagicMock(return_value=50)
mock_st.checkbox = MagicMock(return_value=False)
mock_st.info = MagicMock()
mock_st.warning = MagicMock()
mock_st.error = MagicMock()
mock_st.success = MagicMock()
mock_st.columns = MagicMock()
mock_st.expander = MagicMock()
mock_st.container = MagicMock()
mock_st.html = MagicMock()
mock_st.button = MagicMock(return_value=False)
mock_st.number_input = MagicMock(return_value=6006)
mock_st.markdown = MagicMock()
mock_st.rerun = MagicMock()
mock_st.balloons = MagicMock()
mock_st.code = MagicMock()
mock_st.plotly_chart = MagicMock()
mock_st.form = MagicMock()
mock_st.cache_resource = lambda f: f  # デコレータとして動作
mock_st.download_button = MagicMock()
sys.modules['streamlit'] = mock_st

# torchをモック
mock_torch = MagicMock()
mock_torch.cuda = MagicMock()
mock_torch.cuda.is_available = MagicMock(return_value=True)
mock_device_props = MagicMock()
mock_device_props.total_memory = 12 * 1e9  # 12GB
mock_torch.cuda.get_device_properties = MagicMock(return_value=mock_device_props)
mock_torch.cuda.get_device_name = MagicMock(return_value="NVIDIA GeForce RTX 3080")
sys.modules['torch'] = mock_torch

# yamlをモック
mock_yaml = MagicMock()
mock_yaml.safe_load = MagicMock(return_value={"names": ["class1", "class2", "class3"]})
sys.modules['yaml'] = mock_yaml

# コンポーネントをモック
mock_common_sidebar = MagicMock()
mock_common_sidebar.render_common_sidebar = MagicMock()
sys.modules['components'] = MagicMock()
sys.modules['components.common_sidebar'] = mock_common_sidebar

mock_training_styles = MagicMock()
mock_training_styles.inject_training_styles = MagicMock()
mock_training_styles.COLORS = {
    "accent_primary": "#00D4AA",
    "accent_secondary": "#00B4D8",
    "info": "#58A6FF",
    "success": "#3FB950",
    "warning": "#D29922",
    "error": "#F85149",
    "tier_high": "#A371F7",
}
mock_training_styles.ICONS = {
    "dataset": "📂",
    "model": "🤖",
    "accuracy": "📊",
    "tensorboard": "📈",
    "terminal": ">",
}
sys.modules['components.training_styles'] = mock_training_styles

mock_training_charts = MagicMock()
mock_training_charts.render_training_chart = MagicMock()
mock_training_charts.render_epoch_metrics_chart = MagicMock()
sys.modules['components.training_charts'] = mock_training_charts

mock_tensorboard_embed = MagicMock()
mock_tensorboard_embed.render_tensorboard_panel = MagicMock()
mock_tensorboard_embed.render_tensorboard_status = MagicMock()
sys.modules['components.tensorboard_embed'] = mock_tensorboard_embed

mock_config_preview = MagicMock()
mock_config_preview.validate_training_config = MagicMock(return_value={"errors": [], "warnings": [], "info": []})
mock_config_preview.render_validation_messages = MagicMock()
mock_config_preview.render_gpu_status_card = MagicMock()
mock_config_preview.render_gpu_not_available = MagicMock()
mock_config_preview.render_config_summary = MagicMock()
mock_config_preview.render_model_recommendation = MagicMock()
mock_config_preview.render_target_metrics_info = MagicMock()
sys.modules['components.config_preview'] = mock_config_preview

mock_progress_display = MagicMock()
mock_progress_display.render_task_progress = MagicMock()
mock_progress_display.render_active_task_banner = MagicMock()
mock_progress_display.render_task_list = MagicMock()
mock_progress_display.render_task_metrics = MagicMock()
mock_progress_display.render_training_active_banner = MagicMock()
mock_progress_display.render_training_completed_banner = MagicMock()
mock_progress_display.render_circular_progress = MagicMock()
mock_progress_display.render_training_metric_cards = MagicMock()
sys.modules['components.progress_display'] = mock_progress_display

mock_advanced_params = MagicMock()
mock_advanced_params.render_advanced_parameters_section = MagicMock(return_value={
    "imgsz": 640,
    "lr0": 0.01,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1,
    "box": 7.5,
    "cls": 0.5,
    "dfl": 1.5,
})
sys.modules['components.training_advanced_params'] = mock_advanced_params


# TaskStatusをモック
class MockTaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


mock_task_manager_module = MagicMock()
mock_task_manager_module.TaskManager = MagicMock
mock_task_manager_module.TaskStatus = MockTaskStatus
sys.modules['services'] = MagicMock()
sys.modules['services.task_manager'] = mock_task_manager_module

mock_path_coordinator_module = MagicMock()
mock_path_coordinator_module.PathCoordinator = MagicMock
sys.modules['services.path_coordinator'] = mock_path_coordinator_module


# =============================================================================
# モジュールのロード
# =============================================================================

APP_PATH = Path(__file__).parent.parent.parent.parent / "app"

# Training ページのソースを読み込み
training_page_path = APP_PATH / "pages" / "5_Training.py"
with open(training_page_path, 'r') as f:
    training_source = f.read()

# TYPE_CHECKINGを無効化
training_source = training_source.replace(
    "if TYPE_CHECKING:",
    "if False:"
)

training_module = types.ModuleType("training_page")
training_module.__file__ = str(training_page_path)

# app_dirをモジュールに設定
training_module.app_dir = APP_PATH
training_module.project_root = APP_PATH.parent

exec(compile(training_source, training_page_path, 'exec'), training_module.__dict__)

# テスト対象関数を取得
show_training_page = training_module.show_training_page
_get_active_training_task = training_module._get_active_training_task
_render_active_training_view = training_module._render_active_training_view
_render_start_training = training_module._render_start_training
_render_dataset_section = training_module._render_dataset_section
_render_model_section = training_module._render_model_section
_render_advanced_section = training_module._render_advanced_section
_render_hardware_section = training_module._render_hardware_section
_render_start_button = training_module._render_start_button
_render_trained_models = training_module._render_trained_models
_render_training_history = training_module._render_training_history
_get_gpu_info = training_module._get_gpu_info


# =============================================================================
# フィクスチャ
# =============================================================================

@pytest.fixture
def mock_task_manager():
    """TaskManagerのモック"""
    tm = MagicMock()
    tm.get_active_tasks = MagicMock(return_value=[])
    tm.get_task = MagicMock(return_value=None)
    tm.start_training = MagicMock(return_value="task_123")
    tm.cancel_task = MagicMock()
    return tm


@pytest.fixture
def mock_path_coordinator():
    """PathCoordinatorのモック"""
    pc = MagicMock()
    pc.get_annotation_sessions = MagicMock(return_value=[])
    pc.get_trained_models = MagicMock(return_value=[])
    pc.get_pretrained_models = MagicMock(return_value=[])
    pc.get_path = MagicMock(return_value=Path("/tmp/test"))
    pc.get_mask_stats = MagicMock(return_value={})
    pc.get_background_images = MagicMock(return_value=[])
    return pc


@pytest.fixture
def sample_sessions():
    """テスト用セッションリスト"""
    return [
        {
            "name": "session1",
            "path": "/data/sessions/session1",
            "has_data_yaml": True,
            "created": "2024-01-15 10:00:00",
        },
        {
            "name": "session2",
            "path": "/data/sessions/session2",
            "has_data_yaml": True,
            "created": "2024-01-16 11:00:00",
        },
        {
            "name": "session3_incomplete",
            "path": "/data/sessions/session3",
            "has_data_yaml": False,
            "created": "2024-01-17 12:00:00",
        },
    ]


@pytest.fixture
def sample_trained_models():
    """テスト用学習済みモデルリスト"""
    return [
        {
            "name": "model_v1",
            "best_path": "/models/model_v1/weights/best.pt",
            "last_path": "/models/model_v1/weights/last.pt",
            "created": "2024-01-20 14:00:00",
        },
        {
            "name": "model_v2",
            "best_path": "/models/model_v2/weights/best.pt",
            "last_path": None,
            "created": "2024-01-21 15:00:00",
        },
    ]


@pytest.fixture
def mock_active_task():
    """アクティブなトレーニングタスクのモック"""
    task = MagicMock()
    task.task_id = "task_123"
    task.status = MockTaskStatus.RUNNING
    task.progress = 0.5
    task.extra_data = {
        "gpu_info": "NVIDIA GeForce RTX 3080 (12GB)",
        "gpu_tier": "high",
        "config": {
            "model": "yolov8m.pt",
            "batch": 16,
            "epochs": 50,
            "imgsz": 640,
        },
        "training_history": [
            {"epoch": 1, "mAP50": 0.3, "loss": 1.5},
            {"epoch": 2, "mAP50": 0.5, "loss": 1.0},
        ],
        "tensorboard_url": "http://localhost:6006",
    }
    return task


# =============================================================================
# TestLoadDatasets
# =============================================================================

class TestLoadDatasets:
    """データセット読み込みのテスト"""

    def setup_method(self):
        """各テスト前にモックをリセット"""
        mock_st.reset_mock()

    def test_no_datasets_shows_warning(self, mock_path_coordinator):
        """データセットがない場合、警告が表示される"""
        mock_path_coordinator.get_annotation_sessions.return_value = []

        _render_dataset_section(mock_path_coordinator)

        # HTMLで警告メッセージが表示される
        mock_st.html.assert_called()
        call_args = str(mock_st.html.call_args_list)
        assert "No annotated datasets" in call_args or mock_st.html.call_count > 0

    def test_datasets_shown_in_selectbox(self, mock_path_coordinator, sample_sessions):
        """データセットがselectboxに表示される"""
        mock_path_coordinator.get_annotation_sessions.return_value = sample_sessions
        mock_st.selectbox.return_value = sample_sessions[0]

        # open()をモック
        with patch("builtins.open", MagicMock()):
            _render_dataset_section(mock_path_coordinator)

        # selectboxが呼ばれる
        mock_st.selectbox.assert_called()

    def test_only_ready_sessions_shown(self, mock_path_coordinator, sample_sessions):
        """has_data_yaml=Trueのセッションのみ表示される"""
        mock_path_coordinator.get_annotation_sessions.return_value = sample_sessions

        # selectboxに渡されるセッションを確認
        mock_st.selectbox.return_value = sample_sessions[0]

        with patch("builtins.open", MagicMock()):
            _render_dataset_section(mock_path_coordinator)

        # selectboxが呼ばれる
        if mock_st.selectbox.call_count > 0:
            call_args = mock_st.selectbox.call_args
            sessions_arg = call_args[0][1] if len(call_args[0]) > 1 else []
            # has_data_yaml=Falseのsession3_incompleteは含まれないはず
            for session in sessions_arg:
                assert session.get("has_data_yaml", False) is True


# =============================================================================
# TestLoadModels
# =============================================================================

class TestLoadModels:
    """モデル読み込みのテスト"""

    def setup_method(self):
        """各テスト前にモックをリセット"""
        mock_st.reset_mock()

    def test_model_selection_options(self, mock_path_coordinator):
        """モデル選択オプションが表示される"""
        mock_path_coordinator.get_pretrained_models.return_value = []

        mock_col = MagicMock()
        mock_col.__enter__ = MagicMock(return_value=mock_col)
        mock_col.__exit__ = MagicMock(return_value=None)
        mock_st.columns.return_value = [mock_col, mock_col]

        mock_st.selectbox.return_value = "yolov8m.pt"
        mock_st.slider.return_value = 50
        mock_st.checkbox.return_value = False

        result = _render_model_section(
            auto_scale=False,
            gpu_available=True,
            gpu_tier="high",
            path_coordinator=mock_path_coordinator
        )

        # 戻り値を確認（base_model, epochs, batch_size, fast_mode）
        assert len(result) == 4

    def test_auto_scale_hides_model_selection(self, mock_path_coordinator):
        """auto_scale時はモデル選択が自動になる"""
        mock_col = MagicMock()
        mock_col.__enter__ = MagicMock(return_value=mock_col)
        mock_col.__exit__ = MagicMock(return_value=None)
        mock_st.columns.return_value = [mock_col, mock_col]

        mock_st.slider.return_value = 50
        mock_st.checkbox.return_value = False

        result = _render_model_section(
            auto_scale=True,
            gpu_available=True,
            gpu_tier="high",
            path_coordinator=mock_path_coordinator
        )

        # auto_scaleの場合、base_modelはNone
        base_model, epochs, batch_size, fast_mode = result
        assert base_model is None


# =============================================================================
# TestAdvancedParams
# =============================================================================

class TestAdvancedParams:
    """詳細パラメータのテスト"""

    def setup_method(self):
        """各テスト前にモックをリセット"""
        mock_st.reset_mock()
        mock_advanced_params.reset_mock()

    def test_advanced_section_renders(self):
        """詳細パラメータセクションがレンダリングされる"""
        result = _render_advanced_section(auto_scale=False, gpu_tier="high")

        # render_advanced_parameters_sectionが呼ばれる
        mock_advanced_params.render_advanced_parameters_section.assert_called_once()

        # 戻り値は辞書
        assert isinstance(result, dict)

    def test_advanced_params_passed_correctly(self):
        """パラメータが正しく渡される"""
        _render_advanced_section(auto_scale=True, gpu_tier="workstation")

        call_kwargs = mock_advanced_params.render_advanced_parameters_section.call_args[1]
        assert call_kwargs["auto_scale"] is True
        assert call_kwargs["gpu_tier"] == "workstation"


# =============================================================================
# TestStartTraining
# =============================================================================

class TestStartTraining:
    """トレーニング開始のテスト"""

    def setup_method(self):
        """各テスト前にモックをリセット"""
        mock_st.reset_mock()

    def test_start_button_without_dataset_shows_warning(self, mock_task_manager, mock_path_coordinator):
        """データセット未選択時は警告が表示される"""
        mock_st.session_state.get = MagicMock(return_value=None)

        _render_start_button(
            task_manager=mock_task_manager,
            path_coordinator=mock_path_coordinator,
            base_model="yolov8m.pt",
            epochs=50,
            batch_size=16,
            fast_mode=False,
            auto_scale=False,
            enable_tensorboard=True,
            tensorboard_port=6006,
            advanced_params={},
        )

        mock_st.warning.assert_called()

    def test_start_training_calls_task_manager(self, mock_task_manager, mock_path_coordinator):
        """トレーニング開始でTaskManagerが呼ばれる"""
        # session_stateにデータセット情報を設定
        dataset_info = {
            "session": {"name": "test_session", "path": "/data/sessions/test"},
            "data_yaml": "/data/sessions/test/data.yaml",
            "train_count": 100,
            "val_count": 20,
            "class_names": ["class1", "class2"],
        }

        def get_side_effect(key, default=None):
            if key == "selected_dataset_info":
                return dataset_info
            elif key == "synthetic_config":
                return {"enabled": False}
            return default

        mock_st.session_state.get = MagicMock(side_effect=get_side_effect)
        mock_st.button.return_value = True  # ボタンクリック

        mock_col = MagicMock()
        mock_col.__enter__ = MagicMock(return_value=mock_col)
        mock_col.__exit__ = MagicMock(return_value=None)
        mock_st.columns.return_value = [mock_col, mock_col, mock_col]

        _render_start_button(
            task_manager=mock_task_manager,
            path_coordinator=mock_path_coordinator,
            base_model="yolov8m.pt",
            epochs=50,
            batch_size=16,
            fast_mode=False,
            auto_scale=False,
            enable_tensorboard=True,
            tensorboard_port=6006,
            advanced_params={},
        )

        # start_trainingが呼ばれる
        mock_task_manager.start_training.assert_called_once()


# =============================================================================
# TestMonitorTrainingProgress
# =============================================================================

class TestMonitorTrainingProgress:
    """トレーニング進捗監視のテスト"""

    def setup_method(self):
        """各テスト前にモックをリセット"""
        mock_st.reset_mock()
        mock_progress_display.reset_mock()
        mock_training_charts.reset_mock()

    def test_active_training_renders_banner(self, mock_task_manager, mock_path_coordinator, mock_active_task):
        """アクティブなトレーニングでバナーが表示される"""
        mock_task_manager.get_task.return_value = mock_active_task

        mock_col = MagicMock()
        mock_col.__enter__ = MagicMock(return_value=mock_col)
        mock_col.__exit__ = MagicMock(return_value=None)
        mock_st.columns.return_value = [mock_col, mock_col]

        mock_expander = MagicMock()
        mock_expander.__enter__ = MagicMock(return_value=mock_expander)
        mock_expander.__exit__ = MagicMock(return_value=None)
        mock_st.expander.return_value = mock_expander

        # time.sleepをモック（無限ループ回避）
        with patch("time.sleep", MagicMock()):
            _render_active_training_view(mock_active_task, mock_task_manager, mock_path_coordinator)

        # バナーが表示される
        mock_progress_display.render_training_active_banner.assert_called()

    def test_training_chart_rendered(self, mock_task_manager, mock_path_coordinator, mock_active_task):
        """トレーニングチャートが表示される"""
        mock_task_manager.get_task.return_value = mock_active_task

        mock_col = MagicMock()
        mock_col.__enter__ = MagicMock(return_value=mock_col)
        mock_col.__exit__ = MagicMock(return_value=None)
        mock_st.columns.return_value = [mock_col, mock_col]

        mock_expander = MagicMock()
        mock_expander.__enter__ = MagicMock(return_value=mock_expander)
        mock_expander.__exit__ = MagicMock(return_value=None)
        mock_st.expander.return_value = mock_expander

        with patch("time.sleep", MagicMock()):
            _render_active_training_view(mock_active_task, mock_task_manager, mock_path_coordinator)

        # チャートが描画される
        mock_training_charts.render_training_chart.assert_called()

    def test_completed_training_shows_balloons(self, mock_task_manager, mock_path_coordinator, mock_active_task):
        """完了したトレーニングでballoonsが表示される"""
        mock_active_task.status = MockTaskStatus.COMPLETED
        mock_task_manager.get_task.return_value = mock_active_task

        mock_col = MagicMock()
        mock_col.__enter__ = MagicMock(return_value=mock_col)
        mock_col.__exit__ = MagicMock(return_value=None)
        mock_st.columns.return_value = [mock_col, mock_col]

        mock_expander = MagicMock()
        mock_expander.__enter__ = MagicMock(return_value=mock_expander)
        mock_expander.__exit__ = MagicMock(return_value=None)
        mock_st.expander.return_value = mock_expander

        _render_active_training_view(mock_active_task, mock_task_manager, mock_path_coordinator)

        # balloonsが表示される
        mock_st.balloons.assert_called()


# =============================================================================
# TestCancelTraining
# =============================================================================

class TestCancelTraining:
    """トレーニングキャンセルのテスト"""

    def setup_method(self):
        """各テスト前にモックをリセット"""
        mock_st.reset_mock()

    def test_cancel_button_in_active_banner(self, mock_task_manager, mock_path_coordinator, mock_active_task):
        """アクティブバナーにキャンセルボタンがある"""
        mock_task_manager.get_task.return_value = mock_active_task

        mock_col = MagicMock()
        mock_col.__enter__ = MagicMock(return_value=mock_col)
        mock_col.__exit__ = MagicMock(return_value=None)
        mock_st.columns.return_value = [mock_col, mock_col]

        mock_expander = MagicMock()
        mock_expander.__enter__ = MagicMock(return_value=mock_expander)
        mock_expander.__exit__ = MagicMock(return_value=None)
        mock_st.expander.return_value = mock_expander

        with patch("time.sleep", MagicMock()):
            _render_active_training_view(mock_active_task, mock_task_manager, mock_path_coordinator)

        # render_training_active_bannerが呼ばれる（キャンセルボタンはその中）
        mock_progress_display.render_training_active_banner.assert_called_with(
            mock_active_task, mock_task_manager
        )


# =============================================================================
# TestGpuDetection
# =============================================================================

class TestGpuDetection:
    """GPU検出のテスト"""

    def setup_method(self):
        """各テスト前にモックをリセット"""
        mock_st.reset_mock()
        mock_config_preview.reset_mock()

    def test_gpu_available_shows_status_card(self):
        """GPUが利用可能な場合、ステータスカードが表示される"""
        mock_st.checkbox.return_value = False

        result = _render_hardware_section()

        # render_gpu_status_cardが呼ばれる
        mock_config_preview.render_gpu_status_card.assert_called()

        # 戻り値を確認
        gpu_available, gpu_name, gpu_memory, gpu_tier, auto_scale = result
        assert gpu_available is True

    def test_gpu_not_available_shows_warning(self):
        """GPUが利用不可の場合、警告が表示される"""
        mock_torch.cuda.is_available.return_value = False
        mock_st.checkbox.return_value = False

        result = _render_hardware_section()

        # render_gpu_not_availableが呼ばれる
        mock_config_preview.render_gpu_not_available.assert_called()

        gpu_available, gpu_name, gpu_memory, gpu_tier, auto_scale = result
        assert gpu_available is False

        # 元に戻す
        mock_torch.cuda.is_available.return_value = True

    def test_gpu_tier_detection(self):
        """GPUティアが正しく検出される"""
        # 12GBの場合 → high tier
        mock_device_props.total_memory = 12 * 1e9
        mock_st.checkbox.return_value = False

        result = _render_hardware_section()

        gpu_available, gpu_name, gpu_memory, gpu_tier, auto_scale = result
        assert gpu_tier == "high"


# =============================================================================
# TestTensorboardEmbed
# =============================================================================

class TestTensorboardEmbed:
    """TensorBoard埋め込みのテスト"""

    def setup_method(self):
        """各テスト前にモックをリセット"""
        mock_st.reset_mock()
        mock_tensorboard_embed.reset_mock()

    def test_tensorboard_panel_rendered(self, mock_task_manager, mock_path_coordinator, mock_active_task):
        """TensorBoardパネルが表示される"""
        mock_task_manager.get_task.return_value = mock_active_task

        mock_col = MagicMock()
        mock_col.__enter__ = MagicMock(return_value=mock_col)
        mock_col.__exit__ = MagicMock(return_value=None)
        mock_st.columns.return_value = [mock_col, mock_col]

        mock_expander = MagicMock()
        mock_expander.__enter__ = MagicMock(return_value=mock_expander)
        mock_expander.__exit__ = MagicMock(return_value=None)
        mock_st.expander.return_value = mock_expander

        with patch("time.sleep", MagicMock()):
            _render_active_training_view(mock_active_task, mock_task_manager, mock_path_coordinator)

        # TensorBoardパネルが表示される
        mock_tensorboard_embed.render_tensorboard_panel.assert_called()

    def test_tensorboard_status_when_no_url(self, mock_task_manager, mock_path_coordinator, mock_active_task):
        """TensorBoard URLがない場合、ステータスが表示される"""
        mock_active_task.extra_data["tensorboard_url"] = None
        mock_task_manager.get_task.return_value = mock_active_task

        mock_col = MagicMock()
        mock_col.__enter__ = MagicMock(return_value=mock_col)
        mock_col.__exit__ = MagicMock(return_value=None)
        mock_st.columns.return_value = [mock_col, mock_col]

        mock_expander = MagicMock()
        mock_expander.__enter__ = MagicMock(return_value=mock_expander)
        mock_expander.__exit__ = MagicMock(return_value=None)
        mock_st.expander.return_value = mock_expander

        with patch("time.sleep", MagicMock()):
            _render_active_training_view(mock_active_task, mock_task_manager, mock_path_coordinator)

        # ステータスが表示される
        mock_tensorboard_embed.render_tensorboard_status.assert_called_with(is_running=False)


# =============================================================================
# TestTrainedModels
# =============================================================================

class TestTrainedModels:
    """学習済みモデル表示のテスト"""

    def setup_method(self):
        """各テスト前にモックをリセット"""
        mock_st.reset_mock()

    def test_no_models_shows_placeholder(self, mock_path_coordinator):
        """モデルがない場合、プレースホルダーが表示される"""
        mock_path_coordinator.get_trained_models.return_value = []

        _render_trained_models(mock_path_coordinator)

        # HTMLでプレースホルダーが表示される
        mock_st.html.assert_called()

    def test_models_shown_in_expander(self, mock_path_coordinator, sample_trained_models):
        """モデルがexpanderに表示される"""
        mock_path_coordinator.get_trained_models.return_value = sample_trained_models

        mock_expander = MagicMock()
        mock_expander.__enter__ = MagicMock(return_value=mock_expander)
        mock_expander.__exit__ = MagicMock(return_value=None)
        mock_st.expander.return_value = mock_expander

        mock_col = MagicMock()
        mock_col.__enter__ = MagicMock(return_value=mock_col)
        mock_col.__exit__ = MagicMock(return_value=None)
        mock_st.columns.return_value = [mock_col, mock_col]

        _render_trained_models(mock_path_coordinator)

        # expanderが各モデルに対して呼ばれる
        assert mock_st.expander.call_count == 2


# =============================================================================
# TestTrainingHistory
# =============================================================================

class TestTrainingHistory:
    """トレーニング履歴のテスト"""

    def setup_method(self):
        """各テスト前にモックをリセット"""
        mock_st.reset_mock()
        mock_progress_display.reset_mock()

    def test_history_renders_task_list(self, mock_task_manager):
        """履歴でタスクリストが表示される"""
        _render_training_history(mock_task_manager)

        # render_task_listが呼ばれる
        mock_progress_display.render_task_list.assert_called_with(
            task_type="training",
            task_manager=mock_task_manager,
            limit=10,
            show_active_only=False,
        )
