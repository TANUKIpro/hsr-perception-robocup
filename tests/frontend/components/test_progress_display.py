"""
Progress Display テスト

app/components/progress_display.py のテスト
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta

# アプリパスを追加（インポート前に）
APP_PATH = str(Path(__file__).parent.parent.parent.parent / "app")
sys.path.insert(0, APP_PATH)

# Streamlitをモック（インポート前に）
mock_st = MagicMock()
mock_st.markdown = MagicMock()
mock_st.error = MagicMock()
mock_st.info = MagicMock()
mock_st.warning = MagicMock()
mock_st.success = MagicMock()
mock_st.caption = MagicMock()
mock_st.subheader = MagicMock()
mock_st.progress = MagicMock()
mock_st.metric = MagicMock()
mock_st.write = MagicMock()
mock_st.code = MagicMock()
mock_st.columns = MagicMock(return_value=[MagicMock(), MagicMock(), MagicMock(), MagicMock()])
mock_st.container = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))
mock_st.expander = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))
mock_st.button = MagicMock(return_value=False)
mock_st.rerun = MagicMock()
mock_st.session_state = {}
sys.modules['streamlit'] = mock_st

# TaskStatusとTaskInfo用のモック
from enum import Enum


class TaskStatus(Enum):
    """タスクステータス"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskInfo:
    """タスク情報のモック"""

    def __init__(
        self,
        task_id: str = "test_task",
        task_type: str = "test",
        status: TaskStatus = TaskStatus.PENDING,
        progress: float = 0.0,
        current_step: str = "",
        started_at: str = None,
        completed_at: str = None,
        error_message: str = None,
        result_path: str = None,
        pid: int = None,
        command: str = None,
        extra_data: dict = None,
    ):
        self.task_id = task_id
        self.task_type = task_type
        self.status = status
        self.progress = progress
        self.current_step = current_step
        self.started_at = started_at or datetime.now().isoformat()
        self.completed_at = completed_at
        self.error_message = error_message
        self.result_path = result_path
        self.pid = pid
        self.command = command
        self.extra_data = extra_data or {}

    @property
    def is_active(self) -> bool:
        return self.status in (TaskStatus.PENDING, TaskStatus.RUNNING)

    @property
    def is_finished(self) -> bool:
        return self.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED)

    @property
    def elapsed_time(self) -> float:
        if self.started_at:
            start = datetime.fromisoformat(self.started_at)
            return (datetime.now() - start).total_seconds()
        return 0

    @property
    def elapsed_time_str(self) -> str:
        elapsed = self.elapsed_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        return f"{minutes:02d}:{seconds:02d}"


# services.task_managerをモック
mock_task_manager_module = MagicMock()
mock_task_manager_module.TaskStatus = TaskStatus
mock_task_manager_module.TaskInfo = TaskInfo
mock_task_manager_module.TaskManager = MagicMock
sys.modules['services'] = MagicMock()
sys.modules['services.task_manager'] = mock_task_manager_module

# progress_displayモジュールをインポート
import types

progress_display_path = Path(__file__).parent.parent.parent.parent / "app" / "components" / "progress_display.py"
with open(progress_display_path, 'r') as f:
    source = f.read()

# モジュールを作成して実行
progress_display_module = types.ModuleType("progress_display")
progress_display_module.__file__ = str(progress_display_path)
progress_display_module.TaskStatus = TaskStatus
progress_display_module.TaskInfo = TaskInfo
exec(compile(source, progress_display_path, 'exec'), progress_display_module.__dict__)

# テスト対象関数を取得
render_task_progress = progress_display_module.render_task_progress
render_task_list = progress_display_module.render_task_list
render_active_task_banner = progress_display_module.render_active_task_banner
_render_extra_data = progress_display_module._render_extra_data
render_task_metrics = progress_display_module.render_task_metrics
render_circular_progress = progress_display_module.render_circular_progress
render_training_metric_cards = progress_display_module.render_training_metric_cards
render_training_progress_bar = progress_display_module.render_training_progress_bar
render_training_completed_banner = progress_display_module.render_training_completed_banner


# =============================================================================
# TestRenderTaskProgress
# =============================================================================


def create_mock_columns(count):
    """モックカラムを作成するヘルパー"""
    cols = []
    for _ in range(count):
        mock_col = MagicMock()
        mock_col.__enter__ = MagicMock(return_value=mock_col)
        mock_col.__exit__ = MagicMock()
        cols.append(mock_col)
    return cols


def dynamic_columns(*args):
    """st.columnsの動的モック"""
    if isinstance(args[0], int):
        return create_mock_columns(args[0])
    elif isinstance(args[0], list):
        return create_mock_columns(len(args[0]))
    return create_mock_columns(3)


class TestRenderTaskProgress:
    """render_task_progress() のテスト"""

    def setup_method(self):
        """各テスト前にモックをリセット"""
        mock_st.reset_mock()
        # st.columnsが複数回呼ばれるので、side_effectで動的に戻り値を返す
        mock_st.columns.side_effect = dynamic_columns
        mock_st.button.return_value = False
        mock_st.expander.return_value.__enter__ = MagicMock()
        mock_st.expander.return_value.__exit__ = MagicMock()

    def test_task_not_found(self):
        """タスクが見つからない場合、エラー表示"""
        mock_task_manager = MagicMock()
        mock_task_manager.get_task.return_value = None

        result = render_task_progress("nonexistent", mock_task_manager)

        assert result is None
        mock_st.error.assert_called()

    def test_pending_task(self):
        """保留中のタスクが表示される"""
        task = TaskInfo(
            task_id="test_pending",
            status=TaskStatus.PENDING,
            progress=0.0,
            current_step="Waiting..."
        )
        mock_task_manager = MagicMock()
        mock_task_manager.get_task.return_value = task

        result = render_task_progress("test_pending", mock_task_manager, auto_refresh=False)

        assert result is not None
        assert result.task_id == "test_pending"
        mock_st.progress.assert_called()

    def test_running_task(self):
        """実行中のタスクが表示される"""
        task = TaskInfo(
            task_id="test_running",
            status=TaskStatus.RUNNING,
            progress=0.5,
            current_step="Processing..."
        )
        mock_task_manager = MagicMock()
        mock_task_manager.get_task.return_value = task

        result = render_task_progress("test_running", mock_task_manager, auto_refresh=False)

        assert result is not None
        mock_st.progress.assert_called_with(0.5, text="Processing...")

    def test_completed_task(self):
        """完了したタスクが表示される"""
        task = TaskInfo(
            task_id="test_completed",
            status=TaskStatus.COMPLETED,
            progress=1.0,
            current_step="Done",
            result_path="/path/to/result"
        )
        mock_task_manager = MagicMock()
        mock_task_manager.get_task.return_value = task

        result = render_task_progress("test_completed", mock_task_manager, auto_refresh=False)

        assert result is not None
        mock_st.success.assert_called()

    def test_failed_task(self):
        """失敗したタスクが表示される"""
        task = TaskInfo(
            task_id="test_failed",
            status=TaskStatus.FAILED,
            progress=0.3,
            current_step="Failed",
            error_message="Something went wrong"
        )
        mock_task_manager = MagicMock()
        mock_task_manager.get_task.return_value = task

        result = render_task_progress("test_failed", mock_task_manager, auto_refresh=False)

        assert result is not None
        mock_st.error.assert_called()

    def test_show_cancel_button_for_running(self):
        """実行中のタスクにキャンセルボタンが表示される"""
        task = TaskInfo(
            task_id="test_cancel",
            status=TaskStatus.RUNNING,
            progress=0.5,
        )
        mock_task_manager = MagicMock()
        mock_task_manager.get_task.return_value = task

        render_task_progress("test_cancel", mock_task_manager, show_cancel_button=True, auto_refresh=False)

        mock_st.button.assert_called()

    def test_extra_data_displayed(self):
        """extra_dataが詳細として表示される"""
        task = TaskInfo(
            task_id="test_extra",
            status=TaskStatus.RUNNING,
            progress=0.5,
            extra_data={"key": "value"}
        )
        mock_task_manager = MagicMock()
        mock_task_manager.get_task.return_value = task

        render_task_progress("test_extra", mock_task_manager, show_details=True, auto_refresh=False)

        mock_st.expander.assert_called()


# =============================================================================
# TestRenderTaskList
# =============================================================================


class TestRenderTaskList:
    """render_task_list() のテスト"""

    def setup_method(self):
        """各テスト前にモックをリセット"""
        mock_st.reset_mock()
        mock_st.columns.return_value = [MagicMock(), MagicMock(), MagicMock()]
        mock_st.button.return_value = False
        mock_st.expander.return_value.__enter__ = MagicMock()
        mock_st.expander.return_value.__exit__ = MagicMock()

    def test_no_tasks(self):
        """タスクがない場合、info表示"""
        mock_task_manager = MagicMock()
        mock_task_manager.get_recent_tasks.return_value = []

        render_task_list(task_manager=mock_task_manager)

        mock_st.info.assert_called_with("No tasks found")

    def test_with_tasks(self):
        """タスクがある場合、リスト表示"""
        tasks = [
            TaskInfo(task_id="task1", status=TaskStatus.RUNNING, progress=0.5),
            TaskInfo(task_id="task2", status=TaskStatus.COMPLETED, progress=1.0),
        ]
        mock_task_manager = MagicMock()
        mock_task_manager.get_recent_tasks.return_value = tasks

        render_task_list(task_manager=mock_task_manager)

        # expanderが2回呼ばれる
        assert mock_st.expander.call_count == 2

    def test_active_only_filter(self):
        """アクティブタスクのみフィルタ"""
        tasks = [
            TaskInfo(task_id="active", status=TaskStatus.RUNNING, progress=0.5),
        ]
        mock_task_manager = MagicMock()
        mock_task_manager.get_active_tasks.return_value = tasks

        render_task_list(task_manager=mock_task_manager, show_active_only=True)

        mock_task_manager.get_active_tasks.assert_called()


# =============================================================================
# TestRenderActiveTaskBanner
# =============================================================================


class TestRenderActiveTaskBanner:
    """render_active_task_banner() のテスト"""

    def setup_method(self):
        """各テスト前にモックをリセット"""
        mock_st.reset_mock()
        mock_st.columns.return_value = [MagicMock(), MagicMock()]
        mock_st.button.return_value = False
        mock_st.container.return_value.__enter__ = MagicMock()
        mock_st.container.return_value.__exit__ = MagicMock()

    def test_no_active_tasks(self):
        """アクティブタスクがない場合、Noneを返す"""
        mock_task_manager = MagicMock()
        mock_task_manager.get_active_tasks.return_value = []

        result = render_active_task_banner(task_manager=mock_task_manager)

        assert result is None

    def test_with_active_task(self):
        """アクティブタスクがある場合、バナー表示"""
        task = TaskInfo(
            task_id="active_task",
            status=TaskStatus.RUNNING,
            progress=0.5,
            current_step="Processing..."
        )
        mock_task_manager = MagicMock()
        mock_task_manager.get_active_tasks.return_value = [task]

        result = render_active_task_banner(task_manager=mock_task_manager)

        assert result is not None
        assert result.task_id == "active_task"
        mock_st.info.assert_called()
        mock_st.progress.assert_called()


# =============================================================================
# TestRenderExtraData
# =============================================================================


class TestRenderExtraData:
    """_render_extra_data() のテスト"""

    def setup_method(self):
        """各テスト前にモックをリセット"""
        mock_st.reset_mock()

    def test_simple_dict(self):
        """シンプルな辞書が表示される"""
        data = {"key1": "value1", "key2": 42}

        _render_extra_data(data)

        assert mock_st.write.call_count >= 2

    def test_nested_dict(self):
        """ネストした辞書が表示される"""
        data = {"outer": {"inner": "value"}}

        _render_extra_data(data)

        mock_st.write.assert_called()

    def test_float_formatting(self):
        """浮動小数点が適切にフォーマットされる"""
        data = {"score": 0.123456}

        _render_extra_data(data)

        # 0.1235のような形式でフォーマットされる
        mock_st.write.assert_called()

    def test_list_display(self):
        """リストが表示される"""
        data = {"items": ["a", "b", "c"]}

        _render_extra_data(data)

        mock_st.write.assert_called()


# =============================================================================
# TestRenderTaskMetrics
# =============================================================================


class TestRenderTaskMetrics:
    """render_task_metrics() のテスト"""

    def setup_method(self):
        """各テスト前にモックをリセット"""
        mock_st.reset_mock()
        mock_st.columns.return_value = [MagicMock(), MagicMock(), MagicMock(), MagicMock()]

    def test_no_extra_data(self):
        """extra_dataがない場合、何も表示しない"""
        task = TaskInfo(extra_data=None)

        render_task_metrics(task)

        mock_st.subheader.assert_not_called()

    def test_empty_metrics(self):
        """メトリクスがない場合、何も表示しない"""
        task = TaskInfo(extra_data={})

        render_task_metrics(task)

        mock_st.subheader.assert_not_called()

    def test_with_metrics(self):
        """メトリクスがある場合、表示される"""
        task = TaskInfo(extra_data={"mAP50": 0.85, "success_rate": 0.9})

        render_task_metrics(task)

        mock_st.subheader.assert_called_with("Results")
        mock_st.metric.assert_called()


# =============================================================================
# TestRenderCircularProgress
# =============================================================================


class TestRenderCircularProgress:
    """render_circular_progress() のテスト"""

    def setup_method(self):
        """各テスト前にモックをリセット"""
        mock_st.reset_mock()

    def test_progress_display(self):
        """進捗が表示される"""
        render_circular_progress(0.5)

        mock_st.metric.assert_called_with("Progress", "50%")
        mock_st.progress.assert_called_with(0.5)

    def test_custom_label(self):
        """カスタムラベルが使用される"""
        render_circular_progress(0.75, label="Training")

        mock_st.metric.assert_called_with("Training", "75%")


# =============================================================================
# TestRenderTrainingMetricCards
# =============================================================================


class TestRenderTrainingMetricCards:
    """render_training_metric_cards() のテスト"""

    def setup_method(self):
        """各テスト前にモックをリセット"""
        mock_st.reset_mock()
        mock_col = MagicMock()
        mock_col.__enter__ = MagicMock(return_value=mock_col)
        mock_col.__exit__ = MagicMock()
        mock_container = MagicMock()
        mock_container.__enter__ = MagicMock(return_value=mock_container)
        mock_container.__exit__ = MagicMock()
        mock_st.columns.return_value = [mock_col, mock_col, mock_col, mock_col]
        mock_st.container.return_value = mock_container

    def test_metric_cards_rendered(self):
        """メトリクスカードが表示される"""
        task = TaskInfo(
            progress=0.5,
            extra_data={
                "metrics": {"mAP50": 0.7, "loss": 0.5},
                "config": {"epochs": 100},
                "epoch": 50,
            }
        )

        render_training_metric_cards(task)

        mock_st.columns.assert_called()
        mock_st.metric.assert_called()


# =============================================================================
# TestRenderTrainingProgressBar
# =============================================================================


class TestRenderTrainingProgressBar:
    """render_training_progress_bar() のテスト"""

    def setup_method(self):
        """各テスト前にモックをリセット"""
        mock_st.reset_mock()
        mock_col = MagicMock()
        mock_col.__enter__ = MagicMock(return_value=mock_col)
        mock_col.__exit__ = MagicMock()
        mock_st.columns.return_value = [mock_col, mock_col]

    def test_progress_bar_rendered(self):
        """プログレスバーが表示される"""
        render_training_progress_bar(0.5, "Training epoch 50/100")

        mock_st.progress.assert_called_with(0.5)
        mock_st.caption.assert_called()


# =============================================================================
# TestRenderTrainingCompletedBanner
# =============================================================================


class TestRenderTrainingCompletedBanner:
    """render_training_completed_banner() のテスト"""

    def setup_method(self):
        """各テスト前にモックをリセット"""
        mock_st.reset_mock()
        mock_col = MagicMock()
        mock_col.__enter__ = MagicMock(return_value=mock_col)
        mock_col.__exit__ = MagicMock()
        mock_container = MagicMock()
        mock_container.__enter__ = MagicMock(return_value=mock_container)
        mock_container.__exit__ = MagicMock()
        mock_st.columns.return_value = [mock_col, mock_col, mock_col, mock_col]
        mock_st.container.return_value = mock_container

    def test_target_achieved(self):
        """ターゲット達成時のバナー"""
        task = TaskInfo(
            status=TaskStatus.COMPLETED,
            extra_data={
                "metrics": {"mAP50": 0.90, "mAP50-95": 0.75, "precision": 0.85, "recall": 0.80},
                "training_time_minutes": 30,
                "epochs_completed": 100,
            }
        )

        render_training_completed_banner(task)

        mock_st.success.assert_called()

    def test_target_not_achieved(self):
        """ターゲット未達成時のバナー"""
        task = TaskInfo(
            status=TaskStatus.COMPLETED,
            extra_data={
                "metrics": {"mAP50": 0.70, "mAP50-95": 0.55, "precision": 0.65, "recall": 0.60},
                "training_time_minutes": 30,
                "epochs_completed": 100,
            }
        )

        render_training_completed_banner(task)

        mock_st.warning.assert_called()

    def test_best_model_path_displayed(self):
        """最良モデルパスが表示される"""
        task = TaskInfo(
            status=TaskStatus.COMPLETED,
            extra_data={
                "metrics": {"mAP50": 0.90, "mAP50-95": 0.75, "precision": 0.85, "recall": 0.80},
                "best_model": "/path/to/best.pt",
            }
        )

        render_training_completed_banner(task)

        mock_st.code.assert_called()
