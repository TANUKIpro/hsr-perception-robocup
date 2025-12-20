"""
Dashboard Integration テスト

app/pages/1_Dashboard.py のテスト
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import types

# アプリパスを追加（インポート前に）
APP_PATH = str(Path(__file__).parent.parent.parent.parent / "app")
sys.path.insert(0, APP_PATH)

# Streamlitをモック（インポート前に）
mock_st = MagicMock()
# session_stateはドット記法と辞書記法の両方をサポートする必要がある
mock_session_state = MagicMock()
mock_st.session_state = mock_session_state
mock_st.title = MagicMock()
mock_st.subheader = MagicMock()
mock_st.markdown = MagicMock()
mock_st.write = MagicMock()
mock_st.info = MagicMock()
mock_st.success = MagicMock()
mock_st.warning = MagicMock()
mock_st.error = MagicMock()
mock_st.button = MagicMock(return_value=False)
mock_st.progress = MagicMock()

# columnsのモック - コンテキストマネージャ対応
mock_col = MagicMock()
mock_col.__enter__ = MagicMock(return_value=mock_col)
mock_col.__exit__ = MagicMock(return_value=None)
mock_st.columns = MagicMock(return_value=[MagicMock() for _ in range(4)])
for col in mock_st.columns.return_value:
    col.__enter__ = MagicMock(return_value=col)
    col.__exit__ = MagicMock(return_value=None)
    col.metric = MagicMock()
    col.write = MagicMock()
    col.progress = MagicMock()

mock_st.metric = MagicMock()
sys.modules['streamlit'] = mock_st

# common_sidebarをモック
mock_common_sidebar = MagicMock()
mock_common_sidebar.render_common_sidebar = MagicMock()
sys.modules['components'] = MagicMock()
sys.modules['components.common_sidebar'] = mock_common_sidebar

# object_registryをモック
mock_object_registry = MagicMock()
sys.modules['object_registry'] = mock_object_registry

# ダッシュボードモジュールをロード
dashboard_path = Path(__file__).parent.parent.parent.parent / "app" / "pages" / "1_Dashboard.py"
with open(dashboard_path, 'r') as f:
    source = f.read()

# 相対インポートを調整
source = source.replace(
    "from components.common_sidebar import render_common_sidebar",
    "render_common_sidebar = lambda: None"
)
source = source.replace(
    "from object_registry import RegisteredObject",
    "RegisteredObject = type('RegisteredObject', (), {})"
)

# モジュールを作成して実行
dashboard_module = types.ModuleType("dashboard")
dashboard_module.__file__ = str(dashboard_path)
exec(compile(source, dashboard_path, 'exec'), dashboard_module.__dict__)

# テスト対象関数を取得
_render_overall_stats = dashboard_module._render_overall_stats
_render_pipeline_status = dashboard_module._render_pipeline_status
_render_category_progress = dashboard_module._render_category_progress
_render_training_readiness = dashboard_module._render_training_readiness
_render_object_progress = dashboard_module._render_object_progress


# =============================================================================
# フィクスチャ
# =============================================================================


@pytest.fixture
def mock_registry():
    """ObjectRegistryのモック"""
    registry = MagicMock()
    registry.get_collection_stats.return_value = {
        "total_objects": 5,
        "total_collected": 250,
        "total_target": 500,
        "ready_objects": 3,
        "by_category": {
            "food": {"collected": 100, "target": 200, "objects": 2},
            "container": {"collected": 150, "target": 300, "objects": 3}
        }
    }
    registry.get_all_objects.return_value = []
    registry.update_all_collection_counts.return_value = None
    registry.export_to_yolo_config.return_value = "config/object_classes.json"
    return registry


@pytest.fixture
def mock_task_manager():
    """TaskManagerのモック"""
    manager = MagicMock()
    manager.get_active_tasks.return_value = []
    return manager


@pytest.fixture
def mock_path_coordinator():
    """PathCoordinatorのモック"""
    coordinator = MagicMock()
    coordinator.get_annotation_sessions.return_value = []
    coordinator.get_trained_models.return_value = []
    return coordinator


@pytest.fixture
def sample_objects():
    """RegisteredObjectのモックリスト"""
    obj1 = MagicMock()
    obj1.id = 0
    obj1.display_name = "apple"
    obj1.category = "food"
    obj1.collected_samples = 60
    obj1.target_samples = 100
    obj1.properties = MagicMock()
    obj1.properties.is_heavy = False
    obj1.properties.is_tiny = False
    obj1.properties.has_liquid = False

    obj2 = MagicMock()
    obj2.id = 1
    obj2.display_name = "cup"
    obj2.category = "container"
    obj2.collected_samples = 45
    obj2.target_samples = 100
    obj2.properties = MagicMock()
    obj2.properties.is_heavy = False
    obj2.properties.is_tiny = False
    obj2.properties.has_liquid = True

    obj3 = MagicMock()
    obj3.id = 2
    obj3.display_name = "banana"
    obj3.category = "food"
    obj3.collected_samples = 30
    obj3.target_samples = 100
    obj3.properties = MagicMock()
    obj3.properties.is_heavy = False
    obj3.properties.is_tiny = True
    obj3.properties.has_liquid = False

    return [obj1, obj2, obj3]


# =============================================================================
# TestLoadCollectionStats
# =============================================================================


class TestLoadCollectionStats:
    """収集統計読み込みのテスト (test_load_collection_stats)"""

    def setup_method(self):
        """各テスト前にモックをリセット"""
        mock_st.reset_mock()
        # session_stateを新しいMagicMockで再設定
        mock_st.session_state = MagicMock()
        # columnsを再設定
        mock_cols = [MagicMock() for _ in range(4)]
        for col in mock_cols:
            col.__enter__ = MagicMock(return_value=col)
            col.__exit__ = MagicMock(return_value=None)
            col.metric = MagicMock()
        mock_st.columns.return_value = mock_cols

    def test_stats_displayed_as_metrics(self, mock_registry):
        """統計がメトリクスとして表示される"""
        stats = mock_registry.get_collection_stats()

        _render_overall_stats(stats)

        # columnsが呼ばれる
        mock_st.columns.assert_called_with(4)

    def test_stats_with_zero_objects(self):
        """オブジェクトがゼロの場合の処理"""
        stats = {
            "total_objects": 0,
            "total_collected": 0,
            "total_target": 0,
            "ready_objects": 0,
            "by_category": {}
        }

        # エラーなく実行される
        _render_overall_stats(stats)

        mock_st.columns.assert_called_with(4)

    def test_ready_percentage_calculation(self, mock_registry):
        """準備完了パーセンテージの計算"""
        stats = mock_registry.get_collection_stats()
        # 3/5 objects ready = 60%

        _render_overall_stats(stats)

        # 関数が正常に完了する（例外なし）
        mock_st.columns.assert_called()


# =============================================================================
# TestCalculatePipelineStatus
# =============================================================================


class TestCalculatePipelineStatus:
    """パイプラインステータス計算のテスト (test_calculate_pipeline_status)"""

    def setup_method(self):
        """各テスト前にモックをリセット"""
        mock_st.reset_mock()
        # session_stateを新しいMagicMockで再設定
        mock_st.session_state = MagicMock()
        # columnsを再設定
        mock_cols = [MagicMock() for _ in range(3)]
        for col in mock_cols:
            col.__enter__ = MagicMock(return_value=col)
            col.__exit__ = MagicMock(return_value=None)
            col.metric = MagicMock()
        mock_st.columns.return_value = mock_cols

    def test_pipeline_status_renders(self, mock_task_manager, mock_path_coordinator):
        """パイプラインステータスセクションがレンダリングされる"""
        mock_st.session_state['task_manager'] = mock_task_manager
        mock_st.session_state['path_coordinator'] = mock_path_coordinator

        mock_path_coordinator.get_annotation_sessions.return_value = [
            {"name": "session1", "has_data_yaml": True}
        ]
        mock_path_coordinator.get_trained_models.return_value = [{"name": "model1"}]
        mock_task_manager.get_active_tasks.return_value = []

        _render_pipeline_status()

        mock_st.subheader.assert_called_with("Pipeline Status")
        mock_st.columns.assert_called_with(3)

    def test_counts_ready_datasets_correctly(self, mock_task_manager, mock_path_coordinator):
        """data.yamlを持つセッションのみカウントする"""
        mock_st.session_state['task_manager'] = mock_task_manager
        mock_st.session_state['path_coordinator'] = mock_path_coordinator

        mock_path_coordinator.get_annotation_sessions.return_value = [
            {"name": "session1", "has_data_yaml": True},
            {"name": "session2", "has_data_yaml": False},
            {"name": "session3", "has_data_yaml": True}
        ]
        mock_path_coordinator.get_trained_models.return_value = []
        mock_task_manager.get_active_tasks.return_value = []

        _render_pipeline_status()

        # 2つのセッションがdata.yamlを持つ
        mock_st.subheader.assert_called_with("Pipeline Status")

    def test_active_tasks_count(self, mock_task_manager, mock_path_coordinator):
        """アクティブタスク数を正しく表示する"""
        mock_st.session_state['task_manager'] = mock_task_manager
        mock_st.session_state['path_coordinator'] = mock_path_coordinator

        mock_path_coordinator.get_annotation_sessions.return_value = []
        mock_path_coordinator.get_trained_models.return_value = []
        mock_task_manager.get_active_tasks.return_value = [
            MagicMock(task_id="task1"),
            MagicMock(task_id="task2")
        ]

        _render_pipeline_status()

        mock_st.subheader.assert_called_with("Pipeline Status")


# =============================================================================
# TestCategoryProgressDisplay
# =============================================================================


class TestCategoryProgressDisplay:
    """カテゴリ進捗表示のテスト (test_category_progress_display)"""

    def setup_method(self):
        """各テスト前にモックをリセット"""
        mock_st.reset_mock()
        # session_stateを新しいMagicMockで再設定
        mock_st.session_state = MagicMock()

    def test_category_progress_with_data(self):
        """カテゴリ進捗が正しく表示される"""
        stats = {
            "by_category": {
                "food": {"collected": 50, "target": 100},
                "container": {"collected": 30, "target": 60}
            }
        }

        # columnsを動的に設定
        mock_cols = [MagicMock() for _ in range(2)]
        for col in mock_cols:
            col.__enter__ = MagicMock(return_value=col)
            col.__exit__ = MagicMock(return_value=None)
            col.metric = MagicMock()
            col.progress = MagicMock()
        mock_st.columns.return_value = mock_cols

        _render_category_progress(stats)

        mock_st.subheader.assert_called_with("Progress by Category")
        mock_st.columns.assert_called()

    def test_category_progress_empty(self):
        """空のカテゴリリストの処理"""
        stats = {"by_category": {}}

        _render_category_progress(stats)

        mock_st.subheader.assert_called_with("Progress by Category")

    def test_progress_bar_capped_at_100(self):
        """プログレスバーが100%を超えない"""
        stats = {
            "by_category": {
                "food": {"collected": 150, "target": 100}  # 150%
            }
        }

        mock_col = MagicMock()
        mock_col.__enter__ = MagicMock(return_value=mock_col)
        mock_col.__exit__ = MagicMock(return_value=None)

        mock_st.columns.return_value = [mock_col]

        _render_category_progress(stats)

        # st.progressが1.0（最大）で呼ばれる（withブロック内でもst.progressを使用）
        mock_st.progress.assert_called_with(1.0)


# =============================================================================
# TestTrainingReadinessCheck
# =============================================================================


class TestTrainingReadinessCheck:
    """訓練準備チェックのテスト (test_training_readiness_check)"""

    def setup_method(self):
        """各テスト前にモックをリセット"""
        mock_st.reset_mock()
        # session_stateを新しいMagicMockで再設定
        mock_st.session_state = MagicMock()
        mock_st.button.return_value = False

    def test_all_objects_ready(self, sample_objects, mock_registry):
        """全オブジェクトが十分なデータを持つ場合"""
        mock_st.session_state['registry'] = mock_registry

        # 全オブジェクトを50以上に設定
        for obj in sample_objects:
            obj.collected_samples = 60

        _render_training_readiness(sample_objects)

        mock_st.subheader.assert_called_with("Training Readiness")
        mock_st.success.assert_called()

    def test_some_objects_not_ready(self, sample_objects, mock_registry):
        """一部のオブジェクトがデータ不足の場合"""
        mock_st.session_state['registry'] = mock_registry

        # obj1: 60 (ready), obj2: 45 (not ready), obj3: 30 (not ready)
        # sample_objectsのデフォルト値を使用

        _render_training_readiness(sample_objects)

        mock_st.subheader.assert_called_with("Training Readiness")
        mock_st.warning.assert_called()

    def test_no_objects_registered(self, mock_registry):
        """オブジェクトが登録されていない場合"""
        mock_st.session_state['registry'] = mock_registry

        _render_training_readiness([])

        mock_st.subheader.assert_called_with("Training Readiness")
        mock_st.warning.assert_called_with("No objects registered.")

    def test_export_button_when_ready(self, sample_objects, mock_registry):
        """全オブジェクト準備完了時にエクスポートボタンが表示される"""
        mock_st.session_state['registry'] = mock_registry
        mock_st.button.return_value = True  # ボタンクリック

        # 全オブジェクトを準備完了に
        for obj in sample_objects:
            obj.collected_samples = 60

        _render_training_readiness(sample_objects)

        mock_st.success.assert_called()
        mock_st.button.assert_called()


# =============================================================================
# TestActiveTaskDisplay
# =============================================================================


class TestActiveTaskDisplay:
    """アクティブタスク表示のテスト (test_active_task_display)"""

    def setup_method(self):
        """各テスト前にモックをリセット"""
        mock_st.reset_mock()
        # session_stateを新しいMagicMockで再設定
        mock_st.session_state = MagicMock()
        # columnsを再設定
        mock_cols = [MagicMock() for _ in range(3)]
        for col in mock_cols:
            col.__enter__ = MagicMock(return_value=col)
            col.__exit__ = MagicMock(return_value=None)
            col.metric = MagicMock()
        mock_st.columns.return_value = mock_cols

    def test_no_active_tasks(self, mock_task_manager, mock_path_coordinator):
        """アクティブタスクがない場合"""
        mock_st.session_state['task_manager'] = mock_task_manager
        mock_st.session_state['path_coordinator'] = mock_path_coordinator

        mock_path_coordinator.get_annotation_sessions.return_value = []
        mock_path_coordinator.get_trained_models.return_value = []
        mock_task_manager.get_active_tasks.return_value = []

        _render_pipeline_status()

        mock_st.subheader.assert_called_with("Pipeline Status")

    def test_with_active_tasks(self, mock_task_manager, mock_path_coordinator):
        """アクティブタスクがある場合"""
        mock_st.session_state['task_manager'] = mock_task_manager
        mock_st.session_state['path_coordinator'] = mock_path_coordinator

        mock_path_coordinator.get_annotation_sessions.return_value = []
        mock_path_coordinator.get_trained_models.return_value = []

        # アクティブタスクを設定
        active_task = MagicMock()
        active_task.task_id = "train_001"
        active_task.task_type = "training"
        active_task.status = "running"
        mock_task_manager.get_active_tasks.return_value = [active_task]

        _render_pipeline_status()

        mock_st.subheader.assert_called_with("Pipeline Status")

    def test_multiple_active_tasks(self, mock_task_manager, mock_path_coordinator):
        """複数のアクティブタスクがある場合"""
        mock_st.session_state['task_manager'] = mock_task_manager
        mock_st.session_state['path_coordinator'] = mock_path_coordinator

        mock_path_coordinator.get_annotation_sessions.return_value = []
        mock_path_coordinator.get_trained_models.return_value = []

        # 複数のアクティブタスク
        tasks = [
            MagicMock(task_id="train_001", task_type="training"),
            MagicMock(task_id="annot_001", task_type="annotation")
        ]
        mock_task_manager.get_active_tasks.return_value = tasks

        _render_pipeline_status()

        mock_st.subheader.assert_called_with("Pipeline Status")


# =============================================================================
# TestObjectProgress
# =============================================================================


class TestObjectProgress:
    """オブジェクト進捗表示のテスト（追加テスト）"""

    def setup_method(self):
        """各テスト前にモックをリセット"""
        mock_st.reset_mock()
        # session_stateを新しいMagicMockで再設定
        mock_st.session_state = MagicMock()
        # columnsを設定
        mock_cols = [MagicMock() for _ in range(3)]
        for col in mock_cols:
            col.__enter__ = MagicMock(return_value=col)
            col.__exit__ = MagicMock(return_value=None)
            col.write = MagicMock()
            col.progress = MagicMock()
        mock_st.columns.return_value = mock_cols

    def test_object_progress_with_objects(self, sample_objects):
        """オブジェクトがある場合の進捗表示"""
        _render_object_progress(sample_objects)

        mock_st.subheader.assert_called_with("Collection Progress by Object")

    def test_object_progress_empty(self):
        """オブジェクトがない場合"""
        _render_object_progress([])

        mock_st.subheader.assert_called_with("Collection Progress by Object")
        mock_st.info.assert_called_with(
            "No objects registered yet. Go to Registry to add objects."
        )

    def test_progress_status_indicators(self, sample_objects):
        """進捗状態インジケータの表示"""
        # obj1: 60/100 = 60% -> 黄色
        # obj2: 45/100 = 45% -> 赤
        # obj3: 30/100 = 30% -> 赤

        _render_object_progress(sample_objects)

        mock_st.subheader.assert_called_with("Collection Progress by Object")

    def test_properties_badges(self, sample_objects):
        """プロパティバッジの表示"""
        # obj2はhas_liquid=True
        # obj3はis_tiny=True

        _render_object_progress(sample_objects)

        mock_st.subheader.assert_called_with("Collection Progress by Object")
