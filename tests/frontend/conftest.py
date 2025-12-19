"""
フロントエンドテスト用フィクスチャ

Streamlit、プロファイル管理、UIコンポーネントのテスト用モックを提供します。
"""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import json
import tempfile
import shutil


@pytest.fixture
def mock_streamlit():
    """Streamlitモジュールのモック

    Streamlitのセッション状態とUI関数をモックします。
    """
    st_mock = MagicMock()

    # セッション状態のモック
    st_mock.session_state = {}

    # キャッシュデコレータのモック（何もしない）
    st_mock.cache_data = lambda **kwargs: lambda f: f
    st_mock.cache_resource = lambda **kwargs: lambda f: f

    # UI要素のモック
    st_mock.title = MagicMock()
    st_mock.header = MagicMock()
    st_mock.subheader = MagicMock()
    st_mock.write = MagicMock()
    st_mock.markdown = MagicMock()
    st_mock.text = MagicMock()
    st_mock.text_input = MagicMock(return_value="")
    st_mock.text_area = MagicMock(return_value="")
    st_mock.number_input = MagicMock(return_value=0)
    st_mock.selectbox = MagicMock(return_value=None)
    st_mock.multiselect = MagicMock(return_value=[])
    st_mock.checkbox = MagicMock(return_value=False)
    st_mock.radio = MagicMock(return_value=None)
    st_mock.slider = MagicMock(return_value=0)
    st_mock.button = MagicMock(return_value=False)
    st_mock.file_uploader = MagicMock(return_value=None)
    st_mock.image = MagicMock()
    st_mock.progress = MagicMock()
    st_mock.spinner = MagicMock()
    st_mock.success = MagicMock()
    st_mock.error = MagicMock()
    st_mock.warning = MagicMock()
    st_mock.info = MagicMock()
    st_mock.toast = MagicMock()

    # レイアウトのモック
    st_mock.columns = MagicMock(return_value=[MagicMock() for _ in range(4)])
    st_mock.container = MagicMock()
    st_mock.expander = MagicMock()
    st_mock.tabs = MagicMock(return_value=[MagicMock() for _ in range(3)])
    st_mock.sidebar = MagicMock()

    # コンテキストマネージャのモック
    st_mock.container.return_value.__enter__ = MagicMock()
    st_mock.container.return_value.__exit__ = MagicMock()
    st_mock.expander.return_value.__enter__ = MagicMock()
    st_mock.expander.return_value.__exit__ = MagicMock()
    st_mock.spinner.return_value.__enter__ = MagicMock()
    st_mock.spinner.return_value.__exit__ = MagicMock()

    with patch.dict("sys.modules", {"streamlit": st_mock}):
        yield st_mock


@pytest.fixture
def temp_profile_dir(tmp_path: Path) -> Path:
    """テスト用のプロファイルディレクトリ構造を作成"""
    profiles_dir = tmp_path / "profiles"
    profile_dir = profiles_dir / "prof_1"

    # 必要なサブディレクトリを作成
    subdirs = [
        "app_data",
        "datasets",
        "models/trained",
        "models/pretrained",
        "raw_captures",
        "backgrounds",
        "annotation_sessions",
    ]
    for subdir in subdirs:
        (profile_dir / subdir).mkdir(parents=True)

    # profiles.json を作成
    profiles_meta = {
        "profiles": [
            {
                "id": "prof_1",
                "name": "Test Profile",
                "created_at": "2024-01-01T00:00:00",
                "updated_at": "2024-01-01T00:00:00",
            }
        ],
        "active_profile_id": "prof_1",
    }
    with open(profiles_dir / "profiles.json", "w") as f:
        json.dump(profiles_meta, f)

    return profiles_dir


@pytest.fixture
def mock_profile_manager(temp_profile_dir: Path):
    """ProfileManagerのモック"""
    manager = MagicMock()
    manager.get_active_profile_id.return_value = "prof_1"
    manager.get_profile_path.return_value = temp_profile_dir / "prof_1"
    manager.get_profiles_dir.return_value = temp_profile_dir
    manager.get_all_profiles.return_value = [
        {
            "id": "prof_1",
            "name": "Test Profile",
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00",
        }
    ]
    return manager


@pytest.fixture
def mock_path_coordinator(temp_profile_dir: Path):
    """PathCoordinatorのモック"""
    coordinator = MagicMock()
    profile_path = temp_profile_dir / "prof_1"

    coordinator.get_path.side_effect = lambda key, profile_specific=True: {
        "raw_captures": profile_path / "raw_captures",
        "datasets": profile_path / "datasets",
        "trained_models": profile_path / "models" / "trained",
        "pretrained_models": profile_path / "models" / "pretrained",
        "backgrounds": profile_path / "backgrounds",
        "annotation_sessions": profile_path / "annotation_sessions",
        "app_data": profile_path / "app_data",
    }.get(key, profile_path / key)

    return coordinator


@pytest.fixture
def mock_task_manager():
    """TaskManagerのモック"""
    manager = MagicMock()
    manager.get_all_tasks.return_value = []
    manager.get_active_tasks.return_value = []
    manager.get_task.return_value = None
    return manager


@pytest.fixture
def sample_object_registry_data() -> dict:
    """テスト用のオブジェクトレジストリデータ"""
    return {
        "categories": ["food", "container", "toy"],
        "objects": [
            {
                "id": 0,
                "name": "apple",
                "category": "food",
                "properties": {"heavy": False, "tiny": False, "liquid": False},
                "versions": [],
                "collected_count": 10,
            },
            {
                "id": 1,
                "name": "cup",
                "category": "container",
                "properties": {"heavy": False, "tiny": False, "liquid": False},
                "versions": [],
                "collected_count": 5,
            },
        ],
    }


@pytest.fixture
def sample_task_info() -> dict:
    """テスト用のタスク情報"""
    return {
        "task_id": "task_001",
        "task_type": "training",
        "status": "running",
        "progress": 50,
        "start_time": "2024-01-01T10:00:00",
        "end_time": None,
        "error_message": None,
        "params": {"epochs": 100, "batch_size": 16},
    }


@pytest.fixture
def sample_ui_settings() -> dict:
    """テスト用のUI設定"""
    return {
        "training": {
            "epochs": 100,
            "batch_size": 16,
            "imgsz": 640,
            "model": "yolov8m.pt",
        },
        "annotation": {
            "method": "background_subtraction",
            "threshold": 30,
        },
        "evaluation": {
            "conf_threshold": 0.25,
            "iou_threshold": 0.45,
        },
    }


@pytest.fixture
def mock_ros2():
    """ROS2モジュールのモック"""
    rclpy_mock = MagicMock()
    rclpy_mock.init.return_value = None
    rclpy_mock.shutdown.return_value = None
    rclpy_mock.ok.return_value = True

    node_mock = MagicMock()
    node_mock.get_logger.return_value = MagicMock()

    with patch.dict(
        "sys.modules",
        {
            "rclpy": rclpy_mock,
            "rclpy.node": MagicMock(),
            "sensor_msgs": MagicMock(),
            "cv_bridge": MagicMock(),
        },
    ):
        yield rclpy_mock
