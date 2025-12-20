"""
TaskManager テスト

app/services/task_manager.py のテスト
"""

import json
import pytest
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, mock_open

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "app"))

from services.task_manager import TaskStatus, TaskInfo, TaskManager, update_task_status


# =============================================================================
# TestTaskStatus
# =============================================================================


class TestTaskStatus:
    """TaskStatus Enum のテスト"""

    def test_enum_values(self):
        """Enum値の確認テスト"""
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.RUNNING.value == "running"
        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.FAILED.value == "failed"
        assert TaskStatus.CANCELLED.value == "cancelled"


# =============================================================================
# TestTaskInfo
# =============================================================================


class TestTaskInfo:
    """TaskInfo データクラスのテスト"""

    def test_to_dict(self):
        """辞書変換テスト"""
        task = TaskInfo(
            task_id="annotation_20240101_120000",
            task_type="annotation",
            status=TaskStatus.RUNNING,
            progress=0.5,
            current_step="Processing images...",
            started_at="2024-01-01T12:00:00",
            pid=12345
        )

        result = task.to_dict()

        assert result["task_id"] == "annotation_20240101_120000"
        assert result["task_type"] == "annotation"
        assert result["status"] == "running"
        assert result["progress"] == 0.5
        assert result["pid"] == 12345

    def test_from_dict(self):
        """辞書から作成テスト"""
        data = {
            "task_id": "training_20240101_140000",
            "task_type": "training",
            "status": "completed",
            "progress": 1.0,
            "current_step": "Done",
            "started_at": "2024-01-01T14:00:00",
            "completed_at": "2024-01-01T15:00:00",
            "error_message": None,
            "result_path": "/path/to/model",
            "pid": None,
            "command": None,
            "extra_data": {"epochs": 100}
        }

        task = TaskInfo.from_dict(data)

        assert task.task_id == "training_20240101_140000"
        assert task.status == TaskStatus.COMPLETED
        assert task.extra_data["epochs"] == 100

    def test_is_active_property(self):
        """アクティブ状態プロパティテスト"""
        pending_task = TaskInfo(
            task_id="t1",
            task_type="test",
            status=TaskStatus.PENDING,
            progress=0.0,
            current_step=""
        )
        running_task = TaskInfo(
            task_id="t2",
            task_type="test",
            status=TaskStatus.RUNNING,
            progress=0.5,
            current_step=""
        )
        completed_task = TaskInfo(
            task_id="t3",
            task_type="test",
            status=TaskStatus.COMPLETED,
            progress=1.0,
            current_step=""
        )

        assert pending_task.is_active is True
        assert running_task.is_active is True
        assert completed_task.is_active is False

    def test_is_finished_property(self):
        """完了状態プロパティテスト"""
        running_task = TaskInfo(
            task_id="t1",
            task_type="test",
            status=TaskStatus.RUNNING,
            progress=0.5,
            current_step=""
        )
        completed_task = TaskInfo(
            task_id="t2",
            task_type="test",
            status=TaskStatus.COMPLETED,
            progress=1.0,
            current_step=""
        )
        failed_task = TaskInfo(
            task_id="t3",
            task_type="test",
            status=TaskStatus.FAILED,
            progress=0.3,
            current_step=""
        )
        cancelled_task = TaskInfo(
            task_id="t4",
            task_type="test",
            status=TaskStatus.CANCELLED,
            progress=0.2,
            current_step=""
        )

        assert running_task.is_finished is False
        assert completed_task.is_finished is True
        assert failed_task.is_finished is True
        assert cancelled_task.is_finished is True

    def test_elapsed_time_calculation(self):
        """経過時間計算テスト"""
        start_time = datetime.now() - timedelta(minutes=5)
        task = TaskInfo(
            task_id="t1",
            task_type="test",
            status=TaskStatus.RUNNING,
            progress=0.5,
            current_step="",
            started_at=start_time.isoformat()
        )

        elapsed = task.elapsed_time
        assert elapsed is not None
        assert 299 < elapsed < 310  # 約5分

    def test_elapsed_time_str_format(self):
        """経過時間文字列フォーマットテスト"""
        start_time = datetime.now() - timedelta(minutes=3, seconds=30)
        task = TaskInfo(
            task_id="t1",
            task_type="test",
            status=TaskStatus.RUNNING,
            progress=0.5,
            current_step="",
            started_at=start_time.isoformat()
        )

        elapsed_str = task.elapsed_time_str
        assert "03:" in elapsed_str  # 3分台


# =============================================================================
# TestTaskManager
# =============================================================================


class TestTaskManager:
    """TaskManager クラスのテスト"""

    @pytest.fixture
    def mock_path_coordinator(self, tmp_path):
        """PathCoordinatorのモック"""
        coordinator = MagicMock()
        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir(parents=True)
        coordinator.get_path.return_value = tasks_dir
        coordinator.project_root = tmp_path
        return coordinator

    def test_initialization(self, mock_path_coordinator):
        """初期化テスト"""
        manager = TaskManager(path_coordinator=mock_path_coordinator)

        assert manager.tasks_dir.exists()

    def test_generate_task_id(self, mock_path_coordinator):
        """タスクID生成テスト"""
        manager = TaskManager(path_coordinator=mock_path_coordinator)

        task_id = manager._generate_task_id("annotation")

        assert task_id.startswith("annotation_")
        assert len(task_id) > len("annotation_")

    def test_save_and_load_status(self, mock_path_coordinator):
        """ステータス保存と読み込みテスト"""
        manager = TaskManager(path_coordinator=mock_path_coordinator)

        task = TaskInfo(
            task_id="test_task",
            task_type="test",
            status=TaskStatus.RUNNING,
            progress=0.5,
            current_step="Testing..."
        )

        manager._save_status(task)
        loaded = manager._load_status("test_task")

        assert loaded is not None
        assert loaded.task_id == "test_task"
        assert loaded.status == TaskStatus.RUNNING
        assert loaded.progress == 0.5

    def test_get_task(self, mock_path_coordinator):
        """タスク取得テスト"""
        manager = TaskManager(path_coordinator=mock_path_coordinator)

        task = TaskInfo(
            task_id="get_test",
            task_type="test",
            status=TaskStatus.COMPLETED,
            progress=1.0,
            current_step="Done"
        )
        manager._save_status(task)

        retrieved = manager.get_task("get_test")

        assert retrieved is not None
        assert retrieved.task_id == "get_test"

    def test_get_task_not_found(self, mock_path_coordinator):
        """タスク未発見テスト"""
        manager = TaskManager(path_coordinator=mock_path_coordinator)

        result = manager.get_task("nonexistent")

        assert result is None

    def test_get_all_tasks(self, mock_path_coordinator):
        """全タスク取得テスト"""
        manager = TaskManager(path_coordinator=mock_path_coordinator)

        # 複数タスクを作成
        for i in range(3):
            task = TaskInfo(
                task_id=f"task_{i}",
                task_type="test",
                status=TaskStatus.COMPLETED,
                progress=1.0,
                current_step="Done",
                started_at=datetime.now().isoformat()
            )
            manager._save_status(task)

        tasks = manager.get_all_tasks()

        assert len(tasks) == 3

    def test_get_active_tasks(self, mock_path_coordinator):
        """アクティブタスク取得テスト"""
        manager = TaskManager(path_coordinator=mock_path_coordinator)

        # 完了タスク
        completed = TaskInfo(
            task_id="completed",
            task_type="test",
            status=TaskStatus.COMPLETED,
            progress=1.0,
            current_step="Done",
            started_at=datetime.now().isoformat()
        )
        manager._save_status(completed)

        # 実行中タスク
        running = TaskInfo(
            task_id="running",
            task_type="test",
            status=TaskStatus.RUNNING,
            progress=0.5,
            current_step="Processing...",
            started_at=datetime.now().isoformat()
        )
        manager._save_status(running)

        active = manager.get_active_tasks()

        assert len(active) == 1
        assert active[0].task_id == "running"

    def test_get_recent_tasks(self, mock_path_coordinator):
        """最近のタスク取得テスト"""
        manager = TaskManager(path_coordinator=mock_path_coordinator)

        for i in range(5):
            task = TaskInfo(
                task_id=f"task_{i}",
                task_type="test",
                status=TaskStatus.COMPLETED,
                progress=1.0,
                current_step="Done",
                started_at=datetime.now().isoformat()
            )
            manager._save_status(task)

        recent = manager.get_recent_tasks(limit=3)

        assert len(recent) == 3

    @patch('os.kill')
    @patch('os.killpg')
    @patch('os.getpgid')
    def test_cancel_task(self, mock_getpgid, mock_killpg, mock_kill, mock_path_coordinator):
        """タスクキャンセルテスト"""
        # os.kill(pid, 0) が成功するようにモック（プロセスが存在する）
        mock_kill.return_value = None
        mock_getpgid.return_value = 99999
        mock_killpg.return_value = None

        manager = TaskManager(path_coordinator=mock_path_coordinator)

        task = TaskInfo(
            task_id="to_cancel",
            task_type="test",
            status=TaskStatus.RUNNING,
            progress=0.5,
            current_step="Processing...",
            pid=99999
        )
        manager._save_status(task)

        result = manager.cancel_task("to_cancel")

        assert result is True
        cancelled = manager.get_task("to_cancel")
        assert cancelled.status == TaskStatus.CANCELLED

    def test_cancel_nonexistent_task(self, mock_path_coordinator):
        """存在しないタスクキャンセルテスト"""
        manager = TaskManager(path_coordinator=mock_path_coordinator)

        result = manager.cancel_task("nonexistent")

        assert result is False

    def test_delete_task(self, mock_path_coordinator):
        """タスク削除テスト"""
        manager = TaskManager(path_coordinator=mock_path_coordinator)

        task = TaskInfo(
            task_id="to_delete",
            task_type="test",
            status=TaskStatus.COMPLETED,
            progress=1.0,
            current_step="Done"
        )
        manager._save_status(task)

        result = manager.delete_task("to_delete")

        assert result is True
        assert manager.get_task("to_delete") is None

    def test_cleanup_old_tasks(self, mock_path_coordinator, tmp_path):
        """古いタスクのクリーンアップテスト"""
        manager = TaskManager(path_coordinator=mock_path_coordinator)

        # 古いタスクを作成（ファイルタイムスタンプを操作）
        old_task = TaskInfo(
            task_id="old_task",
            task_type="test",
            status=TaskStatus.COMPLETED,
            progress=1.0,
            current_step="Done"
        )
        manager._save_status(old_task)

        # ファイルの更新日時を2日前に変更
        import os
        import time
        old_time = time.time() - (48 * 3600)
        status_file = manager.tasks_dir / "old_task.json"
        os.utime(status_file, (old_time, old_time))

        deleted = manager.cleanup_old_tasks(max_age_hours=24)

        assert deleted == 1


# =============================================================================
# TestTaskLaunchers
# =============================================================================


class TestTaskLaunchers:
    """タスク起動メソッドのテスト"""

    @pytest.fixture
    def mock_path_coordinator(self, tmp_path):
        """PathCoordinatorのモック"""
        coordinator = MagicMock()
        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir(parents=True)
        coordinator.get_path.return_value = tasks_dir
        coordinator.project_root = tmp_path
        return coordinator

    @patch('subprocess.Popen')
    def test_start_annotation(self, mock_popen, mock_path_coordinator, tmp_path):
        """アノテーション開始テスト"""
        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_popen.return_value = mock_process

        # ランナースクリプトを作成
        runner_dir = tmp_path / "app" / "services" / "task_runners"
        runner_dir.mkdir(parents=True)
        (runner_dir / "run_annotation.py").write_text("# dummy")

        manager = TaskManager(path_coordinator=mock_path_coordinator)

        task_id = manager.start_annotation(
            method="background",
            input_dir="/input",
            output_dir="/output",
            class_config="/config/classes.json",
            background_path="/bg.jpg"
        )

        assert task_id.startswith("annotation_")
        assert mock_popen.called

    def test_start_annotation_requires_background(self, mock_path_coordinator):
        """アノテーションには背景が必要テスト"""
        manager = TaskManager(path_coordinator=mock_path_coordinator)

        with pytest.raises(ValueError, match="Background path required"):
            manager.start_annotation(
                method="background",
                input_dir="/input",
                output_dir="/output",
                class_config="/config/classes.json"
                # background_path なし
            )

    @patch('subprocess.Popen')
    def test_start_training(self, mock_popen, mock_path_coordinator, tmp_path):
        """訓練開始テスト"""
        mock_process = MagicMock()
        mock_process.pid = 12346
        mock_popen.return_value = mock_process

        # ランナースクリプトを作成
        runner_dir = tmp_path / "app" / "services" / "task_runners"
        runner_dir.mkdir(parents=True)
        (runner_dir / "run_training.py").write_text("# dummy")

        manager = TaskManager(path_coordinator=mock_path_coordinator)

        task_id = manager.start_training(
            dataset_yaml="/data.yaml",
            epochs=50,
            batch_size=16
        )

        assert task_id.startswith("training_")
        assert mock_popen.called

    @patch('subprocess.Popen')
    def test_start_evaluation(self, mock_popen, mock_path_coordinator, tmp_path):
        """評価開始テスト"""
        mock_process = MagicMock()
        mock_process.pid = 12347
        mock_popen.return_value = mock_process

        # ランナースクリプトを作成
        runner_dir = tmp_path / "app" / "services" / "task_runners"
        runner_dir.mkdir(parents=True)
        (runner_dir / "run_evaluation.py").write_text("# dummy")

        manager = TaskManager(path_coordinator=mock_path_coordinator)

        task_id = manager.start_evaluation(
            model_path="/model/best.pt",
            dataset_yaml="/data.yaml"
        )

        assert task_id.startswith("evaluation_")
        assert mock_popen.called


# =============================================================================
# TestUpdateTaskStatus
# =============================================================================


class TestUpdateTaskStatus:
    """update_task_status ヘルパー関数のテスト"""

    def test_update_progress(self, tmp_path):
        """進捗更新テスト"""
        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir()

        # 初期タスクを作成
        task_data = {
            "task_id": "test_update",
            "task_type": "test",
            "status": "running",
            "progress": 0.0,
            "current_step": "Starting...",
            "started_at": datetime.now().isoformat(),
            "completed_at": None,
            "error_message": None,
            "result_path": None,
            "pid": None,
            "command": None,
            "extra_data": {}
        }
        status_file = tasks_dir / "test_update.json"
        with open(status_file, "w") as f:
            json.dump(task_data, f)

        update_task_status(
            task_id="test_update",
            progress=0.5,
            tasks_dir=str(tasks_dir)
        )

        with open(status_file) as f:
            updated = json.load(f)

        assert updated["progress"] == 0.5

    def test_update_status(self, tmp_path):
        """ステータス更新テスト"""
        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir()

        task_data = {
            "task_id": "test_status",
            "task_type": "test",
            "status": "running",
            "progress": 0.5,
            "current_step": "Working...",
            "started_at": datetime.now().isoformat(),
            "completed_at": None,
            "error_message": None,
            "result_path": None,
            "pid": None,
            "command": None,
            "extra_data": {}
        }
        status_file = tasks_dir / "test_status.json"
        with open(status_file, "w") as f:
            json.dump(task_data, f)

        update_task_status(
            task_id="test_status",
            status="completed",
            progress=1.0,
            tasks_dir=str(tasks_dir)
        )

        with open(status_file) as f:
            updated = json.load(f)

        assert updated["status"] == "completed"
        assert updated["completed_at"] is not None

    def test_update_error_message(self, tmp_path):
        """エラーメッセージ更新テスト"""
        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir()

        task_data = {
            "task_id": "test_error",
            "task_type": "test",
            "status": "running",
            "progress": 0.3,
            "current_step": "Processing...",
            "started_at": datetime.now().isoformat(),
            "completed_at": None,
            "error_message": None,
            "result_path": None,
            "pid": None,
            "command": None,
            "extra_data": {}
        }
        status_file = tasks_dir / "test_error.json"
        with open(status_file, "w") as f:
            json.dump(task_data, f)

        update_task_status(
            task_id="test_error",
            status="failed",
            error_message="Something went wrong",
            tasks_dir=str(tasks_dir)
        )

        with open(status_file) as f:
            updated = json.load(f)

        assert updated["status"] == "failed"
        assert updated["error_message"] == "Something went wrong"
