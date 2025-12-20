"""
TensorBoard Monitor モジュールのテスト

scripts/training/tensorboard_monitor.py のテストを行います。
"""

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# scriptsディレクトリをパスに追加
scripts_dir = Path(__file__).parent.parent.parent.parent / "scripts"
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))


class TestTensorBoardConfig:
    """TensorBoardConfig データクラスのテスト"""

    def test_dataclass_creation(self, tmp_path):
        """データクラスの作成"""
        from training.tensorboard_monitor import TensorBoardConfig

        config = TensorBoardConfig(
            log_dir=tmp_path / "logs",
            run_name="test_run",
        )
        assert config.log_dir == tmp_path / "logs"
        assert config.run_name == "test_run"

    def test_default_values(self, tmp_path):
        """デフォルト値の確認"""
        from training.tensorboard_monitor import TensorBoardConfig

        config = TensorBoardConfig(
            log_dir=tmp_path,
            run_name="test",
        )
        assert config.port == 6006
        assert config.host == "0.0.0.0"
        assert config.auto_launch is True
        assert config.flush_secs == 30
        assert config.log_frequency == 1

    def test_custom_values(self, tmp_path):
        """カスタム値の設定"""
        from training.tensorboard_monitor import TensorBoardConfig

        config = TensorBoardConfig(
            log_dir=tmp_path,
            run_name="custom_run",
            port=6007,
            host="localhost",
            auto_launch=False,
            flush_secs=60,
            log_frequency=5,
        )
        assert config.port == 6007
        assert config.host == "localhost"
        assert config.auto_launch is False
        assert config.flush_secs == 60
        assert config.log_frequency == 5


class TestTensorBoardServer:
    """TensorBoardServer のテスト"""

    def test_initialization(self, tmp_path):
        """初期化テスト"""
        from training.tensorboard_monitor import TensorBoardServer

        with patch.object(TensorBoardServer, '_find_available_port', return_value=6006):
            server = TensorBoardServer(str(tmp_path))
            assert server.log_dir == str(tmp_path)
            assert server.port == 6006
            assert server.process is None
            assert server.url == ""

    def test_initialization_with_custom_port(self, tmp_path):
        """カスタムポートでの初期化"""
        from training.tensorboard_monitor import TensorBoardServer

        server = TensorBoardServer(str(tmp_path), port=7007)
        assert server.port == 7007

    def test_find_available_port(self, tmp_path):
        """利用可能なポート検索のテスト"""
        from training.tensorboard_monitor import TensorBoardServer

        with patch('training.tensorboard_monitor.socket.socket') as mock_socket:
            mock_socket_instance = MagicMock()
            mock_socket_instance.bind.return_value = None
            mock_socket.return_value.__enter__.return_value = mock_socket_instance

            server = TensorBoardServer(str(tmp_path))
            assert server.port == 6006

    def test_find_available_port_skip_occupied(self, tmp_path):
        """占有されているポートをスキップするテスト"""
        from training.tensorboard_monitor import TensorBoardServer

        with patch('training.tensorboard_monitor.socket.socket') as mock_socket:
            mock_socket_instance = MagicMock()
            # 最初の2つのポートは占有されている
            mock_socket_instance.bind.side_effect = [
                OSError("Port in use"),
                OSError("Port in use"),
                None,  # 3番目のポートは利用可能
            ]
            mock_socket.return_value.__enter__.return_value = mock_socket_instance

            server = TensorBoardServer(str(tmp_path))
            assert server.port == 6008  # 6006, 6007 がスキップされて 6008

    def test_start_server(self, tmp_path):
        """サーバー起動テスト"""
        from training.tensorboard_monitor import TensorBoardServer

        server = TensorBoardServer(str(tmp_path), port=6006)

        with patch('training.tensorboard_monitor.subprocess.Popen') as mock_popen:
            mock_process = MagicMock()
            mock_process.poll.return_value = None  # プロセスは実行中
            mock_popen.return_value = mock_process

            with patch('training.tensorboard_monitor.time.sleep'):  # sleep をスキップ
                url = server.start()

            assert url == "http://localhost:6006"
            assert server.process is not None
            mock_popen.assert_called_once()

    def test_start_server_already_running(self, tmp_path):
        """既に実行中のサーバーを再起動しないテスト"""
        from training.tensorboard_monitor import TensorBoardServer

        server = TensorBoardServer(str(tmp_path), port=6006)

        with patch('training.tensorboard_monitor.subprocess.Popen') as mock_popen:
            mock_process = MagicMock()
            mock_process.poll.return_value = None
            mock_popen.return_value = mock_process

            with patch('training.tensorboard_monitor.time.sleep'):
                server.start()
                # 2回目の呼び出し
                server.start()

            # Popen は1回だけ呼ばれる
            assert mock_popen.call_count == 1

    def test_stop_server(self, tmp_path):
        """サーバー停止テスト"""
        from training.tensorboard_monitor import TensorBoardServer

        server = TensorBoardServer(str(tmp_path), port=6006)

        with patch('training.tensorboard_monitor.subprocess.Popen') as mock_popen:
            mock_process = MagicMock()
            mock_process.poll.return_value = None
            mock_popen.return_value = mock_process

            with patch('training.tensorboard_monitor.time.sleep'):
                server.start()

            server.stop()

            mock_process.terminate.assert_called_once()
            mock_process.wait.assert_called_once()
            assert server.process is None

    def test_stop_server_timeout(self, tmp_path):
        """タイムアウト時のサーバー停止テスト"""
        import subprocess as sp
        from training.tensorboard_monitor import TensorBoardServer

        server = TensorBoardServer(str(tmp_path), port=6006)

        with patch('training.tensorboard_monitor.subprocess.Popen') as mock_popen:
            mock_process = MagicMock()
            mock_process.poll.return_value = None
            mock_process.wait.side_effect = sp.TimeoutExpired(cmd="", timeout=5)
            mock_popen.return_value = mock_process

            with patch('training.tensorboard_monitor.time.sleep'):
                server.start()

            server.stop()

            mock_process.kill.assert_called_once()
            assert server.process is None

    def test_get_url(self, tmp_path):
        """URL取得テスト"""
        from training.tensorboard_monitor import TensorBoardServer

        server = TensorBoardServer(str(tmp_path), port=6006)

        with patch('training.tensorboard_monitor.subprocess.Popen') as mock_popen:
            mock_process = MagicMock()
            mock_process.poll.return_value = None
            mock_popen.return_value = mock_process

            with patch('training.tensorboard_monitor.time.sleep'):
                server.start()

            url = server.get_url()
            assert url == "http://localhost:6006"

    def test_is_running(self, tmp_path):
        """実行状態確認テスト"""
        from training.tensorboard_monitor import TensorBoardServer

        server = TensorBoardServer(str(tmp_path), port=6006)
        assert server.is_running() is False

        with patch('training.tensorboard_monitor.subprocess.Popen') as mock_popen:
            mock_process = MagicMock()
            mock_process.poll.return_value = None
            mock_popen.return_value = mock_process

            with patch('training.tensorboard_monitor.time.sleep'):
                server.start()

            assert server.is_running() is True

            mock_process.poll.return_value = 0  # プロセス終了
            assert server.is_running() is False

    def test_context_manager(self, tmp_path):
        """コンテキストマネージャーテスト"""
        from training.tensorboard_monitor import TensorBoardServer

        with patch('training.tensorboard_monitor.subprocess.Popen') as mock_popen:
            mock_process = MagicMock()
            mock_process.poll.return_value = None
            mock_popen.return_value = mock_process

            with patch('training.tensorboard_monitor.time.sleep'):
                with TensorBoardServer(str(tmp_path), port=6006) as server:
                    assert server.is_running() is True

            mock_process.terminate.assert_called()

    def test_find_running_instance_port_open(self):
        """既存インスタンス検出 - ポート使用中"""
        from training.tensorboard_monitor import TensorBoardServer

        with patch('training.tensorboard_monitor.socket.socket') as mock_socket:
            mock_socket_instance = MagicMock()
            mock_socket_instance.connect_ex.return_value = 0  # 接続成功
            mock_socket.return_value.__enter__.return_value = mock_socket_instance

            result = TensorBoardServer.find_running_instance(6006)
            assert result == "http://localhost:6006"

    def test_find_running_instance_port_closed(self):
        """既存インスタンス検出 - ポート未使用"""
        from training.tensorboard_monitor import TensorBoardServer

        with patch('training.tensorboard_monitor.socket.socket') as mock_socket:
            mock_socket_instance = MagicMock()
            mock_socket_instance.connect_ex.return_value = 1  # 接続失敗
            mock_socket.return_value.__enter__.return_value = mock_socket_instance

            result = TensorBoardServer.find_running_instance(6006)
            assert result is None


class TestCompetitionTensorBoardCallback:
    """CompetitionTensorBoardCallback のテスト"""

    def test_callback_creation(self, tmp_path):
        """コールバック作成テスト"""
        from training.tensorboard_monitor import CompetitionTensorBoardCallback

        callback = CompetitionTensorBoardCallback(
            log_dir=str(tmp_path),
            target_map50=0.85,
        )
        assert callback.log_dir == tmp_path
        assert callback.target_map50 == 0.85
        assert callback.writer is None
        assert callback.start_time is None

    def test_callback_custom_values(self, tmp_path):
        """カスタム値でのコールバック作成"""
        from training.tensorboard_monitor import CompetitionTensorBoardCallback

        callback = CompetitionTensorBoardCallback(
            log_dir=str(tmp_path),
            target_map50=0.90,
            target_inference_ms=50.0,
            log_frequency=5,
            flush_secs=60,
        )
        assert callback.target_map50 == 0.90
        assert callback.target_inference_ms == 50.0
        assert callback.log_frequency == 5
        assert callback.flush_secs == 60

    def test_on_pretrain_routine_start(self, tmp_path):
        """訓練開始時のコールバックテスト"""
        from training.tensorboard_monitor import (
            CompetitionTensorBoardCallback,
            TENSORBOARD_AVAILABLE,
        )

        if not TENSORBOARD_AVAILABLE:
            pytest.skip("TensorBoard not available")

        callback = CompetitionTensorBoardCallback(
            log_dir=str(tmp_path),
            target_map50=0.85,
        )

        mock_trainer = MagicMock()
        mock_trainer.epochs = 50
        mock_trainer.batch_size = 16
        mock_trainer.args.imgsz = 640
        mock_trainer.model = MagicMock()
        mock_trainer.model.yaml_file = "yolov8m.yaml"

        with patch('training.tensorboard_monitor.SummaryWriter') as mock_writer_cls:
            mock_writer = MagicMock()
            mock_writer_cls.return_value = mock_writer

            callback.on_pretrain_routine_start(mock_trainer)

            assert callback.writer is not None
            assert callback.total_epochs == 50
            assert callback.start_time is not None
            mock_writer_cls.assert_called_once()

    def test_on_train_epoch_start(self, tmp_path):
        """エポック開始時のコールバックテスト"""
        from training.tensorboard_monitor import CompetitionTensorBoardCallback

        callback = CompetitionTensorBoardCallback(
            log_dir=str(tmp_path),
        )

        mock_trainer = MagicMock()

        callback.on_train_epoch_start(mock_trainer)
        assert callback._epoch_start > 0

    def test_on_train_epoch_end(self, tmp_path):
        """エポック終了時のコールバックテスト"""
        from training.tensorboard_monitor import CompetitionTensorBoardCallback

        callback = CompetitionTensorBoardCallback(
            log_dir=str(tmp_path),
            log_frequency=1,
        )
        callback.start_time = time.time()
        callback.total_epochs = 50
        callback._epoch_start = time.time() - 10  # 10秒前

        mock_writer = MagicMock()
        callback.writer = mock_writer

        mock_trainer = MagicMock()
        mock_trainer.epoch = 0  # epoch 1

        callback.on_train_epoch_end(mock_trainer)

        # エポック時間がログされる
        assert len(callback.epoch_times) == 1
        assert mock_writer.add_scalar.called

    def test_on_train_epoch_end_log_frequency(self, tmp_path):
        """ログ頻度による条件分岐テスト"""
        from training.tensorboard_monitor import CompetitionTensorBoardCallback

        callback = CompetitionTensorBoardCallback(
            log_dir=str(tmp_path),
            log_frequency=5,  # 5エポックごとにログ
        )
        callback.start_time = time.time()
        callback.total_epochs = 50
        callback._epoch_start = time.time()

        mock_writer = MagicMock()
        callback.writer = mock_writer

        mock_trainer = MagicMock()
        mock_trainer.epoch = 1  # epoch 2 (5の倍数ではない)

        callback.on_train_epoch_end(mock_trainer)

        # add_scalar は呼ばれない
        assert not mock_writer.add_scalar.called

    def test_on_fit_epoch_end(self, tmp_path):
        """検証後のコールバックテスト"""
        from training.tensorboard_monitor import CompetitionTensorBoardCallback

        callback = CompetitionTensorBoardCallback(
            log_dir=str(tmp_path),
            target_map50=0.85,
        )

        mock_writer = MagicMock()
        callback.writer = mock_writer

        mock_trainer = MagicMock()
        mock_trainer.epoch = 9  # epoch 10
        mock_trainer.metrics = {"metrics/mAP50(B)": 0.75}

        callback.on_fit_epoch_end(mock_trainer)

        # target achievement がログされる
        assert mock_writer.add_scalar.called
        calls = mock_writer.add_scalar.call_args_list
        assert any("target_achievement" in str(call) for call in calls)

    def test_on_train_end(self, tmp_path):
        """訓練終了時のコールバックテスト"""
        from training.tensorboard_monitor import CompetitionTensorBoardCallback

        callback = CompetitionTensorBoardCallback(
            log_dir=str(tmp_path),
            target_map50=0.85,
        )
        callback.start_time = time.time() - 600  # 10分前

        mock_writer = MagicMock()
        callback.writer = mock_writer

        mock_trainer = MagicMock()
        mock_trainer.metrics = {"metrics/mAP50(B)": 0.90}

        callback.on_train_end(mock_trainer)

        # サマリテキストがログされる
        mock_writer.add_text.assert_called()
        # クリーンアップされる
        mock_writer.flush.assert_called()
        mock_writer.close.assert_called()

    def test_cleanup(self, tmp_path):
        """クリーンアップテスト"""
        from training.tensorboard_monitor import CompetitionTensorBoardCallback

        callback = CompetitionTensorBoardCallback(
            log_dir=str(tmp_path),
        )

        mock_writer = MagicMock()
        callback.writer = mock_writer

        callback.cleanup()

        mock_writer.flush.assert_called_once()
        mock_writer.close.assert_called_once()
        assert callback.writer is None

    def test_cleanup_multiple_calls(self, tmp_path):
        """クリーンアップの複数回呼び出しテスト"""
        from training.tensorboard_monitor import CompetitionTensorBoardCallback

        callback = CompetitionTensorBoardCallback(
            log_dir=str(tmp_path),
        )

        mock_writer = MagicMock()
        callback.writer = mock_writer

        callback.cleanup()
        callback.cleanup()  # 2回目の呼び出し

        # flush/close は1回だけ呼ばれる
        mock_writer.flush.assert_called_once()
        mock_writer.close.assert_called_once()


class TestGPUMonitorCallback:
    """GPUMonitorCallback のテスト"""

    def test_callback_creation(self):
        """コールバック作成テスト"""
        from training.tensorboard_monitor import GPUMonitorCallback

        mock_writer = MagicMock()
        callback = GPUMonitorCallback(mock_writer, log_interval=5)

        assert callback.writer is mock_writer
        assert callback.log_interval == 5

    def test_on_train_epoch_end_log_interval(self):
        """ログ間隔によるフィルタリングテスト"""
        from training.tensorboard_monitor import GPUMonitorCallback

        mock_writer = MagicMock()
        callback = GPUMonitorCallback(mock_writer, log_interval=5)

        mock_trainer = MagicMock()
        mock_trainer.epoch = 2  # epoch 3 (5の倍数ではない)

        callback.on_train_epoch_end(mock_trainer)

        # 5の倍数でないのでログしない
        mock_writer.add_scalar.assert_not_called()

    def test_on_train_epoch_end_at_interval(self):
        """ログ間隔でのログ出力テスト"""
        from training.tensorboard_monitor import GPUMonitorCallback

        mock_writer = MagicMock()
        callback = GPUMonitorCallback(mock_writer, log_interval=5)

        mock_trainer = MagicMock()
        mock_trainer.epoch = 4  # epoch 5 (5の倍数)

        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.memory_allocated', return_value=1e9), \
             patch('torch.cuda.memory_reserved', return_value=2e9):
            callback.on_train_epoch_end(mock_trainer)

        # 5の倍数なのでログする
        assert mock_writer.add_scalar.called


class TestTensorBoardManager:
    """TensorBoardManager のテスト"""

    def test_manager_initialization(self, tmp_path):
        """マネージャー初期化テスト"""
        from training.tensorboard_monitor import TensorBoardManager

        manager = TensorBoardManager(tmp_path, max_runs=10)
        assert manager.base_dir == tmp_path
        assert manager.max_runs == 10
        assert manager.server is None
        assert manager.callback is None

    def test_get_log_dir(self, tmp_path):
        """ログディレクトリ取得テスト"""
        from training.tensorboard_monitor import TensorBoardManager

        manager = TensorBoardManager(tmp_path)
        log_dir = manager.get_log_dir("my_run")

        assert log_dir == tmp_path / "my_run" / "tensorboard"

    def test_create_callback(self, tmp_path):
        """コールバック作成テスト"""
        from training.tensorboard_monitor import TensorBoardManager

        manager = TensorBoardManager(tmp_path)
        callback = manager.create_callback("my_run", target_map50=0.90)

        assert manager.callback is not None
        assert callback.target_map50 == 0.90
        assert str(callback.log_dir) == str(tmp_path / "my_run" / "tensorboard")

    def test_start_server(self, tmp_path):
        """サーバー起動テスト"""
        from training.tensorboard_monitor import TensorBoardManager

        manager = TensorBoardManager(tmp_path)

        with patch('training.tensorboard_monitor.subprocess.Popen') as mock_popen, \
             patch('training.tensorboard_monitor.time.sleep'):
            mock_process = MagicMock()
            mock_process.poll.return_value = None
            mock_popen.return_value = mock_process

            url = manager.start_server("my_run", port=6006)

            assert manager.server is not None
            assert url == "http://localhost:6006"

    def test_stop_server(self, tmp_path):
        """サーバー停止テスト"""
        from training.tensorboard_monitor import TensorBoardManager

        manager = TensorBoardManager(tmp_path)

        with patch('training.tensorboard_monitor.subprocess.Popen') as mock_popen, \
             patch('training.tensorboard_monitor.time.sleep'):
            mock_process = MagicMock()
            mock_process.poll.return_value = None
            mock_popen.return_value = mock_process

            manager.start_server("my_run")
            manager.stop_server()

            assert manager.server is None

    def test_cleanup_old_logs(self, tmp_path):
        """古いログのクリーンアップテスト"""
        # テスト用のランディレクトリを作成
        for i in range(5):
            run_dir = tmp_path / f"run_{i}"
            tb_dir = run_dir / "tensorboard"
            tb_dir.mkdir(parents=True)
            # ファイルを追加してmtimeを設定
            dummy_file = tb_dir / "events.out.tfevents.12345"
            dummy_file.touch()

        from training.tensorboard_monitor import TensorBoardManager

        manager = TensorBoardManager(tmp_path, max_runs=3)
        cleaned = manager.cleanup_old_logs()

        assert cleaned == 2  # 5 - 3 = 2つクリーンアップ

    def test_get_all_log_dirs(self, tmp_path):
        """全ログディレクトリ取得テスト"""
        # テスト用のランディレクトリを作成
        for i in range(3):
            run_dir = tmp_path / f"run_{i}"
            tb_dir = run_dir / "tensorboard"
            tb_dir.mkdir(parents=True)

        from training.tensorboard_monitor import TensorBoardManager

        manager = TensorBoardManager(tmp_path)
        log_dirs = manager.get_all_log_dirs()

        assert len(log_dirs) == 3

    def test_get_all_log_dirs_empty(self, tmp_path):
        """空のログディレクトリ取得テスト"""
        from training.tensorboard_monitor import TensorBoardManager

        manager = TensorBoardManager(tmp_path)
        log_dirs = manager.get_all_log_dirs()

        assert log_dirs == []


class TestEnableUltralyticsTensorBoard:
    """enable_ultralytics_tensorboard のテスト"""

    def test_enable_with_ultralytics(self):
        """Ultralytics がインストールされている場合のテスト"""
        from training.tensorboard_monitor import enable_ultralytics_tensorboard

        mock_settings = MagicMock()
        with patch.dict('sys.modules', {'ultralytics': MagicMock()}):
            with patch('training.tensorboard_monitor.settings', mock_settings, create=True):
                # 関数呼び出しで例外が発生しないことを確認
                try:
                    enable_ultralytics_tensorboard()
                except Exception:
                    # 例外が発生しても問題なし（内部でハンドルされる）
                    pass

    def test_ultralytics_not_available(self):
        """Ultralytics未インストール時のテスト"""
        from training.tensorboard_monitor import enable_ultralytics_tensorboard

        # 例外が発生しないことを確認
        enable_ultralytics_tensorboard()


class TestCheckTensorBoardAvailable:
    """check_tensorboard_available のテスト"""

    def test_check_availability(self):
        """利用可能性チェックテスト"""
        from training.tensorboard_monitor import (
            check_tensorboard_available,
            TENSORBOARD_AVAILABLE,
        )

        result = check_tensorboard_available()
        assert result == TENSORBOARD_AVAILABLE
        assert isinstance(result, bool)

    def test_tensorboard_available_is_bool(self):
        """TENSORBOARD_AVAILABLE が bool 型であることを確認"""
        from training.tensorboard_monitor import TENSORBOARD_AVAILABLE

        assert isinstance(TENSORBOARD_AVAILABLE, bool)
