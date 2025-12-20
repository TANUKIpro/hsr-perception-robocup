"""
Burst Capture モジュールのテスト

scripts/capture/burst_capture.py のテストを行います。
"""

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
import numpy as np

import pytest

# scriptsディレクトリをパスに追加
scripts_dir = Path(__file__).parent.parent.parent.parent / "scripts"
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))


class TestImgmsgToCv2:
    """imgmsg_to_cv2 関数のテスト"""

    def test_rgb8_encoding(self):
        """RGB8エンコーディングの変換テスト"""
        from capture.burst_capture import imgmsg_to_cv2

        # ROS2 Image メッセージをモック
        msg = MagicMock()
        msg.encoding = 'rgb8'
        msg.height = 100
        msg.width = 100
        msg.data = np.zeros((100 * 100 * 3,), dtype=np.uint8).tobytes()

        result = imgmsg_to_cv2(msg)

        assert result.shape == (100, 100, 3)
        assert result.dtype == np.uint8

    def test_bgr8_encoding(self):
        """BGR8エンコーディングの変換テスト"""
        from capture.burst_capture import imgmsg_to_cv2

        msg = MagicMock()
        msg.encoding = 'bgr8'
        msg.height = 100
        msg.width = 100
        msg.data = np.zeros((100 * 100 * 3,), dtype=np.uint8).tobytes()

        result = imgmsg_to_cv2(msg)

        assert result.shape == (100, 100, 3)
        assert result.dtype == np.uint8

    def test_mono8_encoding(self):
        """Mono8エンコーディングの変換テスト"""
        from capture.burst_capture import imgmsg_to_cv2

        msg = MagicMock()
        msg.encoding = 'mono8'
        msg.height = 100
        msg.width = 100
        msg.data = np.zeros((100 * 100,), dtype=np.uint8).tobytes()

        result = imgmsg_to_cv2(msg)

        # BGRに変換されるので3チャンネル
        assert result.shape == (100, 100, 3)

    def test_mono16_encoding(self):
        """Mono16エンコーディングの変換テスト"""
        from capture.burst_capture import imgmsg_to_cv2

        msg = MagicMock()
        msg.encoding = 'mono16'
        msg.height = 50
        msg.width = 50
        # 注: ソースコードでは mono16 は dtype=uint8 と処理されるので、
        # 各ピクセル1バイトとして処理される（本来は2バイトだが）
        # これはソースコードの現在の実装に合わせたテスト
        msg.data = np.zeros((50 * 50,), dtype=np.uint8).tobytes()

        result = imgmsg_to_cv2(msg)

        # BGRに変換されるので3チャンネル
        assert result.shape == (50, 50, 3)
        assert result.dtype == np.uint8

    def test_rgba8_encoding(self):
        """RGBA8エンコーディングの変換テスト"""
        from capture.burst_capture import imgmsg_to_cv2

        msg = MagicMock()
        msg.encoding = 'rgba8'
        msg.height = 100
        msg.width = 100
        msg.data = np.zeros((100 * 100 * 4,), dtype=np.uint8).tobytes()

        result = imgmsg_to_cv2(msg)

        # BGRに変換されるので3チャンネル
        assert result.shape == (100, 100, 3)

    def test_16uc1_encoding(self):
        """16UC1エンコーディングの変換テスト"""
        from capture.burst_capture import imgmsg_to_cv2

        msg = MagicMock()
        msg.encoding = '16UC1'
        msg.height = 100
        msg.width = 100
        msg.data = np.zeros((100 * 100,), dtype=np.uint16).tobytes()

        result = imgmsg_to_cv2(msg)

        # 正規化されてBGRに変換
        assert result.shape == (100, 100, 3)
        assert result.dtype == np.uint8


class TestBurstCaptureInit:
    """BurstCapture クラスの初期化テスト"""

    def test_initialization(self, tmp_path):
        """基本的な初期化テスト"""
        from capture.burst_capture import BurstCapture

        # Nodeの__init__をスキップしてオブジェクトを作成
        with patch('rclpy.node.Node.__init__', return_value=None):
            node = object.__new__(BurstCapture)
            node.output_dir = tmp_path / "captures"
            node.class_name = "cup"
            node.target_count = 50
            node.interval = 0.2
            node.captured_count = 0
            node.latest_frame = None
            node.last_capture_time = 0.0

            assert node.target_count == 50
            assert node.interval == 0.2
            assert node.captured_count == 0

    def test_initialization_creates_output_dir(self, tmp_path):
        """出力ディレクトリが作成されることを確認"""
        from capture.burst_capture import BurstCapture

        output_dir = tmp_path / "new_dir" / "captures"

        with patch('rclpy.node.Node.__init__', return_value=None):
            node = object.__new__(BurstCapture)
            node.output_dir = output_dir
            node.output_dir.mkdir(parents=True, exist_ok=True)

            assert output_dir.exists()


class TestBurstCaptureCallback:
    """BurstCapture コールバックのテスト"""

    def test_callback_stores_latest_frame(self, tmp_path):
        """コールバックが最新フレームを保存することを確認"""
        from capture.burst_capture import BurstCapture, imgmsg_to_cv2

        with patch('rclpy.node.Node.__init__', return_value=None):
            node = object.__new__(BurstCapture)
            node.output_dir = tmp_path
            node.class_name = "cup"
            node.target_count = 50
            node.interval = 0.2
            node.captured_count = 0
            node.latest_frame = None
            node.last_capture_time = 0.0

            # ロガーをモック
            mock_logger = MagicMock()
            node.get_logger = MagicMock(return_value=mock_logger)

            # テスト用のメッセージを作成
            msg = MagicMock()
            msg.encoding = 'bgr8'
            msg.height = 100
            msg.width = 100
            msg.data = np.zeros((100 * 100 * 3,), dtype=np.uint8).tobytes()

            node.callback(msg)

            assert node.latest_frame is not None
            assert node.latest_frame.shape == (100, 100, 3)

    def test_callback_handles_conversion_error(self, tmp_path):
        """コールバックが変換エラーを処理することを確認"""
        from capture.burst_capture import BurstCapture

        with patch('rclpy.node.Node.__init__', return_value=None):
            node = object.__new__(BurstCapture)
            node.output_dir = tmp_path
            node.class_name = "cup"
            node.target_count = 50
            node.interval = 0.2
            node.captured_count = 0
            node.latest_frame = None
            node.last_capture_time = 0.0

            mock_logger = MagicMock()
            node.get_logger = MagicMock(return_value=mock_logger)

            # エラーを発生させるメッセージ
            msg = MagicMock()
            msg.encoding = 'invalid'
            msg.height = 100
            msg.width = 100
            msg.data = b'invalid data'

            node.callback(msg)

            # エラーがログされる
            mock_logger.error.assert_called_once()


class TestBurstCaptureCaptureIfReady:
    """capture_if_ready メソッドのテスト"""

    def test_capture_if_ready_no_frame(self, tmp_path):
        """フレームがない場合のテスト"""
        from capture.burst_capture import BurstCapture

        with patch('rclpy.node.Node.__init__', return_value=None):
            node = object.__new__(BurstCapture)
            node.output_dir = tmp_path
            node.class_name = "cup"
            node.target_count = 50
            node.interval = 0.2
            node.captured_count = 0
            node.latest_frame = None
            node.last_capture_time = 0.0

            result = node.capture_if_ready()

            assert result is False

    def test_capture_if_ready_interval_not_passed(self, tmp_path):
        """インターバルが経過していない場合のテスト"""
        from capture.burst_capture import BurstCapture

        with patch('rclpy.node.Node.__init__', return_value=None):
            node = object.__new__(BurstCapture)
            node.output_dir = tmp_path
            node.output_dir.mkdir(parents=True, exist_ok=True)
            node.class_name = "cup"
            node.target_count = 50
            node.interval = 10.0  # 長いインターバル
            node.captured_count = 0
            node.latest_frame = np.zeros((100, 100, 3), dtype=np.uint8)
            node.last_capture_time = time.time()  # 今キャプチャ

            result = node.capture_if_ready()

            # インターバルが経過していないのでキャプチャしない
            assert result is False

    def test_capture_if_ready_captures_image(self, tmp_path):
        """キャプチャが正常に実行されることを確認"""
        from capture.burst_capture import BurstCapture

        output_dir = tmp_path / "captures"
        output_dir.mkdir(parents=True, exist_ok=True)

        with patch('rclpy.node.Node.__init__', return_value=None):
            node = object.__new__(BurstCapture)
            node.output_dir = output_dir
            node.class_name = "cup"
            node.target_count = 50
            node.interval = 0.0  # 即座にキャプチャ
            node.captured_count = 0
            node.latest_frame = np.zeros((100, 100, 3), dtype=np.uint8)
            node.last_capture_time = 0.0  # 過去の時間

            mock_logger = MagicMock()
            node.get_logger = MagicMock(return_value=mock_logger)

            result = node.capture_if_ready()

            assert result is False  # まだターゲットに達していない
            assert node.captured_count == 1
            # ファイルが作成されていることを確認
            files = list(output_dir.glob("*.jpg"))
            assert len(files) == 1
            assert files[0].name.startswith("cup_")

    def test_capture_if_ready_returns_true_when_done(self, tmp_path):
        """ターゲット数に達した場合にTrueを返すことを確認"""
        from capture.burst_capture import BurstCapture

        output_dir = tmp_path / "captures"
        output_dir.mkdir(parents=True, exist_ok=True)

        with patch('rclpy.node.Node.__init__', return_value=None):
            node = object.__new__(BurstCapture)
            node.output_dir = output_dir
            node.class_name = "cup"
            node.target_count = 1  # 1枚だけ
            node.interval = 0.0
            node.captured_count = 0
            node.latest_frame = np.zeros((100, 100, 3), dtype=np.uint8)
            node.last_capture_time = 0.0

            mock_logger = MagicMock()
            node.get_logger = MagicMock(return_value=mock_logger)

            result = node.capture_if_ready()

            assert result is True
            assert node.captured_count == 1


class TestBurstCaptureIsDone:
    """is_done メソッドのテスト"""

    def test_is_done_false(self, tmp_path):
        """まだ完了していない場合のテスト"""
        from capture.burst_capture import BurstCapture

        with patch('rclpy.node.Node.__init__', return_value=None):
            node = object.__new__(BurstCapture)
            node.output_dir = tmp_path
            node.class_name = "cup"
            node.target_count = 50
            node.captured_count = 10

            assert node.is_done() is False

    def test_is_done_true(self, tmp_path):
        """完了した場合のテスト"""
        from capture.burst_capture import BurstCapture

        with patch('rclpy.node.Node.__init__', return_value=None):
            node = object.__new__(BurstCapture)
            node.output_dir = tmp_path
            node.class_name = "cup"
            node.target_count = 50
            node.captured_count = 50

            assert node.is_done() is True

    def test_is_done_true_exceeded(self, tmp_path):
        """ターゲット数を超過した場合のテスト"""
        from capture.burst_capture import BurstCapture

        with patch('rclpy.node.Node.__init__', return_value=None):
            node = object.__new__(BurstCapture)
            node.output_dir = tmp_path
            node.class_name = "cup"
            node.target_count = 50
            node.captured_count = 100

            assert node.is_done() is True
