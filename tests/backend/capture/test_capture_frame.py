"""
Capture Frame モジュールのテスト

scripts/capture/capture_frame.py のテストを行います。
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np

import pytest

# scriptsディレクトリをパスに追加
scripts_dir = Path(__file__).parent.parent.parent.parent / "scripts"
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))


class TestFrameCaptureInit:
    """FrameCapture クラスの初期化テスト"""

    def test_initialization(self, tmp_path):
        """基本的な初期化テスト"""
        from capture.capture_frame import FrameCapture

        output_path = tmp_path / "test.jpg"

        with patch('rclpy.node.Node.__init__', return_value=None):
            node = object.__new__(FrameCapture)
            node.output_path = str(output_path)
            node.received = False
            node.bridge = MagicMock()

            assert node.received is False
            assert node.output_path == str(output_path)

    def test_initialization_attributes(self, tmp_path):
        """初期化時の属性値テスト"""
        from capture.capture_frame import FrameCapture

        output_path = tmp_path / "frame.jpg"

        with patch('rclpy.node.Node.__init__', return_value=None):
            node = object.__new__(FrameCapture)
            node.output_path = str(output_path)
            node.received = False
            node.bridge = MagicMock()
            node.subscription = MagicMock()

            assert hasattr(node, 'output_path')
            assert hasattr(node, 'received')
            assert hasattr(node, 'bridge')


class TestFrameCaptureCallback:
    """FrameCapture コールバックのテスト"""

    def test_callback_rgb8_encoding(self, tmp_path):
        """RGB8エンコーディングのコールバックテスト"""
        from capture.capture_frame import FrameCapture

        output_path = tmp_path / "test_rgb8.jpg"

        with patch('rclpy.node.Node.__init__', return_value=None):
            node = object.__new__(FrameCapture)
            node.output_path = str(output_path)
            node.received = False

            mock_bridge = MagicMock()
            mock_bridge.imgmsg_to_cv2.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
            node.bridge = mock_bridge

            mock_logger = MagicMock()
            node.get_logger = MagicMock(return_value=mock_logger)

            # テストメッセージ
            msg = MagicMock()
            msg.encoding = 'rgb8'

            node.callback(msg)

            assert node.received is True
            mock_bridge.imgmsg_to_cv2.assert_called_once_with(msg, "bgr8")
            assert output_path.exists()

    def test_callback_bgr8_encoding(self, tmp_path):
        """BGR8エンコーディングのコールバックテスト"""
        from capture.capture_frame import FrameCapture

        output_path = tmp_path / "test_bgr8.jpg"

        with patch('rclpy.node.Node.__init__', return_value=None):
            node = object.__new__(FrameCapture)
            node.output_path = str(output_path)
            node.received = False

            mock_bridge = MagicMock()
            mock_bridge.imgmsg_to_cv2.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
            node.bridge = mock_bridge

            mock_logger = MagicMock()
            node.get_logger = MagicMock(return_value=mock_logger)

            msg = MagicMock()
            msg.encoding = 'bgr8'

            node.callback(msg)

            assert node.received is True
            assert output_path.exists()

    def test_callback_mono8_encoding(self, tmp_path):
        """Mono8エンコーディングのコールバックテスト"""
        from capture.capture_frame import FrameCapture

        output_path = tmp_path / "test_mono8.jpg"

        with patch('rclpy.node.Node.__init__', return_value=None):
            node = object.__new__(FrameCapture)
            node.output_path = str(output_path)
            node.received = False

            mock_bridge = MagicMock()
            mock_bridge.imgmsg_to_cv2.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
            node.bridge = mock_bridge

            mock_logger = MagicMock()
            node.get_logger = MagicMock(return_value=mock_logger)

            msg = MagicMock()
            msg.encoding = 'mono8'

            node.callback(msg)

            assert node.received is True
            mock_bridge.imgmsg_to_cv2.assert_called_once_with(msg, "bgr8")

    def test_callback_depth_16uc1_encoding(self, tmp_path):
        """16UC1深度エンコーディングのコールバックテスト"""
        from capture.capture_frame import FrameCapture

        output_path = tmp_path / "test_depth.jpg"

        with patch('rclpy.node.Node.__init__', return_value=None):
            node = object.__new__(FrameCapture)
            node.output_path = str(output_path)
            node.received = False

            mock_bridge = MagicMock()
            # 深度画像をシミュレート
            mock_bridge.imgmsg_to_cv2.return_value = np.zeros((100, 100), dtype=np.uint16)
            node.bridge = mock_bridge

            mock_logger = MagicMock()
            node.get_logger = MagicMock(return_value=mock_logger)

            msg = MagicMock()
            msg.encoding = '16UC1'

            node.callback(msg)

            assert node.received is True
            # passthrough で呼び出される
            mock_bridge.imgmsg_to_cv2.assert_called_once_with(msg, "passthrough")
            assert output_path.exists()

    def test_callback_already_received(self, tmp_path):
        """既に受信済みの場合は何もしないテスト"""
        from capture.capture_frame import FrameCapture

        output_path = tmp_path / "test_already.jpg"

        with patch('rclpy.node.Node.__init__', return_value=None):
            node = object.__new__(FrameCapture)
            node.output_path = str(output_path)
            node.received = True  # 既に受信済み

            mock_bridge = MagicMock()
            node.bridge = mock_bridge

            msg = MagicMock()
            msg.encoding = 'bgr8'

            node.callback(msg)

            # bridge.imgmsg_to_cv2 が呼ばれないことを確認
            mock_bridge.imgmsg_to_cv2.assert_not_called()

    def test_callback_conversion_error(self, tmp_path):
        """変換エラー時のコールバックテスト"""
        from capture.capture_frame import FrameCapture

        output_path = tmp_path / "test_error.jpg"

        with patch('rclpy.node.Node.__init__', return_value=None):
            node = object.__new__(FrameCapture)
            node.output_path = str(output_path)
            node.received = False

            mock_bridge = MagicMock()
            mock_bridge.imgmsg_to_cv2.side_effect = Exception("Conversion error")
            node.bridge = mock_bridge

            mock_logger = MagicMock()
            node.get_logger = MagicMock(return_value=mock_logger)

            msg = MagicMock()
            msg.encoding = 'bgr8'

            node.callback(msg)

            # received は False のまま
            assert node.received is False
            # エラーログが出力される
            mock_logger.error.assert_called_once()

    def test_callback_unknown_encoding(self, tmp_path):
        """不明なエンコーディングのコールバックテスト"""
        from capture.capture_frame import FrameCapture

        output_path = tmp_path / "test_unknown.jpg"

        with patch('rclpy.node.Node.__init__', return_value=None):
            node = object.__new__(FrameCapture)
            node.output_path = str(output_path)
            node.received = False

            mock_bridge = MagicMock()
            mock_bridge.imgmsg_to_cv2.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
            node.bridge = mock_bridge

            mock_logger = MagicMock()
            node.get_logger = MagicMock(return_value=mock_logger)

            msg = MagicMock()
            msg.encoding = 'unknown_encoding'

            node.callback(msg)

            assert node.received is True
            # デフォルトで bgr8 変換を試みる
            mock_bridge.imgmsg_to_cv2.assert_called_once_with(msg, "bgr8")


class TestFrameCaptureEncoding:
    """エンコーディング処理のテスト"""

    def test_encoding_list_rgb(self, tmp_path):
        """RGBエンコーディングリストのテスト"""
        from capture.capture_frame import FrameCapture

        output_path = tmp_path / "test.jpg"

        with patch('rclpy.node.Node.__init__', return_value=None):
            node = object.__new__(FrameCapture)
            node.output_path = str(output_path)
            node.received = False

            mock_bridge = MagicMock()
            mock_bridge.imgmsg_to_cv2.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
            node.bridge = mock_bridge
            node.get_logger = MagicMock(return_value=MagicMock())

            for encoding in ['rgb8', 'bgr8', 'mono8']:
                node.received = False
                msg = MagicMock()
                msg.encoding = encoding
                node.callback(msg)
                assert node.received is True

    def test_encoding_depth_list(self, tmp_path):
        """深度エンコーディングリストのテスト"""
        from capture.capture_frame import FrameCapture

        output_path = tmp_path / "test_depth.jpg"

        with patch('rclpy.node.Node.__init__', return_value=None):
            node = object.__new__(FrameCapture)
            node.output_path = str(output_path)

            mock_bridge = MagicMock()
            mock_bridge.imgmsg_to_cv2.return_value = np.zeros((100, 100), dtype=np.float32)
            node.bridge = mock_bridge
            node.get_logger = MagicMock(return_value=MagicMock())

            for encoding in ['16UC1', '32FC1']:
                node.received = False
                msg = MagicMock()
                msg.encoding = encoding
                node.callback(msg)
                assert node.received is True
