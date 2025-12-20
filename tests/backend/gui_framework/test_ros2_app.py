"""
ROS2 App モジュールのテスト

scripts/gui_framework/ros2_app.py のテストを行います。
"""

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
from abc import ABC

import pytest

# scriptsディレクトリをパスに追加
scripts_dir = Path(__file__).parent.parent.parent.parent / "scripts"
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))


class TestROS2AppAbstract:
    """ROS2App が抽象クラスであることのテスト"""

    def test_cannot_instantiate_directly(self):
        """ROS2App を直接インスタンス化できないことを確認"""
        with patch('rclpy.init'):
            with patch('gui_framework.ros2_app.ROS2ImageSubscriber'):
                with patch('gui_framework.ros2_app.SingleThreadedExecutor'):
                    with patch('tkinter.ttk.Style'):
                        with patch('gui_framework.base_app.AppTheme'):
                            from gui_framework.ros2_app import ROS2App

                            mock_root = MagicMock()

                            # 抽象クラスなのでインスタンス化できない
                            with pytest.raises(TypeError):
                                ROS2App(mock_root, "Test App")

    def test_is_subclass_of_base_app(self):
        """ROS2App が BaseApp のサブクラスであることを確認"""
        from gui_framework.ros2_app import ROS2App
        from gui_framework.base_app import BaseApp

        assert issubclass(ROS2App, BaseApp)

    def test_abstract_methods_defined(self):
        """抽象メソッドが定義されていることを確認"""
        from gui_framework.ros2_app import ROS2App

        assert hasattr(ROS2App, '_build_gui')
        assert hasattr(ROS2App, '_on_close')


class TestROS2AppInit:
    """ROS2App 初期化のテスト"""

    def test_init_initializes_ros2(self):
        """初期化時にROS2が初期化されることを確認"""
        with patch('rclpy.init') as mock_rclpy_init:
            with patch('gui_framework.ros2_app.ROS2ImageSubscriber') as mock_subscriber:
                with patch('gui_framework.ros2_app.SingleThreadedExecutor') as mock_executor:
                    with patch('tkinter.ttk.Style'):
                        with patch('gui_framework.base_app.AppTheme'):
                            from gui_framework.ros2_app import ROS2App

                            class TestApp(ROS2App):
                                def _build_gui(self):
                                    pass

                                def _on_close(self):
                                    pass

                            mock_root = MagicMock()
                            mock_node = MagicMock()
                            mock_subscriber.return_value = mock_node

                            mock_exec = MagicMock()
                            mock_executor.return_value = mock_exec

                            app = TestApp(mock_root, "Test App", node_name="test_node")

                            mock_rclpy_init.assert_called_once()
                            mock_subscriber.assert_called_once_with("test_node")
                            mock_exec.add_node.assert_called_once_with(mock_node)

    def test_init_starts_ros_thread(self):
        """初期化時にROS2スレッドが開始されることを確認"""
        with patch('rclpy.init'):
            with patch('gui_framework.ros2_app.ROS2ImageSubscriber') as mock_subscriber:
                with patch('gui_framework.ros2_app.SingleThreadedExecutor'):
                    with patch('tkinter.ttk.Style'):
                        with patch('gui_framework.base_app.AppTheme'):
                            from gui_framework.ros2_app import ROS2App

                            class TestApp(ROS2App):
                                def _build_gui(self):
                                    pass

                                def _on_close(self):
                                    pass

                            mock_root = MagicMock()
                            mock_subscriber.return_value = MagicMock()

                            app = TestApp(mock_root, "Test App")

                            assert app._ros_running is True
                            assert app._ros_thread is not None

    def test_init_ros2_failure(self):
        """ROS2初期化失敗時のテスト"""
        with patch('rclpy.init', side_effect=Exception("ROS2 init failed")):
            with patch('tkinter.ttk.Style'):
                with patch('gui_framework.base_app.AppTheme'):
                    from gui_framework.ros2_app import ROS2App

                    class TestApp(ROS2App):
                        def _build_gui(self):
                            pass

                        def _on_close(self):
                            pass

                    mock_root = MagicMock()

                    app = TestApp(mock_root, "Test App")

                    assert app.ros_node is None
                    assert app.executor is None


class TestROS2AppShutdown:
    """ROS2 シャットダウンのテスト"""

    def test_shutdown_stops_ros_thread(self):
        """シャットダウン時にROS2スレッドが停止することを確認"""
        with patch('rclpy.init'):
            with patch('rclpy.shutdown'):
                with patch('gui_framework.ros2_app.ROS2ImageSubscriber') as mock_subscriber:
                    with patch('gui_framework.ros2_app.SingleThreadedExecutor') as mock_executor_cls:
                        with patch('tkinter.ttk.Style'):
                            with patch('gui_framework.base_app.AppTheme'):
                                from gui_framework.ros2_app import ROS2App

                                class TestApp(ROS2App):
                                    def _build_gui(self):
                                        pass

                                    def _on_close(self):
                                        pass

                                mock_root = MagicMock()
                                mock_node = MagicMock()
                                mock_subscriber.return_value = mock_node
                                mock_executor = MagicMock()
                                mock_executor_cls.return_value = mock_executor

                                app = TestApp(mock_root, "Test App")

                                app._shutdown_ros2()

                                assert app._ros_running is False
                                assert app.ros_node is None
                                assert app.executor is None

    def test_shutdown_calls_rclpy_shutdown(self):
        """シャットダウン時にrclpy.shutdown()が呼ばれることを確認"""
        with patch('rclpy.init'):
            with patch('rclpy.shutdown') as mock_rclpy_shutdown:
                with patch('gui_framework.ros2_app.ROS2ImageSubscriber') as mock_subscriber:
                    with patch('gui_framework.ros2_app.SingleThreadedExecutor') as mock_executor_cls:
                        with patch('tkinter.ttk.Style'):
                            with patch('gui_framework.base_app.AppTheme'):
                                from gui_framework.ros2_app import ROS2App

                                class TestApp(ROS2App):
                                    def _build_gui(self):
                                        pass

                                    def _on_close(self):
                                        pass

                                mock_root = MagicMock()
                                mock_subscriber.return_value = MagicMock()
                                mock_executor_cls.return_value = MagicMock()

                                app = TestApp(mock_root, "Test App")
                                app._shutdown_ros2()

                                mock_rclpy_shutdown.assert_called()


class TestROS2AppGetFrame:
    """get_frame メソッドのテスト"""

    def test_get_frame_returns_frame(self):
        """get_frame がフレームを返すことを確認"""
        with patch('rclpy.init'):
            with patch('gui_framework.ros2_app.ROS2ImageSubscriber') as mock_subscriber:
                with patch('gui_framework.ros2_app.SingleThreadedExecutor'):
                    with patch('tkinter.ttk.Style'):
                        with patch('gui_framework.base_app.AppTheme'):
                            from gui_framework.ros2_app import ROS2App

                            class TestApp(ROS2App):
                                def _build_gui(self):
                                    pass

                                def _on_close(self):
                                    pass

                            mock_root = MagicMock()
                            mock_node = MagicMock()
                            mock_frame = MagicMock()
                            mock_node.get_frame.return_value = mock_frame
                            mock_subscriber.return_value = mock_node

                            app = TestApp(mock_root, "Test App")
                            result = app.get_frame()

                            assert result == mock_frame
                            mock_node.get_frame.assert_called_once()

    def test_get_frame_returns_none_when_no_node(self):
        """ROS2ノードがない場合にNoneを返すことを確認"""
        with patch('rclpy.init', side_effect=Exception("Init failed")):
            with patch('tkinter.ttk.Style'):
                with patch('gui_framework.base_app.AppTheme'):
                    from gui_framework.ros2_app import ROS2App

                    class TestApp(ROS2App):
                        def _build_gui(self):
                            pass

                        def _on_close(self):
                            pass

                    mock_root = MagicMock()
                    app = TestApp(mock_root, "Test App")
                    result = app.get_frame()

                    assert result is None


class TestROS2AppGetImageTopics:
    """get_image_topics メソッドのテスト"""

    def test_get_image_topics_returns_list(self):
        """get_image_topics がトピックリストを返すことを確認"""
        with patch('rclpy.init'):
            with patch('gui_framework.ros2_app.ROS2ImageSubscriber') as mock_subscriber:
                with patch('gui_framework.ros2_app.SingleThreadedExecutor'):
                    with patch('tkinter.ttk.Style'):
                        with patch('gui_framework.base_app.AppTheme'):
                            from gui_framework.ros2_app import ROS2App

                            class TestApp(ROS2App):
                                def _build_gui(self):
                                    pass

                                def _on_close(self):
                                    pass

                            mock_root = MagicMock()
                            mock_node = MagicMock()
                            mock_node.get_image_topics.return_value = ["/camera/image", "/depth/image"]
                            mock_subscriber.return_value = mock_node

                            app = TestApp(mock_root, "Test App")
                            result = app.get_image_topics()

                            assert result == ["/camera/image", "/depth/image"]

    def test_get_image_topics_returns_empty_when_no_node(self):
        """ROS2ノードがない場合に空リストを返すことを確認"""
        with patch('rclpy.init', side_effect=Exception("Init failed")):
            with patch('tkinter.ttk.Style'):
                with patch('gui_framework.base_app.AppTheme'):
                    from gui_framework.ros2_app import ROS2App

                    class TestApp(ROS2App):
                        def _build_gui(self):
                            pass

                        def _on_close(self):
                            pass

                    mock_root = MagicMock()
                    app = TestApp(mock_root, "Test App")
                    result = app.get_image_topics()

                    assert result == []


class TestROS2AppSubscribeToTopic:
    """subscribe_to_topic メソッドのテスト"""

    def test_subscribe_to_topic_calls_node(self):
        """subscribe_to_topic がノードのメソッドを呼ぶことを確認"""
        with patch('rclpy.init'):
            with patch('gui_framework.ros2_app.ROS2ImageSubscriber') as mock_subscriber:
                with patch('gui_framework.ros2_app.SingleThreadedExecutor'):
                    with patch('tkinter.ttk.Style'):
                        with patch('gui_framework.base_app.AppTheme'):
                            from gui_framework.ros2_app import ROS2App

                            class TestApp(ROS2App):
                                def _build_gui(self):
                                    pass

                                def _on_close(self):
                                    pass

                            mock_root = MagicMock()
                            mock_node = MagicMock()
                            mock_subscriber.return_value = mock_node

                            app = TestApp(mock_root, "Test App")
                            app.subscribe_to_topic("/camera/image")

                            mock_node.subscribe_to_topic.assert_called_once_with("/camera/image")

    def test_subscribe_to_topic_no_node(self):
        """ROS2ノードがない場合に何もしないことを確認"""
        with patch('rclpy.init', side_effect=Exception("Init failed")):
            with patch('tkinter.ttk.Style'):
                with patch('gui_framework.base_app.AppTheme'):
                    from gui_framework.ros2_app import ROS2App

                    class TestApp(ROS2App):
                        def _build_gui(self):
                            pass

                        def _on_close(self):
                            pass

                    mock_root = MagicMock()
                    app = TestApp(mock_root, "Test App")
                    # 例外が発生しないことを確認
                    app.subscribe_to_topic("/camera/image")
