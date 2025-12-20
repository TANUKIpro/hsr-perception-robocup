"""
Base App モジュールのテスト

scripts/gui_framework/base_app.py のテストを行います。
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
from abc import ABC

import pytest

# scriptsディレクトリをパスに追加
scripts_dir = Path(__file__).parent.parent.parent.parent / "scripts"
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))


class TestBaseAppAbstract:
    """BaseApp が抽象クラスであることのテスト"""

    def test_cannot_instantiate_directly(self):
        """BaseApp を直接インスタンス化できないことを確認"""
        with patch('tkinter.Tk') as mock_tk:
            with patch('tkinter.ttk.Style') as mock_style:
                from gui_framework.base_app import BaseApp

                mock_root = MagicMock()
                mock_tk.return_value = mock_root

                # 抽象クラスなのでインスタンス化できない
                with pytest.raises(TypeError):
                    BaseApp(mock_root, "Test App")

    def test_is_abstract_class(self):
        """BaseApp が ABC を継承していることを確認"""
        from gui_framework.base_app import BaseApp

        assert issubclass(BaseApp, ABC)

    def test_abstract_methods_defined(self):
        """抽象メソッドが定義されていることを確認"""
        from gui_framework.base_app import BaseApp

        assert hasattr(BaseApp, '_build_gui')
        assert hasattr(BaseApp, '_on_close')


class TestBaseAppSubclass:
    """BaseApp サブクラスのテスト"""

    def test_subclass_can_instantiate(self):
        """サブクラスが正常にインスタンス化できることを確認"""
        with patch('tkinter.ttk.Style') as mock_style_cls:
            with patch('gui_framework.base_app.AppTheme') as mock_theme:
                from gui_framework.base_app import BaseApp

                # テスト用サブクラス
                class TestApp(BaseApp):
                    def _build_gui(self):
                        pass

                    def _on_close(self):
                        pass

                mock_root = MagicMock()
                mock_style_cls.return_value = MagicMock()

                app = TestApp(mock_root, "Test App")

                assert app.root == mock_root
                mock_theme.apply.assert_called_once()

    def test_subclass_sets_title(self):
        """サブクラスがウィンドウタイトルを設定することを確認"""
        with patch('tkinter.ttk.Style') as mock_style_cls:
            with patch('gui_framework.base_app.AppTheme') as mock_theme:
                from gui_framework.base_app import BaseApp

                class TestApp(BaseApp):
                    def _build_gui(self):
                        pass

                    def _on_close(self):
                        pass

                mock_root = MagicMock()
                mock_style_cls.return_value = MagicMock()

                app = TestApp(mock_root, "My Window Title")

                mock_root.title.assert_called_with("My Window Title")

    def test_subclass_sets_geometry(self):
        """サブクラスがウィンドウジオメトリを設定することを確認"""
        with patch('tkinter.ttk.Style') as mock_style_cls:
            with patch('gui_framework.base_app.AppTheme') as mock_theme:
                from gui_framework.base_app import BaseApp

                class TestApp(BaseApp):
                    def _build_gui(self):
                        pass

                    def _on_close(self):
                        pass

                mock_root = MagicMock()
                mock_style_cls.return_value = MagicMock()

                app = TestApp(mock_root, "Test App", "1024x768")

                mock_root.geometry.assert_called_with("1024x768")

    def test_subclass_sets_min_size(self):
        """サブクラスが最小サイズを設定することを確認"""
        with patch('tkinter.ttk.Style') as mock_style_cls:
            with patch('gui_framework.base_app.AppTheme') as mock_theme:
                from gui_framework.base_app import BaseApp

                class TestApp(BaseApp):
                    def _build_gui(self):
                        pass

                    def _on_close(self):
                        pass

                mock_root = MagicMock()
                mock_style_cls.return_value = MagicMock()

                app = TestApp(mock_root, "Test App", min_size=(400, 300))

                mock_root.minsize.assert_called_with(400, 300)


class TestBaseAppKeyboardShortcuts:
    """キーボードショートカットのテスト"""

    def test_q_key_binds_to_close(self):
        """qキーがクローズにバインドされていることを確認"""
        with patch('tkinter.ttk.Style') as mock_style_cls:
            with patch('gui_framework.base_app.AppTheme') as mock_theme:
                from gui_framework.base_app import BaseApp

                class TestApp(BaseApp):
                    def _build_gui(self):
                        pass

                    def _on_close(self):
                        pass

                mock_root = MagicMock()
                mock_style_cls.return_value = MagicMock()

                app = TestApp(mock_root, "Test App")

                # bind が呼ばれたことを確認
                bind_calls = mock_root.bind.call_args_list
                bound_keys = [call[0][0] for call in bind_calls]
                assert "<q>" in bound_keys
                assert "<Q>" in bound_keys
                assert "<Escape>" in bound_keys

    def test_escape_handler_default(self):
        """Escapeハンドラのデフォルト動作テスト"""
        with patch('tkinter.ttk.Style') as mock_style_cls:
            with patch('gui_framework.base_app.AppTheme') as mock_theme:
                from gui_framework.base_app import BaseApp

                class TestApp(BaseApp):
                    def _build_gui(self):
                        pass

                    def _on_close(self):
                        pass

                mock_root = MagicMock()
                mock_style_cls.return_value = MagicMock()

                app = TestApp(mock_root, "Test App")

                # デフォルトのEscapeハンドラは何もしない
                event = MagicMock()
                app._on_escape(event)  # 例外が発生しないことを確認


class TestBaseAppDialogs:
    """ダイアログメソッドのテスト"""

    def test_show_error_dialog(self):
        """エラーダイアログ表示テスト"""
        with patch('tkinter.ttk.Style') as mock_style_cls:
            with patch('gui_framework.base_app.AppTheme') as mock_theme:
                with patch('tkinter.messagebox.showerror') as mock_showerror:
                    from gui_framework.base_app import BaseApp

                    class TestApp(BaseApp):
                        def _build_gui(self):
                            pass

                        def _on_close(self):
                            pass

                    mock_root = MagicMock()
                    mock_style_cls.return_value = MagicMock()

                    app = TestApp(mock_root, "Test App")
                    app.show_error("Error Title", "Error Message")

                    mock_showerror.assert_called_once_with("Error Title", "Error Message")

    def test_show_warning_dialog(self):
        """警告ダイアログ表示テスト"""
        with patch('tkinter.ttk.Style') as mock_style_cls:
            with patch('gui_framework.base_app.AppTheme') as mock_theme:
                with patch('tkinter.messagebox.showwarning') as mock_showwarning:
                    from gui_framework.base_app import BaseApp

                    class TestApp(BaseApp):
                        def _build_gui(self):
                            pass

                        def _on_close(self):
                            pass

                    mock_root = MagicMock()
                    mock_style_cls.return_value = MagicMock()

                    app = TestApp(mock_root, "Test App")
                    app.show_warning("Warning Title", "Warning Message")

                    mock_showwarning.assert_called_once_with("Warning Title", "Warning Message")

    def test_show_info_dialog(self):
        """情報ダイアログ表示テスト"""
        with patch('tkinter.ttk.Style') as mock_style_cls:
            with patch('gui_framework.base_app.AppTheme') as mock_theme:
                with patch('tkinter.messagebox.showinfo') as mock_showinfo:
                    from gui_framework.base_app import BaseApp

                    class TestApp(BaseApp):
                        def _build_gui(self):
                            pass

                        def _on_close(self):
                            pass

                    mock_root = MagicMock()
                    mock_style_cls.return_value = MagicMock()

                    app = TestApp(mock_root, "Test App")
                    app.show_info("Info Title", "Info Message")

                    mock_showinfo.assert_called_once_with("Info Title", "Info Message")

    def test_ask_yes_no_returns_true(self):
        """Yes/Noダイアログ - Yes選択時テスト"""
        with patch('tkinter.ttk.Style') as mock_style_cls:
            with patch('gui_framework.base_app.AppTheme') as mock_theme:
                with patch('tkinter.messagebox.askyesno', return_value=True) as mock_askyesno:
                    from gui_framework.base_app import BaseApp

                    class TestApp(BaseApp):
                        def _build_gui(self):
                            pass

                        def _on_close(self):
                            pass

                    mock_root = MagicMock()
                    mock_style_cls.return_value = MagicMock()

                    app = TestApp(mock_root, "Test App")
                    result = app.ask_yes_no("Confirm", "Are you sure?")

                    assert result is True

    def test_ask_yes_no_returns_false(self):
        """Yes/Noダイアログ - No選択時テスト"""
        with patch('tkinter.ttk.Style') as mock_style_cls:
            with patch('gui_framework.base_app.AppTheme') as mock_theme:
                with patch('tkinter.messagebox.askyesno', return_value=False) as mock_askyesno:
                    from gui_framework.base_app import BaseApp

                    class TestApp(BaseApp):
                        def _build_gui(self):
                            pass

                        def _on_close(self):
                            pass

                    mock_root = MagicMock()
                    mock_style_cls.return_value = MagicMock()

                    app = TestApp(mock_root, "Test App")
                    result = app.ask_yes_no("Confirm", "Are you sure?")

                    assert result is False


class TestBaseAppRun:
    """run メソッドのテスト"""

    def test_run_calls_mainloop(self):
        """run メソッドが mainloop を呼び出すことを確認"""
        with patch('tkinter.ttk.Style') as mock_style_cls:
            with patch('gui_framework.base_app.AppTheme') as mock_theme:
                from gui_framework.base_app import BaseApp

                class TestApp(BaseApp):
                    def _build_gui(self):
                        pass

                    def _on_close(self):
                        pass

                mock_root = MagicMock()
                mock_style_cls.return_value = MagicMock()

                app = TestApp(mock_root, "Test App")
                app.run()

                mock_root.mainloop.assert_called_once()


class TestBaseAppProtocol:
    """プロトコルバインディングのテスト"""

    def test_close_protocol_bound(self):
        """WM_DELETE_WINDOW プロトコルがバインドされていることを確認"""
        with patch('tkinter.ttk.Style') as mock_style_cls:
            with patch('gui_framework.base_app.AppTheme') as mock_theme:
                from gui_framework.base_app import BaseApp

                class TestApp(BaseApp):
                    def _build_gui(self):
                        pass

                    def _on_close(self):
                        pass

                mock_root = MagicMock()
                mock_style_cls.return_value = MagicMock()

                app = TestApp(mock_root, "Test App")

                # protocol が呼ばれたことを確認
                mock_root.protocol.assert_called_with("WM_DELETE_WINDOW", app._on_close)


class TestBaseAppSetStatus:
    """set_status メソッドのテスト"""

    def test_set_status_default(self):
        """デフォルトの set_status は何もしない"""
        with patch('tkinter.ttk.Style') as mock_style_cls:
            with patch('gui_framework.base_app.AppTheme') as mock_theme:
                from gui_framework.base_app import BaseApp

                class TestApp(BaseApp):
                    def _build_gui(self):
                        pass

                    def _on_close(self):
                        pass

                mock_root = MagicMock()
                mock_style_cls.return_value = MagicMock()

                app = TestApp(mock_root, "Test App")
                app.set_status("Test status")  # 例外が発生しないことを確認
