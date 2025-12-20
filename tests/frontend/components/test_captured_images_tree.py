"""
Captured Images Tree テスト

app/components/captured_images_tree.py のテスト
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock

# アプリパスを追加（インポート前に）
APP_PATH = str(Path(__file__).parent.parent.parent.parent / "app")
sys.path.insert(0, APP_PATH)

# Streamlitをモック（インポート前に）
mock_st = MagicMock()
mock_st.markdown = MagicMock()
mock_st.subheader = MagicMock()
mock_st.columns = MagicMock(return_value=[MagicMock(), MagicMock()])
mock_st.button = MagicMock(return_value=False)
mock_st.info = MagicMock()
mock_st.code = MagicMock()
mock_st.rerun = MagicMock()
sys.modules['streamlit'] = mock_st

# captured_images_treeモジュールをインポート
import types

captured_images_tree_path = Path(__file__).parent.parent.parent.parent / "app" / "components" / "captured_images_tree.py"
with open(captured_images_tree_path, 'r') as f:
    source = f.read()

# モジュールを作成して実行
captured_images_tree_module = types.ModuleType("captured_images_tree")
captured_images_tree_module.__file__ = str(captured_images_tree_path)
exec(compile(source, captured_images_tree_path, 'exec'), captured_images_tree_module.__dict__)

# テスト対象関数を取得
render_captured_images_tree = captured_images_tree_module.render_captured_images_tree


# =============================================================================
# TestRenderCapturedImagesTree
# =============================================================================


class TestRenderCapturedImagesTree:
    """render_captured_images_tree() のテスト"""

    def setup_method(self):
        """各テスト前にモックをリセット"""
        mock_st.reset_mock()
        mock_st.columns.return_value = [MagicMock(), MagicMock()]
        mock_st.button.return_value = False

    def test_header_rendered(self, tmp_path):
        """ヘッダー（サブヘッダーとセパレーター）が描画される"""
        mock_path_coordinator = MagicMock()
        mock_path_coordinator.get_path.return_value = tmp_path / "raw_captures"

        render_captured_images_tree(mock_path_coordinator)

        mock_st.markdown.assert_called_with("---")
        mock_st.subheader.assert_called_with("Collected Images")

    def test_directory_not_exists(self, tmp_path):
        """ディレクトリが存在しない場合、info表示"""
        mock_path_coordinator = MagicMock()
        nonexistent_dir = tmp_path / "nonexistent"
        mock_path_coordinator.get_path.return_value = nonexistent_dir

        render_captured_images_tree(mock_path_coordinator)

        mock_st.info.assert_called_with("Capture directory not created yet.")

    def test_empty_directory(self, tmp_path):
        """空のディレクトリの場合、info表示"""
        raw_captures_dir = tmp_path / "raw_captures"
        raw_captures_dir.mkdir()

        mock_path_coordinator = MagicMock()
        mock_path_coordinator.get_path.return_value = raw_captures_dir

        render_captured_images_tree(mock_path_coordinator)

        mock_st.info.assert_called_with("No captured images yet. Launch the Capture App to start collecting.")

    def test_with_subdirectories(self, tmp_path):
        """サブディレクトリがある場合、ツリー表示"""
        raw_captures_dir = tmp_path / "raw_captures"
        raw_captures_dir.mkdir()

        # サブディレクトリを作成
        subdir1 = raw_captures_dir / "class_a"
        subdir1.mkdir()
        subdir2 = raw_captures_dir / "class_b"
        subdir2.mkdir()

        mock_path_coordinator = MagicMock()
        mock_path_coordinator.get_path.return_value = raw_captures_dir

        render_captured_images_tree(mock_path_coordinator)

        mock_st.code.assert_called_once()

    def test_image_counting(self, tmp_path):
        """画像ファイルのカウントが正しい"""
        raw_captures_dir = tmp_path / "raw_captures"
        raw_captures_dir.mkdir()

        # サブディレクトリを作成
        subdir = raw_captures_dir / "class_a"
        subdir.mkdir()

        # 画像ファイルを作成
        (subdir / "img1.jpg").write_text("")
        (subdir / "img2.jpg").write_text("")
        (subdir / "img3.png").write_text("")

        mock_path_coordinator = MagicMock()
        mock_path_coordinator.get_path.return_value = raw_captures_dir

        render_captured_images_tree(mock_path_coordinator)

        # st.codeの引数をチェック
        call_args = mock_st.code.call_args[0][0]
        assert "3 images" in call_args

    def test_tree_format(self, tmp_path):
        """ツリー表示のフォーマットが正しい"""
        raw_captures_dir = tmp_path / "raw_captures"
        raw_captures_dir.mkdir()

        # サブディレクトリを作成
        (raw_captures_dir / "class_a").mkdir()
        (raw_captures_dir / "class_b").mkdir()

        mock_path_coordinator = MagicMock()
        mock_path_coordinator.get_path.return_value = raw_captures_dir

        render_captured_images_tree(mock_path_coordinator)

        # st.codeの引数をチェック
        call_args = mock_st.code.call_args[0][0]
        # ルートディレクトリ表示
        assert "📁 raw_captures/" in call_args
        # 中間サブディレクトリは├──
        assert "├──" in call_args
        # 最後のサブディレクトリは└──
        assert "└──" in call_args

    def test_single_subdirectory(self, tmp_path):
        """単一のサブディレクトリがある場合"""
        raw_captures_dir = tmp_path / "raw_captures"
        raw_captures_dir.mkdir()

        subdir = raw_captures_dir / "class_a"
        subdir.mkdir()

        mock_path_coordinator = MagicMock()
        mock_path_coordinator.get_path.return_value = raw_captures_dir

        render_captured_images_tree(mock_path_coordinator)

        # st.codeの引数をチェック
        call_args = mock_st.code.call_args[0][0]
        # 単一ディレクトリは└──
        assert "└──" in call_args
        assert "📂 class_a" in call_args

    def test_refresh_button_column_layout(self, tmp_path):
        """リフレッシュボタンが正しいカラムレイアウトで表示される"""
        mock_path_coordinator = MagicMock()
        mock_path_coordinator.get_path.return_value = tmp_path / "raw_captures"

        render_captured_images_tree(mock_path_coordinator)

        # st.columnsが[3, 1]で呼ばれる
        mock_st.columns.assert_called_with([3, 1])

    def test_refresh_button_with_correct_key(self, tmp_path):
        """リフレッシュボタンが正しいキーで表示される"""
        raw_captures_dir = tmp_path / "raw_captures"
        raw_captures_dir.mkdir()

        mock_path_coordinator = MagicMock()
        mock_path_coordinator.get_path.return_value = raw_captures_dir

        render_captured_images_tree(mock_path_coordinator)

        # st.buttonがrefresh_capturesキーで呼ばれる
        mock_st.button.assert_called_with("Refresh", key="refresh_captures")

    def test_mixed_file_types_in_subdirectory(self, tmp_path):
        """jpgとpng以外のファイルはカウントされない"""
        raw_captures_dir = tmp_path / "raw_captures"
        raw_captures_dir.mkdir()

        subdir = raw_captures_dir / "class_a"
        subdir.mkdir()

        # 各種ファイルを作成
        (subdir / "img1.jpg").write_text("")
        (subdir / "img2.png").write_text("")
        (subdir / "text.txt").write_text("")
        (subdir / "data.json").write_text("")

        mock_path_coordinator = MagicMock()
        mock_path_coordinator.get_path.return_value = raw_captures_dir

        render_captured_images_tree(mock_path_coordinator)

        # st.codeの引数をチェック（jpg + pngのみ = 2枚）
        call_args = mock_st.code.call_args[0][0]
        assert "2 images" in call_args

    def test_subdirectories_sorted_alphabetically(self, tmp_path):
        """サブディレクトリがアルファベット順にソートされる"""
        raw_captures_dir = tmp_path / "raw_captures"
        raw_captures_dir.mkdir()

        # 順序をばらばらに作成
        (raw_captures_dir / "zebra").mkdir()
        (raw_captures_dir / "apple").mkdir()
        (raw_captures_dir / "mango").mkdir()

        mock_path_coordinator = MagicMock()
        mock_path_coordinator.get_path.return_value = raw_captures_dir

        render_captured_images_tree(mock_path_coordinator)

        # st.codeの引数をチェック
        call_args = mock_st.code.call_args[0][0]
        lines = call_args.split('\n')

        # apple -> mango -> zebra の順序
        apple_idx = next(i for i, line in enumerate(lines) if "apple" in line)
        mango_idx = next(i for i, line in enumerate(lines) if "mango" in line)
        zebra_idx = next(i for i, line in enumerate(lines) if "zebra" in line)

        assert apple_idx < mango_idx < zebra_idx
