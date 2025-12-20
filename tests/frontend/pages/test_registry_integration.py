"""
Registry Integration テスト

app/pages/2_Registry.py のテスト
"""

import pytest
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses import dataclass, field
from typing import List, Optional


# =============================================================================
# モックセットアップ（インポート前に）
# =============================================================================

# Streamlitをモック
mock_st = MagicMock()
mock_st.session_state = MagicMock()
mock_st.title = MagicMock()
mock_st.tabs = MagicMock()
mock_st.selectbox = MagicMock(return_value="All")
mock_st.info = MagicMock()
mock_st.columns = MagicMock()
mock_st.expander = MagicMock()
mock_st.image = MagicMock()
mock_st.write = MagicMock()
mock_st.button = MagicMock(return_value=False)
mock_st.subheader = MagicMock()
mock_st.form = MagicMock()
mock_st.text_input = MagicMock()
mock_st.text_area = MagicMock()
mock_st.number_input = MagicMock()
mock_st.checkbox = MagicMock()
mock_st.success = MagicMock()
mock_st.error = MagicMock()
mock_st.markdown = MagicMock()
mock_st.form_submit_button = MagicMock()
mock_st.file_uploader = MagicMock(return_value=None)
mock_st.rerun = MagicMock()
mock_st.stop = MagicMock()
sys.modules['streamlit'] = mock_st

# コンポーネントをモック
mock_common_sidebar = MagicMock()
mock_common_sidebar.render_common_sidebar = MagicMock()
sys.modules['components'] = MagicMock()
sys.modules['components.common_sidebar'] = mock_common_sidebar

mock_object_editor = MagicMock()
mock_object_editor.render_object_editor = MagicMock(return_value=False)
sys.modules['components.object_editor'] = mock_object_editor

mock_object_viewer = MagicMock()
mock_object_viewer.render_object_viewer = MagicMock()
sys.modules['components.object_viewer'] = mock_object_viewer

mock_object_form = MagicMock()
mock_object_form.render_add_object_form = MagicMock()
sys.modules['components.object_form'] = mock_object_form

mock_thumbnail_upload = MagicMock()
mock_thumbnail_upload.render_thumbnail_upload = MagicMock(return_value=False)
mock_thumbnail_upload.render_thumbnail_upload_new = MagicMock(return_value=(None, ""))
mock_thumbnail_upload.clear_thumbnail_state = MagicMock()
mock_thumbnail_upload.clear_new_thumbnail_state = MagicMock()
sys.modules['components.thumbnail_upload'] = mock_thumbnail_upload


# =============================================================================
# テスト用データクラス
# =============================================================================

@dataclass
class MockObjectProperties:
    """ObjectPropertiesのモック"""
    is_heavy: bool = False
    is_tiny: bool = False
    has_liquid: bool = False
    size_cm: Optional[str] = None


@dataclass
class MockRegisteredObject:
    """RegisteredObjectのモック"""
    id: int
    name: str
    display_name: str
    category: str
    target_samples: int = 100
    collected_samples: int = 0
    remarks: Optional[str] = None
    properties: MockObjectProperties = field(default_factory=MockObjectProperties)
    versions: List[int] = field(default_factory=list)


# ObjectRegistryをモック
mock_object_registry_module = MagicMock()
mock_object_registry_module.ObjectRegistry = MagicMock
mock_object_registry_module.RegisteredObject = MockRegisteredObject
mock_object_registry_module.ObjectProperties = MockObjectProperties
sys.modules['object_registry'] = mock_object_registry_module


# =============================================================================
# モジュールのロード
# =============================================================================

APP_PATH = Path(__file__).parent.parent.parent.parent / "app"

# Registry ページのソースを読み込み
registry_page_path = APP_PATH / "pages" / "2_Registry.py"
with open(registry_page_path, 'r') as f:
    registry_source = f.read()

# 相対インポートを調整（すでにモック済み）
registry_module = types.ModuleType("registry_page")
registry_module.__file__ = str(registry_page_path)

# app_dirをモジュールに設定
registry_module.app_dir = APP_PATH

exec(compile(registry_source, registry_page_path, 'exec'), registry_module.__dict__)

# テスト対象関数を取得
show_registry_page = registry_module.show_registry_page
_render_view_tab = registry_module._render_view_tab


# object_viewer.py のソースを読み込み
object_viewer_path = APP_PATH / "components" / "object_viewer.py"
with open(object_viewer_path, 'r') as f:
    object_viewer_source = f.read()

# TYPE_CHECKING部分を除去
object_viewer_source = object_viewer_source.replace(
    "if TYPE_CHECKING:",
    "if False:"
)

object_viewer_module = types.ModuleType("object_viewer_impl")
object_viewer_module.__file__ = str(object_viewer_path)
exec(compile(object_viewer_source, object_viewer_path, 'exec'), object_viewer_module.__dict__)

render_object_viewer_impl = object_viewer_module.render_object_viewer
_render_action_buttons = object_viewer_module._render_action_buttons
_render_object_details = object_viewer_module._render_object_details


# object_form.py のソースを読み込み
object_form_path = APP_PATH / "components" / "object_form.py"
with open(object_form_path, 'r') as f:
    object_form_source = f.read()

# TYPE_CHECKING部分を除去
object_form_source = object_form_source.replace(
    "if TYPE_CHECKING:",
    "if False:"
)

object_form_module = types.ModuleType("object_form_impl")
object_form_module.__file__ = str(object_form_path)
object_form_module.RegisteredObject = MockRegisteredObject
object_form_module.ObjectProperties = MockObjectProperties
exec(compile(object_form_source, object_form_path, 'exec'), object_form_module.__dict__)

render_add_object_form_impl = object_form_module.render_add_object_form
_handle_form_submit = object_form_module._handle_form_submit


# =============================================================================
# フィクスチャ
# =============================================================================

@pytest.fixture
def mock_registry():
    """ObjectRegistryのモック"""
    registry = MagicMock()
    registry.categories = ["food", "container", "tool"]
    registry.get_all_objects = MagicMock(return_value=[])
    registry.get_next_id = MagicMock(return_value=1)
    registry.get_object_by_name = MagicMock(return_value=None)
    registry.get_thumbnail_path = MagicMock(return_value=None)
    registry.get_reference_images = MagicMock(return_value=[])
    registry.add_object = MagicMock()
    registry.update_object = MagicMock()
    registry.remove_object = MagicMock()
    return registry


@pytest.fixture
def sample_objects():
    """テスト用オブジェクトリスト"""
    return [
        MockRegisteredObject(
            id=1,
            name="redbull",
            display_name="Redbull",
            category="food",
            target_samples=100,
            collected_samples=50,
        ),
        MockRegisteredObject(
            id=2,
            name="water_bottle",
            display_name="Water Bottle",
            category="container",
            target_samples=80,
            collected_samples=80,
        ),
        MockRegisteredObject(
            id=3,
            name="banana",
            display_name="Banana",
            category="food",
            target_samples=100,
            collected_samples=30,
        ),
    ]


# =============================================================================
# TestViewObjects
# =============================================================================

class TestViewObjects:
    """オブジェクト一覧表示のテスト"""

    def setup_method(self):
        """各テスト前にモックをリセット"""
        mock_st.reset_mock()
        mock_object_viewer.reset_mock()

    def test_view_objects_empty_list(self, mock_registry):
        """オブジェクトがない場合、infoメッセージが表示される"""
        mock_registry.get_all_objects.return_value = []

        _render_view_tab(mock_registry)

        mock_st.info.assert_called_once()
        assert "No objects registered" in mock_st.info.call_args[0][0]

    def test_view_objects_with_data(self, mock_registry, sample_objects):
        """オブジェクトがある場合、一覧が表示される"""
        mock_registry.get_all_objects.return_value = sample_objects
        mock_st.selectbox.return_value = "All"

        # columnsとexpanderのコンテキストマネージャをモック
        mock_col = MagicMock()
        mock_col.__enter__ = MagicMock(return_value=mock_col)
        mock_col.__exit__ = MagicMock(return_value=None)
        mock_st.columns.return_value = [mock_col, mock_col]

        mock_expander = MagicMock()
        mock_expander.__enter__ = MagicMock(return_value=mock_expander)
        mock_expander.__exit__ = MagicMock(return_value=None)
        mock_st.expander.return_value = mock_expander

        _render_view_tab(mock_registry)

        # selectboxでカテゴリフィルタが表示される
        mock_st.selectbox.assert_called()
        # expanderが各オブジェクトに対して呼ばれる
        assert mock_st.expander.call_count == 3

    def test_view_objects_calls_viewer(self, mock_registry, sample_objects):
        """各オブジェクトに対してビューアが呼ばれる"""
        mock_registry.get_all_objects.return_value = sample_objects
        mock_st.selectbox.return_value = "All"

        # session_stateにedit_mode設定
        mock_st.session_state.__getitem__ = MagicMock(return_value=False)
        mock_st.session_state.__contains__ = MagicMock(return_value=True)

        mock_col = MagicMock()
        mock_col.__enter__ = MagicMock(return_value=mock_col)
        mock_col.__exit__ = MagicMock(return_value=None)
        mock_st.columns.return_value = [mock_col, mock_col]

        mock_expander = MagicMock()
        mock_expander.__enter__ = MagicMock(return_value=mock_expander)
        mock_expander.__exit__ = MagicMock(return_value=None)
        mock_st.expander.return_value = mock_expander

        _render_view_tab(mock_registry)

        # object_viewerのrender_object_viewerが呼ばれる
        assert mock_object_viewer.render_object_viewer.call_count == 3


# =============================================================================
# TestAddObjectForm
# =============================================================================

class TestAddObjectForm:
    """オブジェクト追加フォームのテスト"""

    def setup_method(self):
        """各テスト前にモックをリセット"""
        mock_st.reset_mock()
        mock_object_form.reset_mock()

    def test_add_object_form_renders(self, mock_registry):
        """追加フォームがレンダリングされる"""
        # タブのコンテキストマネージャをモック
        mock_tab = MagicMock()
        mock_tab.__enter__ = MagicMock(return_value=mock_tab)
        mock_tab.__exit__ = MagicMock(return_value=None)
        mock_st.tabs.return_value = [mock_tab, mock_tab]

        mock_st.session_state.registry = mock_registry

        show_registry_page()

        # タブが作成される
        mock_st.tabs.assert_called_once()
        # render_add_object_formが呼ばれる
        mock_object_form.render_add_object_form.assert_called_once_with(mock_registry)

    def test_add_object_form_elements(self, mock_registry):
        """追加フォームの要素がレンダリングされる"""
        mock_form = MagicMock()
        mock_form.__enter__ = MagicMock(return_value=mock_form)
        mock_form.__exit__ = MagicMock(return_value=None)
        mock_st.form.return_value = mock_form

        mock_col = MagicMock()
        mock_col.__enter__ = MagicMock(return_value=mock_col)
        mock_col.__exit__ = MagicMock(return_value=None)
        mock_st.columns.return_value = [mock_col, mock_col]

        mock_st.form_submit_button.return_value = False

        render_add_object_form_impl(mock_registry)

        # subheaderが表示される
        mock_st.subheader.assert_called_with("Add New Object")
        # formが作成される
        mock_st.form.assert_called_with("add_object_form")

    def test_add_object_validates_name(self, mock_registry):
        """名前が空の場合、エラーが表示される"""
        _handle_form_submit(
            registry=mock_registry,
            new_id=1,
            new_name="",  # 空の名前
            new_display_name="Test",
            new_category="food",
            new_target=100,
            new_remarks="",
            is_heavy=False,
            is_tiny=False,
            has_liquid=False,
            size_cm="",
        )

        mock_st.error.assert_called()
        assert "required" in mock_st.error.call_args[0][0]

    def test_add_object_validates_duplicate_name(self, mock_registry, sample_objects):
        """重複名の場合、エラーが表示される"""
        mock_registry.get_object_by_name.return_value = sample_objects[0]

        _handle_form_submit(
            registry=mock_registry,
            new_id=2,
            new_name="redbull",  # 既存の名前
            new_display_name="Redbull 2",
            new_category="food",
            new_target=100,
            new_remarks="",
            is_heavy=False,
            is_tiny=False,
            has_liquid=False,
            size_cm="",
        )

        mock_st.error.assert_called()
        assert "already exists" in mock_st.error.call_args[0][0]


# =============================================================================
# TestEditObject
# =============================================================================

class TestEditObject:
    """オブジェクト編集のテスト"""

    def setup_method(self):
        """各テスト前にモックをリセット"""
        mock_st.reset_mock()
        mock_object_editor.reset_mock()

    def test_edit_mode_toggle(self, mock_registry, sample_objects):
        """編集モードへの切り替え"""
        mock_registry.get_all_objects.return_value = sample_objects[:1]
        mock_st.selectbox.return_value = "All"

        # 最初はedit_modeがFalse
        edit_states = {"edit_mode_1": False}

        def getitem_side_effect(key):
            return edit_states.get(key, False)

        def setitem_side_effect(key, value):
            edit_states[key] = value

        mock_st.session_state.__getitem__ = MagicMock(side_effect=getitem_side_effect)
        mock_st.session_state.__setitem__ = MagicMock(side_effect=setitem_side_effect)
        mock_st.session_state.__contains__ = MagicMock(return_value=True)

        mock_col = MagicMock()
        mock_col.__enter__ = MagicMock(return_value=mock_col)
        mock_col.__exit__ = MagicMock(return_value=None)
        mock_st.columns.return_value = [mock_col, mock_col]

        mock_expander = MagicMock()
        mock_expander.__enter__ = MagicMock(return_value=mock_expander)
        mock_expander.__exit__ = MagicMock(return_value=None)
        mock_st.expander.return_value = mock_expander

        _render_view_tab(mock_registry)

        # edit_mode=Falseなのでビューアが呼ばれる
        mock_object_viewer.render_object_viewer.assert_called()

    def test_edit_mode_shows_editor(self, mock_registry, sample_objects):
        """edit_mode=Trueの場合、エディタが表示される"""
        mock_registry.get_all_objects.return_value = sample_objects[:1]
        mock_st.selectbox.return_value = "All"

        # edit_modeがTrue
        edit_states = {"edit_mode_1": True}

        def getitem_side_effect(key):
            return edit_states.get(key, False)

        mock_st.session_state.__getitem__ = MagicMock(side_effect=getitem_side_effect)
        mock_st.session_state.__contains__ = MagicMock(return_value=True)

        mock_col = MagicMock()
        mock_col.__enter__ = MagicMock(return_value=mock_col)
        mock_col.__exit__ = MagicMock(return_value=None)
        mock_st.columns.return_value = [mock_col, mock_col]

        mock_expander = MagicMock()
        mock_expander.__enter__ = MagicMock(return_value=mock_expander)
        mock_expander.__exit__ = MagicMock(return_value=None)
        mock_st.expander.return_value = mock_expander

        _render_view_tab(mock_registry)

        # edit_mode=Trueなのでエディタが呼ばれる
        mock_object_editor.render_object_editor.assert_called()


# =============================================================================
# TestDeleteObject
# =============================================================================

class TestDeleteObject:
    """オブジェクト削除のテスト"""

    def setup_method(self):
        """各テスト前にモックをリセット"""
        mock_st.reset_mock()

    def test_delete_button_calls_remove(self, mock_registry, sample_objects):
        """削除ボタンでremove_objectが呼ばれる"""
        obj = sample_objects[0]

        # ボタンのモック（Deleteボタンだけクリックされた状態）
        def button_side_effect(label, **kwargs):
            if "Delete" in label:
                return True
            return False

        mock_st.button.side_effect = button_side_effect

        mock_col = MagicMock()
        mock_col.__enter__ = MagicMock(return_value=mock_col)
        mock_col.__exit__ = MagicMock(return_value=None)
        mock_st.columns.return_value = [mock_col, mock_col, mock_col]

        _render_action_buttons(obj, mock_registry)

        # remove_objectが呼ばれる
        mock_registry.remove_object.assert_called_once_with(obj.id)

    def test_delete_triggers_rerun(self, mock_registry, sample_objects):
        """削除後にrerunが呼ばれる"""
        obj = sample_objects[0]

        def button_side_effect(label, **kwargs):
            if "Delete" in label:
                return True
            return False

        mock_st.button.side_effect = button_side_effect

        mock_col = MagicMock()
        mock_col.__enter__ = MagicMock(return_value=mock_col)
        mock_col.__exit__ = MagicMock(return_value=None)
        mock_st.columns.return_value = [mock_col, mock_col, mock_col]

        _render_action_buttons(obj, mock_registry)

        mock_st.rerun.assert_called()


# =============================================================================
# TestFilterByCategory
# =============================================================================

class TestFilterByCategory:
    """カテゴリフィルタのテスト"""

    def setup_method(self):
        """各テスト前にモックをリセット"""
        mock_st.reset_mock()

    def test_filter_all_shows_all_objects(self, mock_registry, sample_objects):
        """'All'フィルタで全オブジェクトが表示される"""
        mock_registry.get_all_objects.return_value = sample_objects
        mock_st.selectbox.return_value = "All"

        mock_col = MagicMock()
        mock_col.__enter__ = MagicMock(return_value=mock_col)
        mock_col.__exit__ = MagicMock(return_value=None)
        mock_st.columns.return_value = [mock_col, mock_col]

        mock_expander = MagicMock()
        mock_expander.__enter__ = MagicMock(return_value=mock_expander)
        mock_expander.__exit__ = MagicMock(return_value=None)
        mock_st.expander.return_value = mock_expander

        mock_st.session_state.__getitem__ = MagicMock(return_value=False)
        mock_st.session_state.__contains__ = MagicMock(return_value=True)

        _render_view_tab(mock_registry)

        # 全3オブジェクトに対してexpanderが呼ばれる
        assert mock_st.expander.call_count == 3

    def test_filter_category_shows_filtered(self, mock_registry, sample_objects):
        """カテゴリフィルタでフィルタされたオブジェクトのみ表示"""
        mock_registry.get_all_objects.return_value = sample_objects
        mock_st.selectbox.return_value = "food"  # foodカテゴリでフィルタ

        mock_col = MagicMock()
        mock_col.__enter__ = MagicMock(return_value=mock_col)
        mock_col.__exit__ = MagicMock(return_value=None)
        mock_st.columns.return_value = [mock_col, mock_col]

        mock_expander = MagicMock()
        mock_expander.__enter__ = MagicMock(return_value=mock_expander)
        mock_expander.__exit__ = MagicMock(return_value=None)
        mock_st.expander.return_value = mock_expander

        mock_st.session_state.__getitem__ = MagicMock(return_value=False)
        mock_st.session_state.__contains__ = MagicMock(return_value=True)

        _render_view_tab(mock_registry)

        # foodカテゴリは2つ（redbull, banana）
        assert mock_st.expander.call_count == 2

    def test_filter_category_container(self, mock_registry, sample_objects):
        """containerカテゴリでフィルタ"""
        mock_registry.get_all_objects.return_value = sample_objects
        mock_st.selectbox.return_value = "container"

        mock_col = MagicMock()
        mock_col.__enter__ = MagicMock(return_value=mock_col)
        mock_col.__exit__ = MagicMock(return_value=None)
        mock_st.columns.return_value = [mock_col, mock_col]

        mock_expander = MagicMock()
        mock_expander.__enter__ = MagicMock(return_value=mock_expander)
        mock_expander.__exit__ = MagicMock(return_value=None)
        mock_st.expander.return_value = mock_expander

        mock_st.session_state.__getitem__ = MagicMock(return_value=False)
        mock_st.session_state.__contains__ = MagicMock(return_value=True)

        _render_view_tab(mock_registry)

        # containerカテゴリは1つ（water_bottle）
        assert mock_st.expander.call_count == 1

    def test_selectbox_includes_all_option(self, mock_registry, sample_objects):
        """selectboxに'All'オプションが含まれる"""
        mock_registry.get_all_objects.return_value = sample_objects
        mock_st.selectbox.return_value = "All"

        mock_col = MagicMock()
        mock_col.__enter__ = MagicMock(return_value=mock_col)
        mock_col.__exit__ = MagicMock(return_value=None)
        mock_st.columns.return_value = [mock_col, mock_col]

        mock_expander = MagicMock()
        mock_expander.__enter__ = MagicMock(return_value=mock_expander)
        mock_expander.__exit__ = MagicMock(return_value=None)
        mock_st.expander.return_value = mock_expander

        mock_st.session_state.__getitem__ = MagicMock(return_value=False)
        mock_st.session_state.__contains__ = MagicMock(return_value=True)

        _render_view_tab(mock_registry)

        # selectboxの呼び出しを確認
        call_args = mock_st.selectbox.call_args
        categories = call_args[0][1]  # 第2引数がカテゴリリスト
        assert "All" in categories


# =============================================================================
# TestObjectDetails
# =============================================================================

class TestObjectDetails:
    """オブジェクト詳細表示のテスト"""

    def setup_method(self):
        """各テスト前にモックをリセット"""
        mock_st.reset_mock()

    def test_details_shows_name(self, sample_objects):
        """名前が表示される"""
        obj = sample_objects[0]

        _render_object_details(obj)

        # writeが呼ばれて名前が含まれる
        calls = [str(call) for call in mock_st.write.call_args_list]
        assert any("redbull" in call for call in calls)

    def test_details_shows_category(self, sample_objects):
        """カテゴリが表示される"""
        obj = sample_objects[0]

        _render_object_details(obj)

        calls = [str(call) for call in mock_st.write.call_args_list]
        assert any("food" in call for call in calls)

    def test_details_shows_target_samples(self, sample_objects):
        """目標サンプル数が表示される"""
        obj = sample_objects[0]

        _render_object_details(obj)

        calls = [str(call) for call in mock_st.write.call_args_list]
        assert any("100" in call for call in calls)

    def test_details_shows_properties(self):
        """プロパティが表示される"""
        obj = MockRegisteredObject(
            id=1,
            name="test",
            display_name="Test",
            category="food",
            properties=MockObjectProperties(
                is_heavy=True,
                is_tiny=False,
                has_liquid=True,
                size_cm="10x10x10"
            )
        )

        _render_object_details(obj)

        calls = [str(call) for call in mock_st.write.call_args_list]
        # プロパティが含まれる
        assert any("Heavy" in call for call in calls)
        assert any("Liquid" in call for call in calls)
