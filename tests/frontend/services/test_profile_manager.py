"""
ProfileManager テスト

app/services/profile_manager.py のテスト
"""

import json
import pytest
import zipfile
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "app"))

from services.profile_manager import ProfileMetadata, ProfileManager


# =============================================================================
# TestProfileMetadata
# =============================================================================


class TestProfileMetadata:
    """ProfileMetadata データクラスのテスト"""

    def test_to_dict(self):
        """辞書変換テスト"""
        metadata = ProfileMetadata(
            id="prof_1",
            display_name="Test Profile",
            created_at="2024-01-01T00:00:00",
            last_accessed="2024-01-02T00:00:00",
            description="Test description"
        )
        result = metadata.to_dict()

        assert result["id"] == "prof_1"
        assert result["display_name"] == "Test Profile"
        assert result["created_at"] == "2024-01-01T00:00:00"
        assert result["last_accessed"] == "2024-01-02T00:00:00"
        assert result["description"] == "Test description"

    def test_from_dict(self):
        """辞書から作成テスト"""
        data = {
            "id": "prof_2",
            "display_name": "Another Profile",
            "created_at": "2024-03-01T10:00:00",
            "last_accessed": None,
            "description": "Another description"
        }
        metadata = ProfileMetadata.from_dict(data)

        assert metadata.id == "prof_2"
        assert metadata.display_name == "Another Profile"
        assert metadata.created_at == "2024-03-01T10:00:00"
        assert metadata.last_accessed is None
        assert metadata.description == "Another description"

    def test_default_timestamps(self):
        """デフォルトタイムスタンプテスト"""
        metadata = ProfileMetadata(
            id="prof_1",
            display_name="Test",
            created_at="2024-01-01T00:00:00"
        )

        assert metadata.last_accessed is None
        assert metadata.description == ""


# =============================================================================
# TestProfileManager
# =============================================================================


class TestProfileManager:
    """ProfileManager クラスのテスト"""

    def test_initialization(self, tmp_path: Path):
        """初期化テスト"""
        manager = ProfileManager(project_root=tmp_path)

        assert manager.profiles_dir == tmp_path / "profiles"
        assert manager.profiles_dir.exists()
        assert manager.registry_file.exists()

    def test_initialization_creates_default_profile(self, tmp_path: Path):
        """デフォルトプロファイル作成テスト"""
        manager = ProfileManager(project_root=tmp_path)

        profiles = manager.get_all_profiles()
        assert len(profiles) == 1
        assert profiles[0].id == "prof_1"
        assert profiles[0].display_name == "Default Profile"

    def test_create_profile(self, tmp_path: Path):
        """プロファイル作成テスト"""
        manager = ProfileManager(project_root=tmp_path)
        profile = manager.create_profile("New Profile", "New description")

        assert profile.display_name == "New Profile"
        assert profile.description == "New description"
        assert profile.id.startswith("prof_")

        # ディレクトリが作成されていることを確認
        profile_dir = manager.profiles_dir / profile.id
        assert profile_dir.exists()

    def test_create_profile_with_custom_id(self, tmp_path: Path):
        """カスタムID付きプロファイル作成テスト（自動生成ID）"""
        manager = ProfileManager(project_root=tmp_path)

        # 複数プロファイルを作成
        profile1 = manager.create_profile("Profile 1")
        profile2 = manager.create_profile("Profile 2")

        assert profile1.id == "prof_2"  # prof_1 はデフォルト
        assert profile2.id == "prof_3"

    def test_get_profile(self, tmp_path: Path):
        """プロファイル取得テスト"""
        manager = ProfileManager(project_root=tmp_path)
        manager.create_profile("Test Profile")

        profile = manager.get_profile("prof_1")
        assert profile is not None
        assert profile.id == "prof_1"

    def test_get_profile_not_found(self, tmp_path: Path):
        """プロファイル未発見テスト"""
        manager = ProfileManager(project_root=tmp_path)

        profile = manager.get_profile("nonexistent")
        assert profile is None

    def test_get_all_profiles(self, tmp_path: Path):
        """全プロファイル取得テスト"""
        manager = ProfileManager(project_root=tmp_path)
        manager.create_profile("Profile 2")
        manager.create_profile("Profile 3")

        profiles = manager.get_all_profiles()
        assert len(profiles) == 3

    def test_set_active_profile(self, tmp_path: Path):
        """アクティブプロファイル設定テスト"""
        manager = ProfileManager(project_root=tmp_path)
        new_profile = manager.create_profile("New Profile")

        manager.set_active_profile(new_profile.id)

        assert manager.get_active_profile_id() == new_profile.id

    def test_set_active_profile_not_found(self, tmp_path: Path):
        """存在しないプロファイルをアクティブにするテスト"""
        manager = ProfileManager(project_root=tmp_path)

        with pytest.raises(ValueError, match="Profile not found"):
            manager.set_active_profile("nonexistent")

    def test_get_active_profile_id(self, tmp_path: Path):
        """アクティブプロファイルID取得テスト"""
        manager = ProfileManager(project_root=tmp_path)

        active_id = manager.get_active_profile_id()
        assert active_id == "prof_1"

    def test_update_profile(self, tmp_path: Path):
        """プロファイル更新テスト"""
        manager = ProfileManager(project_root=tmp_path)

        updated = manager.update_profile(
            "prof_1",
            display_name="Updated Name",
            description="Updated description"
        )

        assert updated.display_name == "Updated Name"
        assert updated.description == "Updated description"

    def test_update_profile_name(self, tmp_path: Path):
        """プロファイル名更新テスト"""
        manager = ProfileManager(project_root=tmp_path)

        updated = manager.update_profile("prof_1", display_name="New Name")

        assert updated.display_name == "New Name"

        # 永続化を確認
        reloaded = manager.get_profile("prof_1")
        assert reloaded.display_name == "New Name"

    def test_delete_profile(self, tmp_path: Path):
        """プロファイル削除テスト"""
        manager = ProfileManager(project_root=tmp_path)
        new_profile = manager.create_profile("To Delete")

        # 削除前にアクティブプロファイルを変更
        manager.set_active_profile("prof_1")

        result = manager.delete_profile(new_profile.id)

        assert result is True
        assert manager.get_profile(new_profile.id) is None

    def test_cannot_delete_last_profile(self, tmp_path: Path):
        """最後のプロファイル削除不可テスト"""
        manager = ProfileManager(project_root=tmp_path)

        with pytest.raises(ValueError, match="Cannot delete the last profile"):
            manager.delete_profile("prof_1")

    def test_cannot_delete_active_profile(self, tmp_path: Path):
        """アクティブプロファイル削除不可テスト"""
        manager = ProfileManager(project_root=tmp_path)
        manager.create_profile("Second Profile")

        with pytest.raises(ValueError, match="Cannot delete the active profile"):
            manager.delete_profile("prof_1")

    def test_duplicate_profile(self, tmp_path: Path):
        """プロファイル複製テスト"""
        manager = ProfileManager(project_root=tmp_path)

        # オリジナルプロファイルにデータを追加
        original_path = manager.get_profile_path("prof_1")
        test_file = original_path / "app_data" / "test_data.json"
        test_file.write_text('{"test": "data"}')

        # 複製
        duplicated = manager.duplicate_profile("prof_1", "Duplicated Profile")

        assert duplicated.display_name == "Duplicated Profile"
        assert "Duplicated from prof_1" in duplicated.description

        # データがコピーされたことを確認
        duplicated_file = manager.get_profile_path(duplicated.id) / "app_data" / "test_data.json"
        assert duplicated_file.exists()

    def test_generate_profile_id(self, tmp_path: Path):
        """プロファイルID生成テスト"""
        manager = ProfileManager(project_root=tmp_path)

        # _generate_profile_id は private メソッドだが、
        # create_profile を通じてテスト
        for i in range(5):
            manager.create_profile(f"Profile {i+2}")

        profiles = manager.get_all_profiles()
        ids = [p.id for p in profiles]

        # ユニークなIDが生成されていることを確認
        assert len(ids) == len(set(ids))

    def test_get_profile_path(self, tmp_path: Path):
        """プロファイルパス取得テスト"""
        manager = ProfileManager(project_root=tmp_path)

        path = manager.get_profile_path("prof_1")

        assert path == tmp_path / "profiles" / "prof_1"
        assert path.exists()


# =============================================================================
# TestProfileExportImport
# =============================================================================


class TestProfileExportImport:
    """プロファイルのエクスポート・インポートテスト"""

    def test_export_profile(self, tmp_path: Path):
        """プロファイルエクスポートテスト"""
        manager = ProfileManager(project_root=tmp_path)

        # テストデータを追加
        profile_path = manager.get_profile_path("prof_1")
        test_file = profile_path / "app_data" / "registry.json"
        test_file.write_text('{"objects": []}')

        # エクスポート
        output_path = tmp_path / "export" / "profile_export.zip"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        result_path = manager.export_profile("prof_1", str(output_path))

        assert Path(result_path).exists()
        assert zipfile.is_zipfile(result_path)

    def test_export_profile_to_bytes(self, tmp_path: Path):
        """バイトへエクスポートテスト"""
        manager = ProfileManager(project_root=tmp_path)

        zip_bytes = manager.export_profile_to_bytes("prof_1")

        assert isinstance(zip_bytes, bytes)
        assert len(zip_bytes) > 0

    def test_import_profile(self, tmp_path: Path):
        """プロファイルインポートテスト"""
        manager = ProfileManager(project_root=tmp_path)

        # エクスポート
        output_path = tmp_path / "export" / "profile_export.zip"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        manager.export_profile("prof_1", str(output_path))

        # インポート
        imported = manager.import_profile(str(output_path), "Imported Profile")

        assert imported.display_name == "Imported Profile"
        assert imported.id != "prof_1"  # 新しいIDが生成される

    def test_safe_extract_zip_prevents_path_traversal(self, tmp_path: Path):
        """パストラバーサル防止テスト"""
        manager = ProfileManager(project_root=tmp_path)

        # 悪意のあるZIPファイルを作成
        malicious_zip = tmp_path / "malicious.zip"
        with zipfile.ZipFile(malicious_zip, 'w') as zf:
            # パストラバーサル試行
            zf.writestr("../../../etc/passwd", "malicious content")

        with pytest.raises(ValueError, match="path traversal"):
            with zipfile.ZipFile(malicious_zip, 'r') as zf:
                extract_dir = tmp_path / "extract"
                extract_dir.mkdir()
                manager._safe_extract_zip(zf, extract_dir)

    def test_resolve_duplicate_name(self, tmp_path: Path):
        """重複名解決テスト"""
        manager = ProfileManager(project_root=tmp_path)

        # 同じ名前のプロファイルを作成
        manager.create_profile("Test Profile")

        # _resolve_duplicate_name をテスト
        resolved = manager._resolve_duplicate_name("Default Profile")

        assert resolved == "Default Profile (2)"

    def test_import_invalid_zip(self, tmp_path: Path):
        """無効なZIPインポートテスト"""
        manager = ProfileManager(project_root=tmp_path)

        # 無効なファイルを作成
        invalid_file = tmp_path / "not_a_zip.txt"
        invalid_file.write_text("This is not a zip file")

        with pytest.raises(zipfile.BadZipFile):
            manager.import_profile(str(invalid_file))

    def test_import_too_large_zip(self, tmp_path: Path):
        """大きすぎるZIPインポートテスト"""
        manager = ProfileManager(project_root=tmp_path)

        # 大きなファイルをモック
        large_zip = tmp_path / "large.zip"
        with zipfile.ZipFile(large_zip, 'w') as zf:
            # ダミーファイルを追加
            zf.writestr("prof_test/dummy.txt", "test content")

        # ZipFile.infolist をモックして大きいサイズを返す
        with patch.object(zipfile.ZipFile, 'infolist') as mock_infolist:
            mock_info = MagicMock()
            mock_info.file_size = 5 * 1024 * 1024 * 1024  # 5GB
            mock_infolist.return_value = [mock_info]

            with pytest.raises(ValueError, match="ZIP file too large"):
                manager.import_profile(str(large_zip))
