"""
ObjectRegistry テスト

app/object_registry.py のテスト
"""

import json
import pytest
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "app"))

from object_registry import (
    ObjectVersion,
    ObjectProperties,
    RegisteredObject,
    ObjectRegistry
)


# =============================================================================
# TestObjectVersion
# =============================================================================


class TestObjectVersion:
    """ObjectVersion データクラスのテスト"""

    def test_default_values(self):
        """デフォルト値テスト"""
        version = ObjectVersion(version=1)

        assert version.version == 1
        assert version.image_path is None
        assert version.source_link is None

    def test_custom_values(self):
        """カスタム値テスト"""
        version = ObjectVersion(
            version=2,
            image_path="images/v2.jpg",
            source_link="https://example.com"
        )

        assert version.version == 2
        assert version.image_path == "images/v2.jpg"
        assert version.source_link == "https://example.com"


# =============================================================================
# TestObjectProperties
# =============================================================================


class TestObjectProperties:
    """ObjectProperties データクラスのテスト"""

    def test_default_values(self):
        """デフォルト値テスト"""
        props = ObjectProperties()

        assert props.is_heavy is False
        assert props.is_tiny is False
        assert props.has_liquid is False
        assert props.size_cm is None
        assert props.grasp_strategy is None

    def test_to_dict(self):
        """辞書変換テスト"""
        props = ObjectProperties(
            is_heavy=True,
            is_tiny=False,
            has_liquid=True,
            size_cm="10x5x5",
            grasp_strategy="side_grasp"
        )

        # dataclasses.asdict経由で辞書変換される
        from dataclasses import asdict
        result = asdict(props)

        assert result["is_heavy"] is True
        assert result["has_liquid"] is True
        assert result["size_cm"] == "10x5x5"

    def test_from_dict(self):
        """辞書から作成テスト"""
        data = {
            "is_heavy": False,
            "is_tiny": True,
            "has_liquid": False,
            "size_cm": "3x3x3",
            "grasp_strategy": "pinch"
        }

        props = ObjectProperties(**data)

        assert props.is_tiny is True
        assert props.size_cm == "3x3x3"


# =============================================================================
# TestRegisteredObject
# =============================================================================


class TestRegisteredObject:
    """RegisteredObject データクラスのテスト"""

    def test_to_dict(self):
        """辞書変換テスト"""
        obj = RegisteredObject(
            id=1,
            name="apple",
            display_name="Apple",
            category="Food",
            versions=[ObjectVersion(version=1)],
            properties=ObjectProperties(is_heavy=False),
            remarks="Red apple",
            target_samples=100,
            collected_samples=50
        )

        result = obj.to_dict()

        assert result["id"] == 1
        assert result["name"] == "apple"
        assert result["display_name"] == "Apple"
        assert result["category"] == "Food"
        assert len(result["versions"]) == 1
        assert result["properties"]["is_heavy"] is False

    def test_from_dict(self):
        """辞書から作成テスト"""
        data = {
            "id": 2,
            "name": "cup",
            "display_name": "Coffee Cup",
            "category": "Kitchen Item",
            "versions": [{"version": 1, "image_path": None, "source_link": None}],
            "properties": {"is_heavy": False, "is_tiny": False, "has_liquid": True},
            "remarks": "Blue cup",
            "target_samples": 80,
            "collected_samples": 40,
            "last_updated": "2024-01-01T00:00:00",
            "thumbnail_path": None
        }

        obj = RegisteredObject.from_dict(data)

        assert obj.id == 2
        assert obj.name == "cup"
        assert obj.display_name == "Coffee Cup"
        assert obj.properties.has_liquid is True

    def test_roundtrip_serialization(self):
        """往復シリアライズテスト"""
        original = RegisteredObject(
            id=3,
            name="bottle",
            display_name="Water Bottle",
            category="Drink",
            versions=[
                ObjectVersion(version=1, image_path="v1.jpg"),
                ObjectVersion(version=2, image_path="v2.jpg")
            ],
            properties=ObjectProperties(is_heavy=True, has_liquid=True),
            target_samples=120
        )

        # 辞書に変換して戻す
        data = original.to_dict()
        restored = RegisteredObject.from_dict(data)

        assert restored.id == original.id
        assert restored.name == original.name
        assert len(restored.versions) == len(original.versions)
        assert restored.properties.is_heavy == original.properties.is_heavy


# =============================================================================
# TestObjectRegistry
# =============================================================================


class TestObjectRegistry:
    """ObjectRegistry クラスのテスト"""

    @pytest.fixture
    def mock_path_coordinator(self, tmp_path):
        """PathCoordinatorのモック"""
        coordinator = MagicMock()
        app_data = tmp_path / "app_data"
        app_data.mkdir()

        coordinator.get_path.side_effect = lambda key: {
            "app_data_dir": app_data,
            "app_registry_file": app_data / "object_registry.json",
            "app_reference_dir": app_data / "reference_images",
            "raw_captures_dir": tmp_path / "raw_captures",
            "app_thumbnails_dir": app_data / "thumbnails",
        }[key]

        return coordinator

    def test_initialization(self, mock_path_coordinator):
        """初期化テスト"""
        registry = ObjectRegistry(path_coordinator=mock_path_coordinator)

        assert registry.data_dir.exists()
        assert isinstance(registry.objects, dict)
        assert isinstance(registry.categories, list)

    def test_initialization_creates_file(self, mock_path_coordinator):
        """初期化でファイル作成テスト"""
        registry = ObjectRegistry(path_coordinator=mock_path_coordinator)

        assert registry.registry_file.exists()

    def test_add_object(self, mock_path_coordinator):
        """オブジェクト追加テスト"""
        registry = ObjectRegistry(path_coordinator=mock_path_coordinator)

        obj = RegisteredObject(
            id=1,
            name="test_object",
            display_name="Test Object",
            category="Food"
        )

        registry.add_object(obj)

        assert 1 in registry.objects
        assert registry.objects[1].name == "test_object"

    def test_add_object_duplicate_name(self, mock_path_coordinator):
        """重複名オブジェクト追加テスト"""
        registry = ObjectRegistry(path_coordinator=mock_path_coordinator)

        obj1 = RegisteredObject(id=1, name="apple", display_name="Apple", category="Food")
        obj2 = RegisteredObject(id=2, name="apple", display_name="Apple 2", category="Food")

        registry.add_object(obj1)
        registry.add_object(obj2)

        # 両方とも追加される（IDが異なる）
        assert len(registry.objects) == 2

    def test_remove_object(self, mock_path_coordinator):
        """オブジェクト削除テスト"""
        registry = ObjectRegistry(path_coordinator=mock_path_coordinator)

        obj = RegisteredObject(id=1, name="to_remove", display_name="Remove Me", category="Food")
        registry.add_object(obj)

        result = registry.remove_object(1)

        assert result is True
        assert 1 not in registry.objects

    def test_get_object(self, mock_path_coordinator):
        """オブジェクト取得テスト"""
        registry = ObjectRegistry(path_coordinator=mock_path_coordinator)

        obj = RegisteredObject(id=1, name="apple", display_name="Apple", category="Food")
        registry.add_object(obj)

        retrieved = registry.get_object(1)

        assert retrieved is not None
        assert retrieved.name == "apple"

    def test_get_object_not_found(self, mock_path_coordinator):
        """オブジェクト未発見テスト"""
        registry = ObjectRegistry(path_coordinator=mock_path_coordinator)

        result = registry.get_object(999)

        assert result is None

    def test_get_object_by_name(self, mock_path_coordinator):
        """名前でオブジェクト取得テスト"""
        registry = ObjectRegistry(path_coordinator=mock_path_coordinator)

        obj = RegisteredObject(id=1, name="banana", display_name="Banana", category="Food")
        registry.add_object(obj)

        retrieved = registry.get_object_by_name("banana")

        assert retrieved is not None
        assert retrieved.id == 1

    def test_get_all_objects(self, mock_path_coordinator):
        """全オブジェクト取得テスト"""
        registry = ObjectRegistry(path_coordinator=mock_path_coordinator)

        for i in range(3):
            obj = RegisteredObject(
                id=i+1,
                name=f"object_{i}",
                display_name=f"Object {i}",
                category="Food"
            )
            registry.add_object(obj)

        all_objects = registry.get_all_objects()

        assert len(all_objects) == 3
        # IDでソートされている
        assert all_objects[0].id == 1

    def test_get_objects_by_category(self, mock_path_coordinator):
        """カテゴリでオブジェクト取得テスト"""
        registry = ObjectRegistry(path_coordinator=mock_path_coordinator)

        registry.add_object(RegisteredObject(id=1, name="apple", display_name="Apple", category="Food"))
        registry.add_object(RegisteredObject(id=2, name="cup", display_name="Cup", category="Kitchen Item"))
        registry.add_object(RegisteredObject(id=3, name="orange", display_name="Orange", category="Food"))

        food_objects = registry.get_objects_by_category("Food")

        assert len(food_objects) == 2

    def test_get_next_id(self, mock_path_coordinator):
        """次のID取得テスト"""
        registry = ObjectRegistry(path_coordinator=mock_path_coordinator)

        # 空の場合
        assert registry.get_next_id() == 1

        # オブジェクト追加後
        registry.add_object(RegisteredObject(id=1, name="obj1", display_name="Obj1", category="Food"))
        registry.add_object(RegisteredObject(id=5, name="obj5", display_name="Obj5", category="Food"))

        assert registry.get_next_id() == 6

    def test_add_category(self, mock_path_coordinator):
        """カテゴリ追加テスト"""
        registry = ObjectRegistry(path_coordinator=mock_path_coordinator)

        initial_count = len(registry.categories)

        registry.add_category("New Category")

        assert len(registry.categories) == initial_count + 1
        assert "New Category" in registry.categories

    def test_add_duplicate_category(self, mock_path_coordinator):
        """重複カテゴリ追加テスト"""
        registry = ObjectRegistry(path_coordinator=mock_path_coordinator)

        registry.add_category("Test Category")
        count_after_first = len(registry.categories)

        registry.add_category("Test Category")

        # 重複は追加されない
        assert len(registry.categories) == count_after_first

    def test_update_object(self, mock_path_coordinator):
        """オブジェクト更新テスト"""
        registry = ObjectRegistry(path_coordinator=mock_path_coordinator)

        obj = RegisteredObject(id=1, name="old_name", display_name="Old Name", category="Food")
        registry.add_object(obj)

        result = registry.update_object(1, {"display_name": "New Name", "category": "Drink"})

        assert result is True
        updated = registry.get_object(1)
        assert updated.display_name == "New Name"
        assert updated.category == "Drink"

    def test_update_object_name_renames_directories(self, mock_path_coordinator, tmp_path):
        """名前変更時ディレクトリリネームテスト"""
        registry = ObjectRegistry(path_coordinator=mock_path_coordinator)

        obj = RegisteredObject(id=1, name="old_name", display_name="Old", category="Food")
        registry.add_object(obj)

        # add_object がディレクトリを作成するので、そこにファイルを追加
        old_dir = registry.raw_captures_dir / "old_name"
        # ディレクトリが存在しない場合は作成
        old_dir.mkdir(parents=True, exist_ok=True)
        (old_dir / "test.jpg").write_text("dummy")

        result = registry.update_object(1, {"name": "new_name"})

        assert result is True
        # 新しいディレクトリが存在
        new_dir = registry.raw_captures_dir / "new_name"
        assert new_dir.exists()
        assert (new_dir / "test.jpg").exists()


# =============================================================================
# TestThumbnailManagement
# =============================================================================


class TestThumbnailManagement:
    """サムネイル管理のテスト"""

    @pytest.fixture
    def mock_path_coordinator(self, tmp_path):
        """PathCoordinatorのモック"""
        coordinator = MagicMock()
        app_data = tmp_path / "app_data"
        app_data.mkdir()

        coordinator.get_path.side_effect = lambda key: {
            "app_data_dir": app_data,
            "app_registry_file": app_data / "object_registry.json",
            "app_reference_dir": app_data / "reference_images",
            "raw_captures_dir": tmp_path / "raw_captures",
            "app_thumbnails_dir": app_data / "thumbnails",
        }[key]

        return coordinator

    def test_set_thumbnail(self, mock_path_coordinator, tmp_path):
        """サムネイル設定テスト"""
        registry = ObjectRegistry(path_coordinator=mock_path_coordinator)

        obj = RegisteredObject(id=1, name="test_obj", display_name="Test", category="Food")
        registry.add_object(obj)

        # ソース画像を作成
        source = tmp_path / "source_thumb.jpg"
        source.write_text("dummy image")

        result = registry.set_thumbnail(1, str(source))

        assert result is not None
        assert Path(result).exists()

    def test_save_thumbnail_from_bytes(self, mock_path_coordinator):
        """バイトからサムネイル保存テスト"""
        registry = ObjectRegistry(path_coordinator=mock_path_coordinator)

        obj = RegisteredObject(id=1, name="test_obj", display_name="Test", category="Food")
        registry.add_object(obj)

        image_data = b"dummy image bytes"

        result = registry.save_thumbnail_from_bytes(1, image_data)

        assert result is not None
        assert Path(result).exists()

    def test_get_thumbnail_path(self, mock_path_coordinator, tmp_path):
        """サムネイルパス取得テスト"""
        registry = ObjectRegistry(path_coordinator=mock_path_coordinator)

        obj = RegisteredObject(id=1, name="test_obj", display_name="Test", category="Food")
        registry.add_object(obj)

        # サムネイルを設定
        source = tmp_path / "thumb.jpg"
        source.write_text("dummy")
        registry.set_thumbnail(1, str(source))

        path = registry.get_thumbnail_path(1)

        assert path is not None
        assert Path(path).exists()

    def test_get_thumbnail_path_not_found(self, mock_path_coordinator):
        """サムネイルパス未発見テスト"""
        registry = ObjectRegistry(path_coordinator=mock_path_coordinator)

        obj = RegisteredObject(id=1, name="test_obj", display_name="Test", category="Food")
        registry.add_object(obj)

        # サムネイルなし
        path = registry.get_thumbnail_path(1)

        assert path is None


# =============================================================================
# TestReferenceImageManagement
# =============================================================================


class TestReferenceImageManagement:
    """参照画像管理のテスト"""

    @pytest.fixture
    def mock_path_coordinator(self, tmp_path):
        """PathCoordinatorのモック"""
        coordinator = MagicMock()
        app_data = tmp_path / "app_data"
        app_data.mkdir()

        coordinator.get_path.side_effect = lambda key: {
            "app_data_dir": app_data,
            "app_registry_file": app_data / "object_registry.json",
            "app_reference_dir": app_data / "reference_images",
            "raw_captures_dir": tmp_path / "raw_captures",
            "app_thumbnails_dir": app_data / "thumbnails",
        }[key]

        return coordinator

    def test_add_reference_image(self, mock_path_coordinator, tmp_path):
        """参照画像追加テスト"""
        registry = ObjectRegistry(path_coordinator=mock_path_coordinator)

        obj = RegisteredObject(id=1, name="test_obj", display_name="Test", category="Food")
        registry.add_object(obj)

        # ソース画像を作成
        source = tmp_path / "ref_image.jpg"
        source.write_text("dummy image")

        result = registry.add_reference_image(1, str(source), version=1)

        assert result is not None
        assert Path(result).exists()

    def test_get_reference_images(self, mock_path_coordinator, tmp_path):
        """参照画像取得テスト"""
        registry = ObjectRegistry(path_coordinator=mock_path_coordinator)

        obj = RegisteredObject(id=1, name="test_obj", display_name="Test", category="Food")
        registry.add_object(obj)

        # 参照画像を追加
        source = tmp_path / "ref.jpg"
        source.write_text("dummy")
        registry.add_reference_image(1, str(source), version=1)

        images = registry.get_reference_images(1)

        assert len(images) == 1

    def test_delete_reference_image(self, mock_path_coordinator, tmp_path):
        """参照画像削除テスト（現在は未実装だがテスト骨格）"""
        registry = ObjectRegistry(path_coordinator=mock_path_coordinator)

        obj = RegisteredObject(id=1, name="test_obj", display_name="Test", category="Food")
        registry.add_object(obj)

        # ObjectRegistryに delete_reference_image メソッドがある場合のテスト
        # 現在の実装にはないため、画像追加→versions更新でテスト
        source = tmp_path / "ref.jpg"
        source.write_text("dummy")
        registry.add_reference_image(1, str(source), version=1)

        # バージョンリストから手動削除するシナリオ
        updated_obj = registry.get_object(1)
        assert len(updated_obj.versions) >= 1


# =============================================================================
# TestCollectionManagement
# =============================================================================


class TestCollectionManagement:
    """収集画像管理のテスト"""

    @pytest.fixture
    def mock_path_coordinator(self, tmp_path):
        """PathCoordinatorのモック"""
        coordinator = MagicMock()
        app_data = tmp_path / "app_data"
        app_data.mkdir()

        coordinator.get_path.side_effect = lambda key: {
            "app_data_dir": app_data,
            "app_registry_file": app_data / "object_registry.json",
            "app_reference_dir": app_data / "reference_images",
            "raw_captures_dir": tmp_path / "raw_captures",
            "app_thumbnails_dir": app_data / "thumbnails",
        }[key]

        return coordinator

    def test_add_collected_image(self, mock_path_coordinator, tmp_path):
        """収集画像追加テスト"""
        registry = ObjectRegistry(path_coordinator=mock_path_coordinator)

        obj = RegisteredObject(id=1, name="test_obj", display_name="Test", category="Food")
        registry.add_object(obj)

        # ソース画像を作成
        source = tmp_path / "collected.jpg"
        source.write_text("dummy image")

        result = registry.add_collected_image(1, str(source))

        assert result is not None
        assert Path(result).exists()

    def test_save_collected_image(self, mock_path_coordinator):
        """収集画像保存テスト"""
        registry = ObjectRegistry(path_coordinator=mock_path_coordinator)

        obj = RegisteredObject(id=1, name="test_obj", display_name="Test", category="Food")
        registry.add_object(obj)

        image_data = b"dummy image bytes"

        result = registry.save_collected_image(1, image_data)

        assert result is not None
        assert Path(result).exists()

    def test_get_collected_images(self, mock_path_coordinator, tmp_path):
        """収集画像取得テスト"""
        registry = ObjectRegistry(path_coordinator=mock_path_coordinator)

        obj = RegisteredObject(id=1, name="test_obj", display_name="Test", category="Food")
        registry.add_object(obj)

        # 画像を追加
        for i in range(3):
            source = tmp_path / f"img_{i}.jpg"
            source.write_text("dummy")
            registry.add_collected_image(1, str(source))

        images = registry.get_collected_images(1)

        assert len(images) == 3

    def test_update_collection_count(self, mock_path_coordinator, tmp_path):
        """収集カウント更新テスト"""
        registry = ObjectRegistry(path_coordinator=mock_path_coordinator)

        obj = RegisteredObject(id=1, name="test_obj", display_name="Test", category="Food")
        registry.add_object(obj)

        # 画像を追加
        for i in range(5):
            source = tmp_path / f"img_{i}.jpg"
            source.write_text("dummy")
            registry.add_collected_image(1, str(source))

        count = registry.update_collection_count(1)

        assert count == 5
        assert registry.get_object(1).collected_samples == 5

    def test_update_all_collection_counts(self, mock_path_coordinator, tmp_path):
        """全収集カウント更新テスト"""
        registry = ObjectRegistry(path_coordinator=mock_path_coordinator)

        # 複数オブジェクトを追加
        for i in range(2):
            obj = RegisteredObject(id=i+1, name=f"obj_{i}", display_name=f"Obj {i}", category="Food")
            registry.add_object(obj)

            # 画像を追加
            for j in range(i + 1):
                source = tmp_path / f"img_{i}_{j}.jpg"
                source.write_text("dummy")
                registry.add_collected_image(i+1, str(source))

        registry.update_all_collection_counts()

        assert registry.get_object(1).collected_samples == 1
        assert registry.get_object(2).collected_samples == 2


# =============================================================================
# TestStatistics
# =============================================================================


class TestStatistics:
    """統計機能のテスト"""

    @pytest.fixture
    def mock_path_coordinator(self, tmp_path):
        """PathCoordinatorのモック"""
        coordinator = MagicMock()
        app_data = tmp_path / "app_data"
        app_data.mkdir()

        coordinator.get_path.side_effect = lambda key: {
            "app_data_dir": app_data,
            "app_registry_file": app_data / "object_registry.json",
            "app_reference_dir": app_data / "reference_images",
            "raw_captures_dir": tmp_path / "raw_captures",
            "app_thumbnails_dir": app_data / "thumbnails",
        }[key]

        return coordinator

    def test_get_collection_stats(self, mock_path_coordinator):
        """収集統計取得テスト"""
        registry = ObjectRegistry(path_coordinator=mock_path_coordinator)

        # オブジェクトを追加
        registry.add_object(RegisteredObject(
            id=1, name="obj1", display_name="Obj1", category="Food",
            target_samples=100, collected_samples=50
        ))
        registry.add_object(RegisteredObject(
            id=2, name="obj2", display_name="Obj2", category="Drink",
            target_samples=80, collected_samples=80
        ))

        stats = registry.get_collection_stats()

        assert stats["total_objects"] == 2
        assert stats["total_target"] == 180
        assert stats["total_collected"] == 130
        assert "by_category" in stats
        assert "Food" in stats["by_category"]

    def test_get_category_progress(self, mock_path_coordinator):
        """カテゴリ進捗取得テスト"""
        registry = ObjectRegistry(path_coordinator=mock_path_coordinator)

        registry.add_object(RegisteredObject(
            id=1, name="apple", display_name="Apple", category="Food",
            target_samples=100, collected_samples=60
        ))
        registry.add_object(RegisteredObject(
            id=2, name="orange", display_name="Orange", category="Food",
            target_samples=100, collected_samples=40
        ))

        stats = registry.get_collection_stats()

        food_stats = stats["by_category"]["Food"]
        assert food_stats["objects"] == 2
        assert food_stats["target"] == 200
        assert food_stats["collected"] == 100


# =============================================================================
# TestExport
# =============================================================================


class TestExport:
    """エクスポート機能のテスト"""

    @pytest.fixture
    def mock_path_coordinator(self, tmp_path):
        """PathCoordinatorのモック"""
        coordinator = MagicMock()
        app_data = tmp_path / "app_data"
        app_data.mkdir()

        coordinator.get_path.side_effect = lambda key: {
            "app_data_dir": app_data,
            "app_registry_file": app_data / "object_registry.json",
            "app_reference_dir": app_data / "reference_images",
            "raw_captures_dir": tmp_path / "raw_captures",
            "app_thumbnails_dir": app_data / "thumbnails",
        }[key]

        return coordinator

    def test_export_to_yolo_config(self, mock_path_coordinator, tmp_path):
        """YOLO設定エクスポートテスト"""
        registry = ObjectRegistry(path_coordinator=mock_path_coordinator)

        # オブジェクトを追加
        registry.add_object(RegisteredObject(
            id=1, name="apple", display_name="Apple", category="Food",
            target_samples=100, collected_samples=50
        ))
        registry.add_object(RegisteredObject(
            id=2, name="cup", display_name="Cup", category="Kitchen Item",
            target_samples=80, collected_samples=30
        ))

        output_path = tmp_path / "object_classes.json"
        result = registry.export_to_yolo_config(str(output_path))

        assert Path(result).exists()

        with open(result) as f:
            config = json.load(f)

        assert "categories" in config
        assert "objects" in config
        assert len(config["objects"]) == 2
