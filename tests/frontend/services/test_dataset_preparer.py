"""
DatasetPreparer テスト

app/services/dataset_preparer.py のテスト
"""

import pytest
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "app"))

from services.dataset_preparer import (
    _extract_timestamp,
    _group_by_timestamp,
    ClassInfo,
    DatasetResult,
    DatasetPreparer
)


# =============================================================================
# TestHelperFunctions
# =============================================================================


class TestHelperFunctions:
    """ヘルパー関数のテスト"""

    def test_extract_timestamp(self):
        """タイムスタンプ抽出テスト"""
        filename = "apple_20251211_123456_123.jpg"
        result = _extract_timestamp(filename)

        assert result is not None
        assert result.year == 2025
        assert result.month == 12
        assert result.day == 11
        assert result.hour == 12
        assert result.minute == 34
        assert result.second == 56

    def test_extract_timestamp_invalid_format(self):
        """無効なフォーマットのタイムスタンプ抽出テスト"""
        filename = "some_random_file.jpg"
        result = _extract_timestamp(filename)

        assert result is None

    def test_group_by_timestamp(self, tmp_path):
        """タイムスタンプによるグループ化テスト"""
        # テスト用のペアを作成
        pairs = [
            (tmp_path / "apple_20251211_120000_001.jpg", tmp_path / "label1.txt", "apple"),
            (tmp_path / "apple_20251211_120001_001.jpg", tmp_path / "label2.txt", "apple"),  # 1秒後
            (tmp_path / "apple_20251211_120010_001.jpg", tmp_path / "label3.txt", "apple"),  # 10秒後
        ]

        # 2秒間隔でグループ化
        groups = _group_by_timestamp(pairs, interval_sec=2.0)

        # 最初の2つは同じグループ、3つ目は別グループ
        assert len(groups) == 2

    def test_group_by_timestamp_empty(self):
        """空リストのグループ化テスト"""
        groups = _group_by_timestamp([], interval_sec=2.0)
        assert groups == []


# =============================================================================
# TestClassInfo
# =============================================================================


class TestClassInfo:
    """ClassInfo データクラスのテスト"""

    def test_match_ratio(self, tmp_path):
        """マッチ率テスト"""
        info = ClassInfo(
            name="apple",
            image_count=100,
            label_count=80,
            matched_count=80,
            images_dir=tmp_path / "apple",
            labels_dir=tmp_path / "labels"
        )

        assert info.match_ratio == 0.8

    def test_match_ratio_zero_images(self, tmp_path):
        """画像0枚時のマッチ率テスト"""
        info = ClassInfo(
            name="apple",
            image_count=0,
            label_count=0,
            matched_count=0,
            images_dir=tmp_path / "apple",
            labels_dir=tmp_path / "labels"
        )

        assert info.match_ratio == 0.0

    def test_is_ready(self, tmp_path):
        """準備完了判定テスト"""
        ready_info = ClassInfo(
            name="apple",
            image_count=50,
            label_count=50,
            matched_count=50,
            images_dir=tmp_path / "apple",
            labels_dir=tmp_path / "labels"
        )

        not_ready_info = ClassInfo(
            name="orange",
            image_count=5,
            label_count=5,
            matched_count=5,
            images_dir=tmp_path / "orange",
            labels_dir=tmp_path / "labels"
        )

        assert ready_info.is_ready is True
        assert not_ready_info.is_ready is False

    def test_status(self, tmp_path):
        """ステータス判定テスト"""
        no_data = ClassInfo(
            name="a", image_count=10, label_count=0, matched_count=0,
            images_dir=tmp_path / "a", labels_dir=tmp_path / "l"
        )
        insufficient = ClassInfo(
            name="b", image_count=10, label_count=5, matched_count=5,
            images_dir=tmp_path / "b", labels_dir=tmp_path / "l"
        )
        partial = ClassInfo(
            name="c", image_count=20, label_count=15, matched_count=15,
            images_dir=tmp_path / "c", labels_dir=tmp_path / "l"
        )
        ready = ClassInfo(
            name="d", image_count=20, label_count=20, matched_count=20,
            images_dir=tmp_path / "d", labels_dir=tmp_path / "l"
        )

        assert no_data.status == "no_data"
        assert insufficient.status == "insufficient"
        assert partial.status == "partial"
        assert ready.status == "ready"


# =============================================================================
# TestDatasetResult
# =============================================================================


class TestDatasetResult:
    """DatasetResult データクラスのテスト"""

    def test_success_result(self, tmp_path):
        """成功結果テスト"""
        result = DatasetResult(
            success=True,
            output_dir=tmp_path / "output",
            train_count=80,
            val_count=20,
            class_names=["apple", "banana"]
        )

        assert result.success is True
        assert result.train_count == 80
        assert result.val_count == 20
        assert len(result.class_names) == 2

    def test_failure_result(self):
        """失敗結果テスト"""
        result = DatasetResult(
            success=False,
            output_dir=None,
            train_count=0,
            val_count=0,
            class_names=[],
            error_message="No data found"
        )

        assert result.success is False
        assert result.error_message == "No data found"


# =============================================================================
# TestDatasetPreparer
# =============================================================================


class TestDatasetPreparer:
    """DatasetPreparer クラスのテスト"""

    @pytest.fixture
    def mock_path_coordinator(self, tmp_path):
        """PathCoordinatorのモック"""
        coordinator = MagicMock()

        raw_captures = tmp_path / "raw_captures"
        annotated = tmp_path / "annotated"
        raw_captures.mkdir()
        annotated.mkdir()

        coordinator.get_path.side_effect = lambda key: {
            "raw_captures_dir": raw_captures,
            "annotated_dir": annotated,
        }[key]

        return coordinator

    def test_initialization(self, mock_path_coordinator):
        """初期化テスト"""
        preparer = DatasetPreparer(path_coordinator=mock_path_coordinator)

        assert preparer.path_coordinator == mock_path_coordinator

    def test_prepare_dataset(self, mock_path_coordinator, tmp_path):
        """データセット準備テスト"""
        preparer = DatasetPreparer(path_coordinator=mock_path_coordinator)

        # テストデータを作成
        raw_captures = tmp_path / "raw_captures"
        annotated = tmp_path / "annotated"

        # apple クラス
        apple_dir = raw_captures / "apple"
        apple_dir.mkdir()
        apple_labels = annotated / "apple" / "labels"
        apple_labels.mkdir(parents=True)

        for i in range(15):
            img = apple_dir / f"apple_{i}.jpg"
            img.write_text("dummy")
            label = apple_labels / f"apple_{i}.txt"
            label.write_text("0 0.5 0.5 0.2 0.2")

        result = preparer.prepare_dataset(
            class_names=["apple"],
            output_name="test_dataset",
            val_ratio=0.2,
            group_continuous_frames=False
        )

        assert result.success is True
        assert result.train_count + result.val_count == 15

    def test_prepare_dataset_with_split(self, mock_path_coordinator, tmp_path):
        """分割付きデータセット準備テスト"""
        preparer = DatasetPreparer(path_coordinator=mock_path_coordinator)

        # テストデータを作成
        raw_captures = tmp_path / "raw_captures"
        annotated = tmp_path / "annotated"

        apple_dir = raw_captures / "apple"
        apple_dir.mkdir()
        apple_labels = annotated / "apple" / "labels"
        apple_labels.mkdir(parents=True)

        for i in range(20):
            img = apple_dir / f"apple_{i}.jpg"
            img.write_text("dummy")
            label = apple_labels / f"apple_{i}.txt"
            label.write_text("0 0.5 0.5 0.2 0.2")

        result = preparer.prepare_dataset(
            class_names=["apple"],
            output_name="split_dataset",
            val_ratio=0.3,
            group_continuous_frames=False
        )

        assert result.success is True
        # 30%がvalidation
        assert result.val_count >= 5

    def test_validate_annotations(self, mock_path_coordinator, tmp_path):
        """アノテーション検証テスト"""
        preparer = DatasetPreparer(path_coordinator=mock_path_coordinator)

        # get_available_classes を通じて検証
        raw_captures = tmp_path / "raw_captures"
        annotated = tmp_path / "annotated"

        apple_dir = raw_captures / "apple"
        apple_dir.mkdir()

        for i in range(5):
            (apple_dir / f"apple_{i}.jpg").write_text("dummy")

        classes = preparer.get_available_classes()

        assert len(classes) == 1
        assert classes[0].name == "apple"
        assert classes[0].image_count == 5
        assert classes[0].matched_count == 0  # ラベルなし

    def test_create_yaml(self, mock_path_coordinator, tmp_path):
        """YAML作成テスト"""
        preparer = DatasetPreparer(path_coordinator=mock_path_coordinator)

        # テストデータを作成
        raw_captures = tmp_path / "raw_captures"
        annotated = tmp_path / "annotated"

        apple_dir = raw_captures / "apple"
        apple_dir.mkdir()
        apple_labels = annotated / "apple" / "labels"
        apple_labels.mkdir(parents=True)

        for i in range(10):
            (apple_dir / f"apple_{i}.jpg").write_text("dummy")
            (apple_labels / f"apple_{i}.txt").write_text("0 0.5 0.5 0.2 0.2")

        result = preparer.prepare_dataset(
            class_names=["apple"],
            output_name="yaml_test",
            val_ratio=0.2,
            group_continuous_frames=False
        )

        assert result.success is True
        yaml_path = result.output_dir / "data.yaml"
        assert yaml_path.exists()

    def test_copy_images(self, mock_path_coordinator, tmp_path):
        """画像コピーテスト"""
        preparer = DatasetPreparer(path_coordinator=mock_path_coordinator)

        raw_captures = tmp_path / "raw_captures"
        annotated = tmp_path / "annotated"

        apple_dir = raw_captures / "apple"
        apple_dir.mkdir()
        apple_labels = annotated / "apple" / "labels"
        apple_labels.mkdir(parents=True)

        for i in range(5):
            (apple_dir / f"apple_{i}.jpg").write_text(f"image_{i}")
            (apple_labels / f"apple_{i}.txt").write_text("0 0.5 0.5 0.2 0.2")

        result = preparer.prepare_dataset(
            class_names=["apple"],
            output_name="copy_test",
            val_ratio=0.2,
            group_continuous_frames=False
        )

        assert result.success is True
        # 画像がコピーされたことを確認
        train_images = list((result.output_dir / "images" / "train").glob("*.jpg"))
        val_images = list((result.output_dir / "images" / "val").glob("*.jpg"))
        assert len(train_images) + len(val_images) == 5

    def test_get_dataset_stats(self, mock_path_coordinator, tmp_path):
        """データセット統計取得テスト"""
        preparer = DatasetPreparer(path_coordinator=mock_path_coordinator)

        raw_captures = tmp_path / "raw_captures"
        annotated = tmp_path / "annotated"

        # 複数クラスを作成
        for cls_name in ["apple", "banana"]:
            cls_dir = raw_captures / cls_name
            cls_dir.mkdir()
            labels_dir = annotated / cls_name / "labels"
            labels_dir.mkdir(parents=True)

            for i in range(10):
                (cls_dir / f"{cls_name}_{i}.jpg").write_text("dummy")
                (labels_dir / f"{cls_name}_{i}.txt").write_text("0 0.5 0.5 0.2 0.2")

        classes = preparer.get_available_classes()
        ready_classes = preparer.get_ready_classes()

        assert len(classes) == 2
        assert len(ready_classes) == 2  # 両方とも10サンプルあり

    def test_prepare_dataset_class_not_found(self, mock_path_coordinator, tmp_path):
        """クラス未発見時のデータセット準備テスト"""
        preparer = DatasetPreparer(path_coordinator=mock_path_coordinator)

        result = preparer.prepare_dataset(
            class_names=["nonexistent"],
            output_name="error_test",
            val_ratio=0.2
        )

        assert result.success is False
        assert "not found" in result.error_message

    def test_prepare_dataset_no_pairs(self, mock_path_coordinator, tmp_path):
        """ペアなし時のデータセット準備テスト"""
        preparer = DatasetPreparer(path_coordinator=mock_path_coordinator)

        raw_captures = tmp_path / "raw_captures"
        apple_dir = raw_captures / "apple"
        apple_dir.mkdir()

        # 画像だけ（ラベルなし）
        for i in range(5):
            (apple_dir / f"apple_{i}.jpg").write_text("dummy")

        result = preparer.prepare_dataset(
            class_names=["apple"],
            output_name="no_pairs_test",
            val_ratio=0.2
        )

        assert result.success is False
        assert "No image-label pairs" in result.error_message
