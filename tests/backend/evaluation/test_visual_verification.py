"""
Visual Verification モジュールのテスト

scripts/evaluation/visual_verification.py のテストを行います。
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


class TestVisualVerifierInit:
    """VisualVerifier 初期化のテスト"""

    def test_init_with_model_path(self, tmp_path):
        """モデルパスでの初期化テスト"""
        with patch('ultralytics.YOLO') as mock_yolo_cls:
            mock_model = MagicMock()
            mock_model.names = {0: "apple", 1: "cup"}
            mock_yolo_cls.return_value = mock_model

            from evaluation.visual_verification import VisualVerifier
            verifier = VisualVerifier("/path/to/model.pt")

            mock_yolo_cls.assert_called_once_with("/path/to/model.pt")
            assert verifier.model == mock_model
            assert verifier.class_names == {0: "apple", 1: "cup"}

    def test_init_with_class_config(self, tmp_path):
        """クラス設定ファイル付きの初期化テスト"""
        import json

        # クラス設定ファイルを作成
        config_path = tmp_path / "class_config.json"
        config_data = {"categories": ["food"], "objects": [{"id": 0, "name": "apple"}]}
        with open(config_path, "w") as f:
            json.dump(config_data, f)

        with patch('ultralytics.YOLO') as mock_yolo_cls:
            mock_model = MagicMock()
            mock_model.names = {0: "apple"}
            mock_yolo_cls.return_value = mock_model

            from evaluation.visual_verification import VisualVerifier
            verifier = VisualVerifier("/path/to/model.pt", str(config_path))

            assert verifier.class_config is not None
            assert verifier.class_config["categories"] == ["food"]

    def test_init_without_class_config(self):
        """クラス設定ファイルなしの初期化テスト"""
        with patch('ultralytics.YOLO') as mock_yolo_cls:
            mock_model = MagicMock()
            mock_model.names = {0: "apple"}
            mock_yolo_cls.return_value = mock_model

            from evaluation.visual_verification import VisualVerifier
            verifier = VisualVerifier("/path/to/model.pt")

            assert verifier.class_config is None


class TestGenerateColors:
    """色生成のテスト"""

    def test_generate_colors_for_each_class(self):
        """各クラスに対して色が生成されることを確認"""
        with patch('ultralytics.YOLO') as mock_yolo_cls:
            mock_model = MagicMock()
            mock_model.names = {0: "apple", 1: "cup", 2: "bottle"}
            mock_yolo_cls.return_value = mock_model

            from evaluation.visual_verification import VisualVerifier
            verifier = VisualVerifier("/path/to/model.pt")

            assert len(verifier.class_colors) == 3
            assert 0 in verifier.class_colors
            assert 1 in verifier.class_colors
            assert 2 in verifier.class_colors

    def test_color_format_is_tuple(self):
        """色がタプル形式であることを確認"""
        with patch('ultralytics.YOLO') as mock_yolo_cls:
            mock_model = MagicMock()
            mock_model.names = {0: "apple"}
            mock_yolo_cls.return_value = mock_model

            from evaluation.visual_verification import VisualVerifier
            verifier = VisualVerifier("/path/to/model.pt")

            color = verifier.class_colors[0]
            assert isinstance(color, tuple)
            assert len(color) == 3  # RGB


class TestDrawDetections:
    """検出描画のテスト"""

    def test_draw_detections_returns_image(self):
        """検出描画が画像を返すことを確認"""
        with patch('ultralytics.YOLO') as mock_yolo_cls:
            mock_model = MagicMock()
            mock_model.names = {0: "apple"}
            mock_yolo_cls.return_value = mock_model

            from evaluation.visual_verification import VisualVerifier
            verifier = VisualVerifier("/path/to/model.pt")

            # テスト用の画像と検出
            image = np.zeros((480, 640, 3), dtype=np.uint8)
            detections = [
                {"bbox": [100, 100, 200, 200], "class_id": 0, "confidence": 0.95}
            ]

            result = verifier._draw_detections(image, detections)

            assert result.shape == (480, 640, 3)
            assert result.dtype == np.uint8

    def test_draw_detections_preserves_original(self):
        """元の画像が変更されないことを確認"""
        with patch('ultralytics.YOLO') as mock_yolo_cls:
            mock_model = MagicMock()
            mock_model.names = {0: "apple"}
            mock_yolo_cls.return_value = mock_model

            from evaluation.visual_verification import VisualVerifier
            verifier = VisualVerifier("/path/to/model.pt")

            image = np.zeros((480, 640, 3), dtype=np.uint8)
            original = image.copy()
            detections = [
                {"bbox": [100, 100, 200, 200], "class_id": 0, "confidence": 0.95}
            ]

            verifier._draw_detections(image, detections)

            # 元の画像が変更されていないことを確認
            np.testing.assert_array_equal(image, original)

    def test_draw_detections_with_color_override(self):
        """色オーバーライドのテスト"""
        with patch('ultralytics.YOLO') as mock_yolo_cls:
            mock_model = MagicMock()
            mock_model.names = {0: "apple"}
            mock_yolo_cls.return_value = mock_model

            from evaluation.visual_verification import VisualVerifier
            verifier = VisualVerifier("/path/to/model.pt")

            image = np.zeros((480, 640, 3), dtype=np.uint8)
            detections = [
                {"bbox": [100, 100, 200, 200], "class_id": 0, "confidence": 0.95}
            ]

            # カスタム色でテスト (赤)
            result = verifier._draw_detections(
                image, detections, color_override=(0, 0, 255)
            )

            # 画像にピクセルが描画されていることを確認
            assert np.any(result != 0)

    def test_draw_empty_detections(self):
        """空の検出リストでのテスト"""
        with patch('ultralytics.YOLO') as mock_yolo_cls:
            mock_model = MagicMock()
            mock_model.names = {0: "apple"}
            mock_yolo_cls.return_value = mock_model

            from evaluation.visual_verification import VisualVerifier
            verifier = VisualVerifier("/path/to/model.pt")

            image = np.zeros((480, 640, 3), dtype=np.uint8)
            result = verifier._draw_detections(image, [])

            # 空の検出では元の画像と同じ
            np.testing.assert_array_equal(result, image)


class TestPredictImage:
    """画像予測のテスト"""

    def test_predict_returns_tuple(self, tmp_path):
        """予測結果がタプルで返されることを確認"""
        # テスト画像を作成
        image_path = tmp_path / "test.jpg"
        import cv2
        cv2.imwrite(str(image_path), np.zeros((100, 100, 3), dtype=np.uint8))

        with patch('ultralytics.YOLO') as mock_yolo_cls:
            mock_model = MagicMock()
            mock_model.names = {0: "apple"}

            # モック結果を設定
            mock_box = MagicMock()
            mock_box.cls = MagicMock()
            mock_box.cls.item.return_value = 0
            mock_box.conf = MagicMock()
            mock_box.conf.item.return_value = 0.95
            mock_box.xyxy = MagicMock()
            mock_box.xyxy.__getitem__ = lambda self, x: MagicMock(
                tolist=lambda: [10, 10, 50, 50]
            )

            mock_result = MagicMock()
            mock_result.boxes = [mock_box]
            mock_result.plot.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

            mock_model.return_value = [mock_result]
            mock_yolo_cls.return_value = mock_model

            from evaluation.visual_verification import VisualVerifier
            verifier = VisualVerifier("/path/to/model.pt")

            annotated, detections = verifier.predict_image(str(image_path))

            assert isinstance(annotated, np.ndarray)
            assert isinstance(detections, list)

    def test_predict_detection_format(self, tmp_path):
        """検出結果のフォーマットを確認"""
        image_path = tmp_path / "test.jpg"
        import cv2
        cv2.imwrite(str(image_path), np.zeros((100, 100, 3), dtype=np.uint8))

        with patch('ultralytics.YOLO') as mock_yolo_cls:
            mock_model = MagicMock()
            mock_model.names = {0: "apple"}

            mock_box = MagicMock()
            mock_box.cls = MagicMock()
            mock_box.cls.item.return_value = 0
            mock_box.conf = MagicMock()
            mock_box.conf.item.return_value = 0.95
            mock_box.xyxy = MagicMock()
            mock_box.xyxy.__getitem__ = lambda self, x: MagicMock(
                tolist=lambda: [10.0, 10.0, 50.0, 50.0]
            )

            mock_result = MagicMock()
            mock_result.boxes = [mock_box]
            mock_result.plot.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

            mock_model.return_value = [mock_result]
            mock_yolo_cls.return_value = mock_model

            from evaluation.visual_verification import VisualVerifier
            verifier = VisualVerifier("/path/to/model.pt")

            _, detections = verifier.predict_image(str(image_path))

            assert len(detections) == 1
            det = detections[0]
            assert "class_id" in det
            assert "class_name" in det
            assert "confidence" in det
            assert "bbox" in det
            assert det["class_id"] == 0
            assert det["class_name"] == "apple"
            assert det["confidence"] == 0.95


class TestVerifyBatch:
    """バッチ検証のテスト"""

    def test_verify_batch_empty_dir(self, tmp_path, capsys):
        """空のディレクトリでのバッチ検証テスト"""
        with patch('ultralytics.YOLO') as mock_yolo_cls:
            mock_model = MagicMock()
            mock_model.names = {0: "apple"}
            mock_yolo_cls.return_value = mock_model

            from evaluation.visual_verification import VisualVerifier
            verifier = VisualVerifier("/path/to/model.pt")

            result = verifier.verify_batch(str(tmp_path))

            assert result == {}
            captured = capsys.readouterr()
            assert "No images found" in captured.out


class TestCreateComparisonGrid:
    """比較グリッド作成のテスト"""

    def test_create_grid_returns_array(self, tmp_path):
        """グリッド作成がnumpy配列を返すことを確認"""
        # テスト画像を作成
        images = []
        for i in range(4):
            img_path = tmp_path / f"test_{i}.jpg"
            import cv2
            cv2.imwrite(str(img_path), np.zeros((100, 100, 3), dtype=np.uint8))
            images.append(str(img_path))

        with patch('ultralytics.YOLO') as mock_yolo_cls:
            mock_model = MagicMock()
            mock_model.names = {0: "apple"}

            mock_result = MagicMock()
            mock_result.boxes = []
            mock_result.plot.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

            mock_model.return_value = [mock_result]
            mock_yolo_cls.return_value = mock_model

            from evaluation.visual_verification import VisualVerifier
            verifier = VisualVerifier("/path/to/model.pt")

            grid = verifier.create_comparison_grid(images, grid_cols=2)

            assert isinstance(grid, np.ndarray)
            assert grid.dtype == np.uint8

    def test_create_grid_with_output(self, tmp_path):
        """グリッドをファイルに保存するテスト"""
        # テスト画像を作成
        images = []
        for i in range(2):
            img_path = tmp_path / f"test_{i}.jpg"
            import cv2
            cv2.imwrite(str(img_path), np.zeros((100, 100, 3), dtype=np.uint8))
            images.append(str(img_path))

        output_path = tmp_path / "grid.jpg"

        with patch('ultralytics.YOLO') as mock_yolo_cls:
            mock_model = MagicMock()
            mock_model.names = {0: "apple"}

            mock_result = MagicMock()
            mock_result.boxes = []
            mock_result.plot.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

            mock_model.return_value = [mock_result]
            mock_yolo_cls.return_value = mock_model

            from evaluation.visual_verification import VisualVerifier
            verifier = VisualVerifier("/path/to/model.pt")

            verifier.create_comparison_grid(images, output_path=str(output_path))

            assert output_path.exists()


class TestGenerateReportSamples:
    """レポートサンプル生成のテスト"""

    def test_generate_samples_creates_files(self, tmp_path):
        """サンプル画像ファイルが生成されることを確認"""
        # テスト用ディレクトリを作成
        test_dir = tmp_path / "test_images"
        test_dir.mkdir()
        output_dir = tmp_path / "samples"

        # テスト画像を作成
        for i in range(3):
            img_path = test_dir / f"test_{i}.jpg"
            import cv2
            cv2.imwrite(str(img_path), np.zeros((100, 100, 3), dtype=np.uint8))

        with patch('ultralytics.YOLO') as mock_yolo_cls:
            mock_model = MagicMock()
            mock_model.names = {0: "apple"}

            mock_box = MagicMock()
            mock_box.cls = MagicMock()
            mock_box.cls.item.return_value = 0
            mock_box.conf = MagicMock()
            mock_box.conf.item.return_value = 0.95
            mock_box.xyxy = MagicMock()
            mock_box.xyxy.__getitem__ = lambda self, x: MagicMock(
                tolist=lambda: [10.0, 10.0, 50.0, 50.0]
            )

            mock_result = MagicMock()
            mock_result.boxes = [mock_box]
            mock_result.plot.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

            mock_model.return_value = [mock_result]
            mock_yolo_cls.return_value = mock_model

            from evaluation.visual_verification import VisualVerifier
            verifier = VisualVerifier("/path/to/model.pt")

            generated = verifier.generate_report_samples(
                str(test_dir),
                str(output_dir),
                samples_per_class=2,
            )

            assert len(generated) > 0
            assert output_dir.exists()

    def test_generate_samples_empty_dir(self, tmp_path):
        """空のディレクトリでのテスト"""
        test_dir = tmp_path / "test_images"
        test_dir.mkdir()
        output_dir = tmp_path / "samples"

        with patch('ultralytics.YOLO') as mock_yolo_cls:
            mock_model = MagicMock()
            mock_model.names = {0: "apple"}

            mock_result = MagicMock()
            mock_result.boxes = []
            mock_result.plot.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

            mock_model.return_value = [mock_result]
            mock_yolo_cls.return_value = mock_model

            from evaluation.visual_verification import VisualVerifier
            verifier = VisualVerifier("/path/to/model.pt")

            generated = verifier.generate_report_samples(
                str(test_dir),
                str(output_dir),
            )

            # 空のリストを返す
            assert generated == []
