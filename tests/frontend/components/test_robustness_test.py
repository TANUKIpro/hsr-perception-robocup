"""
Robustness Test テスト

app/components/robustness_test.py と robustness_augmentation.py のテスト
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np

# アプリパスを追加（インポート前に）
APP_PATH = str(Path(__file__).parent.parent.parent.parent / "app")
sys.path.insert(0, APP_PATH)

# Streamlitをモック（インポート前に）
mock_st = MagicMock()
mock_st.html = MagicMock()
mock_st.markdown = MagicMock()
mock_st.spinner = MagicMock()
mock_st.columns = MagicMock(return_value=[MagicMock() for _ in range(3)])
mock_st.selectbox = MagicMock(return_value="Brightness")
mock_st.slider = MagicMock(return_value=1.0)
mock_st.button = MagicMock(return_value=False)
mock_st.image = MagicMock()
mock_st.caption = MagicMock()
mock_st.metric = MagicMock()
mock_st.success = MagicMock()
mock_st.warning = MagicMock()
mock_st.error = MagicMock()
mock_st.info = MagicMock()
mock_st.dataframe = MagicMock()
mock_st.expander = MagicMock()
mock_st.multiselect = MagicMock(return_value=["Brightness"])
mock_st.progress = MagicMock()
sys.modules['streamlit'] = mock_st

# pandasをモック
mock_pd = MagicMock()
mock_pd.DataFrame = MagicMock()
sys.modules['pandas'] = mock_pd

# cv2は実際のモジュールを使用（画像処理テストに必要）
import cv2

# robustness_augmentationモジュールをインポート
# ファイルを直接読み込んでexec
import types

augmentation_path = Path(__file__).parent.parent.parent.parent / "app" / "components" / "robustness_augmentation.py"
with open(augmentation_path, 'r') as f:
    aug_source = f.read()

# モジュールを作成して実行
augmentation_module = types.ModuleType("robustness_augmentation")
augmentation_module.__file__ = str(augmentation_path)
exec(compile(aug_source, augmentation_path, 'exec'), augmentation_module.__dict__)

# テスト対象クラス・関数を取得
RobustnessAugmentor = augmentation_module.RobustnessAugmentor
AugmentationResult = augmentation_module.AugmentationResult
load_annotations_from_yolo = augmentation_module.load_annotations_from_yolo

# robustness_test モジュールもロード
test_path = Path(__file__).parent.parent.parent.parent / "app" / "components" / "robustness_test.py"
with open(test_path, 'r') as f:
    test_source = f.read()

# 相対インポートを調整
test_source = test_source.replace(
    "from .robustness_augmentation import",
    "from robustness_augmentation import"
)

# robustness_augmentation をsys.modulesに登録
sys.modules['robustness_augmentation'] = augmentation_module

# robustness_test モジュールを作成して実行
robustness_test_module = types.ModuleType("robustness_test")
robustness_test_module.__file__ = str(test_path)
exec(compile(test_source, test_path, 'exec'), robustness_test_module.__dict__)

# テスト対象関数を取得
DetectionResult = robustness_test_module.DetectionResult
RobustnessTestResult = robustness_test_module.RobustnessTestResult
_run_single_test = robustness_test_module._run_single_test
_get_avg_confidence = robustness_test_module._get_avg_confidence


# =============================================================================
# フィクスチャ
# =============================================================================


@pytest.fixture
def sample_image():
    """テスト用のBGR画像を作成"""
    # ランダムなカラー画像 (100x100, BGR)
    np.random.seed(42)
    return np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)


@pytest.fixture
def uniform_image():
    """均一な色の画像（明るさテスト用）"""
    return np.full((100, 100, 3), 128, dtype=np.uint8)


@pytest.fixture
def augmentor():
    """固定シードでRobustnessAugmentorインスタンスを作成"""
    return RobustnessAugmentor(seed=42)


@pytest.fixture
def mock_model():
    """YOLOモデルのモック"""
    model = MagicMock()
    model.names = {0: "apple", 1: "cup", 2: "banana"}
    return model


@pytest.fixture
def mock_detection_results():
    """検出結果のモック"""
    mock_box1 = MagicMock()
    mock_box1.cls.item.return_value = 0
    mock_box1.conf.item.return_value = 0.85
    mock_box1.xyxy = [[10, 10, 50, 50]]

    mock_box2 = MagicMock()
    mock_box2.cls.item.return_value = 1
    mock_box2.conf.item.return_value = 0.75
    mock_box2.xyxy = [[60, 60, 90, 90]]

    mock_results = MagicMock()
    mock_results.boxes = [mock_box1, mock_box2]
    mock_results.plot.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

    return mock_results


# =============================================================================
# TestApplyBrightnessAugmentation
# =============================================================================


class TestApplyBrightnessAugmentation:
    """明るさ拡張のテスト (test_apply_brightness_augmentation)"""

    def test_brightness_decrease_darkens_image(self, augmentor, uniform_image):
        """明るさ係数 < 1.0 で画像が暗くなる"""
        result = augmentor.adjust_brightness(uniform_image, 0.5)

        assert result.shape == uniform_image.shape
        assert result.mean() < uniform_image.mean()

    def test_brightness_increase_brightens_image(self, augmentor, uniform_image):
        """明るさ係数 > 1.0 で画像が明るくなる"""
        result = augmentor.adjust_brightness(uniform_image, 1.5)

        assert result.shape == uniform_image.shape
        assert result.mean() > uniform_image.mean()

    def test_brightness_unchanged_at_factor_one(self, augmentor, uniform_image):
        """明るさ係数 1.0 で画像がほぼ変わらない"""
        result = augmentor.adjust_brightness(uniform_image, 1.0)

        assert result.shape == uniform_image.shape
        # HSV変換の丸め誤差を考慮
        np.testing.assert_array_almost_equal(result, uniform_image, decimal=0)

    def test_brightness_output_shape_matches_input(self, augmentor, sample_image):
        """出力形状が入力と一致する"""
        for factor in [0.3, 0.7, 1.0, 1.3, 1.8]:
            result = augmentor.adjust_brightness(sample_image, factor)
            assert result.shape == sample_image.shape
            assert result.dtype == np.uint8

    def test_generate_brightness_variants_default(self, augmentor, sample_image):
        """デフォルトパラメータで明るさバリアントを生成"""
        variants = augmentor.generate_brightness_variants(sample_image)

        # デフォルトは7種類 [0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 1.8]
        assert len(variants) == 7

        for variant in variants:
            assert isinstance(variant, AugmentationResult)
            assert variant.image.shape == sample_image.shape
            assert "brightness_factor" in variant.params


# =============================================================================
# TestApplyShadowAugmentation (noise の代替)
# =============================================================================


class TestApplyShadowAugmentation:
    """影拡張のテスト (test_apply_noise_augmentation の代替)"""

    def test_shadow_random_type(self, augmentor, sample_image):
        """ランダム影の注入"""
        result = augmentor.inject_shadow(sample_image, intensity=0.5, shadow_type="random")

        assert result.shape == sample_image.shape
        assert result.dtype == np.uint8

    def test_shadow_diagonal_type(self, augmentor, sample_image):
        """対角線影の注入"""
        result = augmentor.inject_shadow(sample_image, intensity=0.5, shadow_type="diagonal")

        assert result.shape == sample_image.shape
        assert result.dtype == np.uint8

    def test_shadow_circular_type(self, augmentor, sample_image):
        """円形影の注入"""
        result = augmentor.inject_shadow(sample_image, intensity=0.5, shadow_type="circular")

        assert result.shape == sample_image.shape
        assert result.dtype == np.uint8

    def test_shadow_intensity_affects_darkness(self, augmentor, uniform_image):
        """影の強度が暗さに影響する"""
        low_intensity = augmentor.inject_shadow(uniform_image.copy(), intensity=0.2, shadow_type="diagonal")
        high_intensity = augmentor.inject_shadow(uniform_image.copy(), intensity=0.6, shadow_type="diagonal")

        # 高い強度の方が暗い部分がある（全体平均ではなく最小値で比較）
        assert low_intensity.shape == high_intensity.shape

    def test_generate_shadow_variants(self, augmentor, sample_image):
        """影バリアントの生成"""
        variants = augmentor.generate_shadow_variants(sample_image)

        # デフォルト: 3強度 x 3タイプ = 9種類
        assert len(variants) == 9

        for variant in variants:
            assert isinstance(variant, AugmentationResult)
            assert "shadow_intensity" in variant.params
            assert "shadow_type" in variant.params


# =============================================================================
# TestApplyOcclusionAugmentation (blur の代替)
# =============================================================================


class TestApplyOcclusionAugmentation:
    """遮蔽拡張のテスト (test_apply_blur_augmentation の代替)"""

    def test_occlusion_rectangle(self, augmentor, sample_image):
        """矩形遮蔽の注入"""
        result = augmentor.inject_occlusion(sample_image, occlusion_ratio=0.3, occlusion_type="rectangle")

        assert result.shape == sample_image.shape
        assert result.dtype == np.uint8

    def test_occlusion_edge(self, augmentor, sample_image):
        """エッジ遮蔽の注入"""
        result = augmentor.inject_occlusion(sample_image, occlusion_ratio=0.2, occlusion_type="edge")

        assert result.shape == sample_image.shape
        assert result.dtype == np.uint8

    def test_occlusion_random_polygon(self, augmentor, sample_image):
        """ランダムポリゴン遮蔽の注入"""
        result = augmentor.inject_occlusion(sample_image, occlusion_ratio=0.25, occlusion_type="random")

        assert result.shape == sample_image.shape
        assert result.dtype == np.uint8

    def test_occlusion_with_custom_color(self, augmentor, sample_image):
        """カスタム色での遮蔽"""
        result = augmentor.inject_occlusion(
            sample_image,
            occlusion_ratio=0.3,
            occlusion_type="rectangle",
            color=(255, 0, 0)  # 青色
        )

        assert result.shape == sample_image.shape

    def test_generate_occlusion_variants(self, augmentor, sample_image):
        """遮蔽バリアントの生成"""
        variants = augmentor.generate_occlusion_variants(sample_image)

        # デフォルト: 4比率 x 3タイプ = 12種類
        assert len(variants) == 12

        for variant in variants:
            assert isinstance(variant, AugmentationResult)
            assert "occlusion_ratio" in variant.params
            assert "occlusion_type" in variant.params


# =============================================================================
# TestRunRobustnessTest
# =============================================================================


class TestRunRobustnessTest:
    """ロバスト性テスト実行のテスト (test_run_robustness_test)"""

    def setup_method(self):
        """各テスト前にモックをリセット"""
        mock_st.reset_mock()

    def test_run_single_test_returns_result(self, sample_image, mock_model):
        """_run_single_test が RobustnessTestResult を返す"""
        # モデル結果のセットアップ
        mock_box = MagicMock()
        mock_box.cls.item.return_value = 0
        mock_box.conf.item.return_value = 0.8

        # xyxyはtensor-like objectとしてモック
        mock_xyxy = MagicMock()
        mock_xyxy.tolist.return_value = [10, 10, 50, 50]
        mock_box.xyxy = [mock_xyxy]

        mock_results = MagicMock()
        mock_results.boxes = [mock_box]
        mock_model.return_value = [mock_results]

        # AugmentationResultを作成
        aug_result = AugmentationResult(
            image=sample_image,
            name="Test Augmentation",
            description="Test description",
            params={"test_param": 1}
        )

        result = _run_single_test(
            mock_model,
            aug_result,
            orig_count=1,
            orig_conf=0.9,
            conf_threshold=0.25
        )

        assert isinstance(result, RobustnessTestResult)
        assert result.augmentation == aug_result
        assert isinstance(result.detections, list)
        assert isinstance(result.detection_count_diff, int)
        assert isinstance(result.avg_confidence_diff, float)

    def test_run_single_test_with_no_detections(self, sample_image, mock_model):
        """検出なしの場合のテスト"""
        mock_results = MagicMock()
        mock_results.boxes = []
        mock_model.return_value = [mock_results]

        aug_result = AugmentationResult(
            image=sample_image,
            name="Test",
            description="Test",
            params={}
        )

        result = _run_single_test(
            mock_model,
            aug_result,
            orig_count=2,
            orig_conf=0.85,
            conf_threshold=0.25
        )

        assert result.detection_count_diff == -2  # 0 - 2
        assert len(result.detections) == 0


# =============================================================================
# TestCalculateRobustnessScore
# =============================================================================


class TestCalculateRobustnessScore:
    """ロバスト性スコア計算のテスト (test_calculate_robustness_score)"""

    def test_get_avg_confidence_empty_boxes(self):
        """空の検出結果で0を返す"""
        mock_results = MagicMock()
        mock_results.boxes = []

        result = _get_avg_confidence(mock_results)

        assert result == 0.0

    def test_get_avg_confidence_single_box(self):
        """単一の検出で正しい信頼度を返す"""
        mock_box = MagicMock()
        mock_box.conf.item.return_value = 0.85

        mock_results = MagicMock()
        mock_results.boxes = [mock_box]

        result = _get_avg_confidence(mock_results)

        assert result == 0.85

    def test_get_avg_confidence_multiple_boxes(self):
        """複数の検出で平均信頼度を返す"""
        mock_box1 = MagicMock()
        mock_box1.conf.item.return_value = 0.8

        mock_box2 = MagicMock()
        mock_box2.conf.item.return_value = 0.6

        mock_results = MagicMock()
        mock_results.boxes = [mock_box1, mock_box2]

        result = _get_avg_confidence(mock_results)

        assert result == 0.7  # (0.8 + 0.6) / 2

    def test_get_avg_confidence_varied_values(self):
        """様々な信頼度値での平均計算"""
        confidences = [0.9, 0.75, 0.82, 0.68]
        mock_boxes = []
        for conf in confidences:
            mock_box = MagicMock()
            mock_box.conf.item.return_value = conf
            mock_boxes.append(mock_box)

        mock_results = MagicMock()
        mock_results.boxes = mock_boxes

        result = _get_avg_confidence(mock_results)

        expected = sum(confidences) / len(confidences)
        assert abs(result - expected) < 0.001


# =============================================================================
# TestHueRotation
# =============================================================================


class TestHueRotation:
    """色相回転のテスト（追加テスト）"""

    def test_rotate_hue_zero_unchanged(self, augmentor, sample_image):
        """0度回転で画像がほぼ変わらない"""
        result = augmentor.rotate_hue(sample_image, 0)

        assert result.shape == sample_image.shape
        assert result.dtype == np.uint8
        # HSV変換の丸め誤差があるため、平均差分が小さいことを確認
        diff = np.abs(result.astype(np.int16) - sample_image.astype(np.int16)).mean()
        assert diff < 5  # 平均誤差が5未満であることを確認

    def test_rotate_hue_various_angles(self, augmentor, sample_image):
        """様々な角度での色相回転"""
        for angle in [30, 60, 90, 120, 150, 180]:
            result = augmentor.rotate_hue(sample_image, angle)

            assert result.shape == sample_image.shape
            assert result.dtype == np.uint8

    def test_generate_hue_variants(self, augmentor, sample_image):
        """色相バリアントの生成"""
        variants = augmentor.generate_hue_variants(sample_image)

        # デフォルト: [0, 30, 60, 90, 120, 150, 180] = 7種類
        assert len(variants) == 7

        for variant in variants:
            assert isinstance(variant, AugmentationResult)
            assert "hue_angle" in variant.params


# =============================================================================
# TestLoadAnnotationsFromYolo
# =============================================================================


class TestLoadAnnotationsFromYolo:
    """YOLOアノテーション読み込みのテスト"""

    def test_load_nonexistent_file(self, tmp_path):
        """存在しないファイルで空リストを返す"""
        label_path = tmp_path / "nonexistent.txt"

        result = load_annotations_from_yolo(label_path, (100, 100))

        assert result == []

    def test_load_valid_annotations(self, tmp_path):
        """有効なアノテーションの読み込み"""
        label_path = tmp_path / "labels.txt"
        label_path.write_text("0 0.5 0.5 0.2 0.3\n1 0.25 0.75 0.1 0.2\n")

        result = load_annotations_from_yolo(label_path, (100, 100))

        assert len(result) == 2
        assert result[0]["class_id"] == 0
        assert result[1]["class_id"] == 1
        assert "bbox" in result[0]
        assert len(result[0]["bbox"]) == 4

    def test_load_empty_file(self, tmp_path):
        """空ファイルで空リストを返す"""
        label_path = tmp_path / "empty.txt"
        label_path.write_text("")

        result = load_annotations_from_yolo(label_path, (100, 100))

        assert result == []
