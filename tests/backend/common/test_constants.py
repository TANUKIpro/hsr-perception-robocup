"""
Constants モジュールのテスト

scripts/common/constants.py の定数値をテストします。
"""

import sys
from pathlib import Path

# scriptsディレクトリをパスに追加
scripts_dir = Path(__file__).parent.parent.parent.parent / "scripts"
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

from common.constants import (
    # Image extensions
    IMAGE_EXTENSIONS,
    IMAGE_EXTENSIONS_ALL,
    # Competition defaults
    DEFAULT_TARGET_SAMPLES,
    DEFAULT_BURST_INTERVAL,
    MIN_SAMPLES_FOR_TRAINING,
    DEFAULT_TRAIN_RATIO,
    DEFAULT_GROUP_INTERVAL_SEC,
    # Model/Inference defaults
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_IOU_THRESHOLD,
    TARGET_MAP50,
    TARGET_INFERENCE_MS,
    # Annotation defaults
    DEFAULT_BBOX_MARGIN_RATIO,
    DEFAULT_MIN_CONTOUR_AREA,
    DEFAULT_MAX_CONTOUR_AREA_RATIO,
    # Training defaults
    DEFAULT_YOLO_MODEL,
    DEFAULT_EPOCHS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_PATIENCE,
    # GPU scaling constants
    GPU_VRAM_THRESHOLDS,
    MODEL_VRAM_OVERHEAD,
    MODEL_PER_SAMPLE_MEMORY_MB,
    STANDARD_IMAGE_SIZES,
    DEFAULT_VRAM_SAFETY_MARGIN,
    DEFAULT_MAX_VRAM_UTILIZATION,
    OOM_BATCH_REDUCTION_FACTOR,
    MAX_OOM_RETRIES,
    # TensorBoard settings
    TENSORBOARD_DEFAULT_PORT,
    TENSORBOARD_FLUSH_SECS,
    TENSORBOARD_LOG_FREQUENCY,
    # Copy-paste augmentation
    DEFAULT_ALPHA_BLUR_SIGMA,
    DEFAULT_SCALE_RANGE,
    DEFAULT_ROTATION_RANGE,
    DEFAULT_SYNTHETIC_RATIO,
    DEFAULT_MAX_OBJECTS_PER_IMAGE,
    DEFAULT_OVERLAP_IOU_THRESHOLD,
)


class TestImageExtensions:
    """画像ファイル拡張子の定数テスト"""

    def test_supported_extensions(self):
        """サポートする拡張子が含まれていることを確認"""
        assert ".jpg" in IMAGE_EXTENSIONS
        assert ".jpeg" in IMAGE_EXTENSIONS
        assert ".png" in IMAGE_EXTENSIONS
        assert ".bmp" in IMAGE_EXTENSIONS

    def test_extensions_are_lowercase(self):
        """拡張子が小文字であることを確認"""
        for ext in IMAGE_EXTENSIONS:
            assert ext == ext.lower()
            assert ext.startswith(".")

    def test_all_extensions_includes_uppercase(self):
        """IMAGE_EXTENSIONS_ALL に大文字が含まれることを確認"""
        assert ".JPG" in IMAGE_EXTENSIONS_ALL
        assert ".PNG" in IMAGE_EXTENSIONS_ALL
        assert len(IMAGE_EXTENSIONS_ALL) == len(IMAGE_EXTENSIONS) * 2


class TestCompetitionDefaults:
    """競技会デフォルト値のテスト"""

    def test_default_sample_count(self):
        """デフォルトサンプル数の確認"""
        assert DEFAULT_TARGET_SAMPLES == 100
        assert isinstance(DEFAULT_TARGET_SAMPLES, int)

    def test_grouping_interval(self):
        """グルーピング間隔の確認"""
        assert DEFAULT_GROUP_INTERVAL_SEC == 2.0
        assert isinstance(DEFAULT_GROUP_INTERVAL_SEC, float)

    def test_burst_interval(self):
        """バースト間隔の確認"""
        assert DEFAULT_BURST_INTERVAL == 0.2
        assert isinstance(DEFAULT_BURST_INTERVAL, float)

    def test_min_samples_for_training(self):
        """訓練用最小サンプル数の確認"""
        assert MIN_SAMPLES_FOR_TRAINING == 50
        assert MIN_SAMPLES_FOR_TRAINING < DEFAULT_TARGET_SAMPLES

    def test_train_ratio(self):
        """訓練比率の確認"""
        assert DEFAULT_TRAIN_RATIO == 0.80
        assert 0.0 < DEFAULT_TRAIN_RATIO < 1.0


class TestModelDefaults:
    """モデル/推論デフォルト値のテスト"""

    def test_min_map_threshold(self):
        """最小mAP閾値の確認"""
        assert TARGET_MAP50 == 0.85
        assert isinstance(TARGET_MAP50, float)
        assert 0.0 <= TARGET_MAP50 <= 1.0

    def test_max_inference_time(self):
        """最大推論時間の確認"""
        assert TARGET_INFERENCE_MS == 100.0
        assert isinstance(TARGET_INFERENCE_MS, float)
        assert TARGET_INFERENCE_MS > 0

    def test_confidence_threshold(self):
        """信頼度閾値の確認"""
        assert DEFAULT_CONFIDENCE_THRESHOLD == 0.25
        assert 0.0 <= DEFAULT_CONFIDENCE_THRESHOLD <= 1.0

    def test_iou_threshold(self):
        """IoU閾値の確認"""
        assert DEFAULT_IOU_THRESHOLD == 0.5
        assert 0.0 <= DEFAULT_IOU_THRESHOLD <= 1.0


class TestTrainingDefaults:
    """訓練デフォルト値のテスト"""

    def test_default_model(self):
        """デフォルトモデルの確認"""
        assert DEFAULT_YOLO_MODEL == "yolov8m.pt"
        assert DEFAULT_YOLO_MODEL.endswith(".pt")

    def test_batch_size_range(self):
        """バッチサイズ範囲の確認"""
        assert DEFAULT_BATCH_SIZE == 16
        assert isinstance(DEFAULT_BATCH_SIZE, int)
        assert DEFAULT_BATCH_SIZE > 0

    def test_epochs(self):
        """エポック数の確認"""
        assert DEFAULT_EPOCHS == 50
        assert isinstance(DEFAULT_EPOCHS, int)
        assert DEFAULT_EPOCHS > 0

    def test_image_size(self):
        """画像サイズの確認"""
        assert DEFAULT_IMAGE_SIZE == 640
        assert DEFAULT_IMAGE_SIZE in STANDARD_IMAGE_SIZES

    def test_patience(self):
        """早期終了の patience の確認"""
        assert DEFAULT_PATIENCE == 10
        assert DEFAULT_PATIENCE > 0


class TestGPUScalingConstants:
    """GPUスケーリング定数のテスト"""

    def test_vram_thresholds(self):
        """VRAM閾値の確認"""
        assert isinstance(GPU_VRAM_THRESHOLDS, dict)
        assert "low" in GPU_VRAM_THRESHOLDS
        assert "medium" in GPU_VRAM_THRESHOLDS
        assert "high" in GPU_VRAM_THRESHOLDS
        assert "workstation" in GPU_VRAM_THRESHOLDS

    def test_vram_thresholds_ranges(self):
        """VRAM閾値の範囲が正しいことを確認"""
        # low: 0-6GB
        assert GPU_VRAM_THRESHOLDS["low"] == (0, 6)
        # medium: 6-12GB
        assert GPU_VRAM_THRESHOLDS["medium"] == (6, 12)
        # high: 12-24GB
        assert GPU_VRAM_THRESHOLDS["high"] == (12, 24)
        # workstation: 24GB+
        assert GPU_VRAM_THRESHOLDS["workstation"][0] == 24

    def test_model_vram_overhead(self):
        """モデルVRAMオーバーヘッドの確認"""
        assert isinstance(MODEL_VRAM_OVERHEAD, dict)
        assert "yolov8n.pt" in MODEL_VRAM_OVERHEAD
        assert "yolov8m.pt" in MODEL_VRAM_OVERHEAD
        assert "yolov8x.pt" in MODEL_VRAM_OVERHEAD
        # オーバーヘッドは正の値
        for model, overhead in MODEL_VRAM_OVERHEAD.items():
            assert overhead > 0, f"{model} should have positive overhead"

    def test_per_sample_memory(self):
        """サンプルあたりメモリの確認"""
        assert isinstance(MODEL_PER_SAMPLE_MEMORY_MB, dict)
        # モデルサイズが大きいほどメモリが増加
        assert MODEL_PER_SAMPLE_MEMORY_MB["yolov8n.pt"] < MODEL_PER_SAMPLE_MEMORY_MB["yolov8m.pt"]
        assert MODEL_PER_SAMPLE_MEMORY_MB["yolov8m.pt"] < MODEL_PER_SAMPLE_MEMORY_MB["yolov8x.pt"]

    def test_standard_image_sizes(self):
        """標準画像サイズの確認"""
        assert isinstance(STANDARD_IMAGE_SIZES, list)
        assert 640 in STANDARD_IMAGE_SIZES
        # サイズは昇順
        assert STANDARD_IMAGE_SIZES == sorted(STANDARD_IMAGE_SIZES)

    def test_safety_margins(self):
        """安全マージンの確認"""
        assert DEFAULT_VRAM_SAFETY_MARGIN == 0.15
        assert DEFAULT_MAX_VRAM_UTILIZATION == 0.85
        # マージンと使用率の合計が100%以下
        assert DEFAULT_VRAM_SAFETY_MARGIN + DEFAULT_MAX_VRAM_UTILIZATION <= 1.0

    def test_oom_recovery_settings(self):
        """OOMリカバリー設定の確認"""
        assert OOM_BATCH_REDUCTION_FACTOR == 0.5
        assert MAX_OOM_RETRIES == 3
        assert 0.0 < OOM_BATCH_REDUCTION_FACTOR < 1.0
        assert MAX_OOM_RETRIES > 0


class TestAnnotationDefaults:
    """アノテーションデフォルト値のテスト"""

    def test_bbox_margin_ratio(self):
        """BBox マージン比率の確認"""
        assert DEFAULT_BBOX_MARGIN_RATIO == 0.02
        assert 0.0 <= DEFAULT_BBOX_MARGIN_RATIO <= 0.1

    def test_contour_area_thresholds(self):
        """輪郭面積閾値の確認"""
        assert DEFAULT_MIN_CONTOUR_AREA == 500
        assert DEFAULT_MAX_CONTOUR_AREA_RATIO == 0.9
        assert DEFAULT_MIN_CONTOUR_AREA > 0
        assert 0.0 < DEFAULT_MAX_CONTOUR_AREA_RATIO <= 1.0


class TestTensorBoardSettings:
    """TensorBoard設定のテスト"""

    def test_default_port(self):
        """デフォルトポートの確認"""
        assert TENSORBOARD_DEFAULT_PORT == 6006
        assert 1024 <= TENSORBOARD_DEFAULT_PORT <= 65535

    def test_flush_settings(self):
        """フラッシュ設定の確認"""
        assert TENSORBOARD_FLUSH_SECS == 30
        assert TENSORBOARD_FLUSH_SECS > 0

    def test_log_frequency(self):
        """ログ頻度の確認"""
        assert TENSORBOARD_LOG_FREQUENCY == 1
        assert TENSORBOARD_LOG_FREQUENCY > 0


class TestCopyPasteAugmentationDefaults:
    """Copy-Paste拡張デフォルト値のテスト"""

    def test_alpha_blur_sigma(self):
        """アルファブラーシグマの確認"""
        assert DEFAULT_ALPHA_BLUR_SIGMA == 2.0
        assert DEFAULT_ALPHA_BLUR_SIGMA > 0

    def test_scale_range(self):
        """スケール範囲の確認"""
        assert DEFAULT_SCALE_RANGE == (0.5, 1.5)
        assert DEFAULT_SCALE_RANGE[0] < DEFAULT_SCALE_RANGE[1]
        assert DEFAULT_SCALE_RANGE[0] > 0

    def test_rotation_range(self):
        """回転範囲の確認"""
        assert DEFAULT_ROTATION_RANGE == (-15.0, 15.0)
        assert DEFAULT_ROTATION_RANGE[0] < DEFAULT_ROTATION_RANGE[1]
        # 対称的な範囲
        assert DEFAULT_ROTATION_RANGE[0] == -DEFAULT_ROTATION_RANGE[1]

    def test_synthetic_ratio(self):
        """合成比率の確認"""
        assert DEFAULT_SYNTHETIC_RATIO == 2.0
        assert DEFAULT_SYNTHETIC_RATIO > 0

    def test_objects_per_image(self):
        """画像あたりオブジェクト数の確認"""
        assert DEFAULT_MAX_OBJECTS_PER_IMAGE == 3
        assert DEFAULT_MAX_OBJECTS_PER_IMAGE > 0

    def test_overlap_threshold(self):
        """オーバーラップ閾値の確認"""
        assert DEFAULT_OVERLAP_IOU_THRESHOLD == 0.1
        assert 0.0 <= DEFAULT_OVERLAP_IOU_THRESHOLD <= 1.0
