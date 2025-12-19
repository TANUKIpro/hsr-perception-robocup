"""
バックエンドテスト用フィクスチャ

torch, ultralytics などの重いライブラリのモックを提供します。
GPU関連のテストを実際のGPUなしで実行可能にします。
"""

import pytest
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path


# scriptsディレクトリをパスに追加
@pytest.fixture(autouse=True)
def add_scripts_to_path():
    """scriptsディレクトリをsys.pathに追加"""
    scripts_path = Path(__file__).parent.parent.parent / "scripts"
    if str(scripts_path) not in sys.path:
        sys.path.insert(0, str(scripts_path))
    yield
    # クリーンアップは不要（他のテストでも使用するため）


@pytest.fixture
def mock_torch():
    """torchモジュールのモック

    GPU関連の機能をモックし、CUDAなしでテストを実行可能にします。
    """
    torch_mock = MagicMock()
    torch_mock.cuda.is_available.return_value = False
    torch_mock.cuda.device_count.return_value = 0
    torch_mock.cuda.get_device_name.return_value = "Mock GPU"
    torch_mock.cuda.memory_allocated.return_value = 0
    torch_mock.cuda.memory_reserved.return_value = 0
    torch_mock.cuda.max_memory_allocated.return_value = 0
    torch_mock.cuda.empty_cache.return_value = None
    torch_mock.cuda.synchronize.return_value = None

    # テンソル操作のモック
    torch_mock.tensor.return_value = MagicMock()
    torch_mock.zeros.return_value = MagicMock()
    torch_mock.ones.return_value = MagicMock()

    with patch.dict("sys.modules", {"torch": torch_mock}):
        yield torch_mock


@pytest.fixture
def mock_torch_with_gpu():
    """GPU利用可能なtorchモジュールのモック"""
    torch_mock = MagicMock()
    torch_mock.cuda.is_available.return_value = True
    torch_mock.cuda.device_count.return_value = 1
    torch_mock.cuda.get_device_name.return_value = "NVIDIA GeForce RTX 3080"
    torch_mock.cuda.memory_allocated.return_value = 1024 * 1024 * 1024  # 1GB
    torch_mock.cuda.memory_reserved.return_value = 2 * 1024 * 1024 * 1024  # 2GB
    torch_mock.cuda.max_memory_allocated.return_value = 8 * 1024 * 1024 * 1024  # 8GB
    torch_mock.cuda.get_device_properties.return_value = MagicMock(
        total_memory=12 * 1024 * 1024 * 1024  # 12GB
    )
    torch_mock.cuda.empty_cache.return_value = None
    torch_mock.cuda.synchronize.return_value = None

    with patch.dict("sys.modules", {"torch": torch_mock}):
        yield torch_mock


@pytest.fixture
def mock_ultralytics():
    """ultralyticsモジュールのモック"""
    ultralytics_mock = MagicMock()

    # YOLO モデルのモック
    model_mock = MagicMock()
    model_mock.train.return_value = MagicMock(
        results_dict={"metrics/mAP50-95(B)": 0.85, "metrics/precision(B)": 0.9}
    )
    model_mock.val.return_value = MagicMock(
        results_dict={"metrics/mAP50-95(B)": 0.85}
    )
    model_mock.predict.return_value = []
    ultralytics_mock.YOLO.return_value = model_mock

    # 設定のモック
    ultralytics_mock.cfg.DEFAULT_CFG_DICT = {}

    with patch.dict("sys.modules", {"ultralytics": ultralytics_mock}):
        yield ultralytics_mock


@pytest.fixture
def mock_cv2():
    """cv2モジュールのモック（画像I/O）"""
    import numpy as np

    cv2_mock = MagicMock()
    cv2_mock.imread.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2_mock.imwrite.return_value = True
    cv2_mock.cvtColor.side_effect = lambda img, code: img
    cv2_mock.resize.side_effect = lambda img, size: np.zeros(
        (size[1], size[0], 3), dtype=np.uint8
    )

    with patch.dict("sys.modules", {"cv2": cv2_mock}):
        yield cv2_mock


@pytest.fixture
def sample_training_config():
    """テスト用のトレーニング設定を返す"""
    return {
        "epochs": 10,
        "batch_size": 16,
        "imgsz": 640,
        "lr0": 0.01,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1,
        "box": 7.5,
        "cls": 0.5,
        "dfl": 1.5,
        "patience": 50,
        "amp": True,
        "workers": 4,
    }


@pytest.fixture
def sample_augmentation_config():
    """テスト用のデータ拡張設定を返す"""
    return {
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 0.0,
        "translate": 0.1,
        "scale": 0.5,
        "shear": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "mixup": 0.0,
    }


@pytest.fixture
def mock_sam2_model():
    """SAM2モデルのモック"""
    import numpy as np

    sam2_mock = MagicMock()
    # マスク生成結果のモック
    mask_result = MagicMock()
    mask_result.masks = [np.ones((100, 100), dtype=bool)]
    sam2_mock.generate.return_value = [mask_result]

    return sam2_mock


@pytest.fixture
def sample_annotation_result():
    """テスト用のアノテーション結果"""
    return {
        "success": True,
        "image_path": "/path/to/image.jpg",
        "label_path": "/path/to/label.txt",
        "class_id": 0,
        "bbox": [100, 100, 200, 200],
        "confidence": 0.95,
    }
