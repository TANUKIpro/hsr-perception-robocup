"""
共通テストフィクスチャ

このファイルはバックエンド・フロントエンド両方のテストで
使用される共通のフィクスチャを定義します。

Note: テストはDockerコンテナ内で実行することを前提としています。
詳細は tests/README.md を参照してください。
"""

import pytest
from pathlib import Path
import tempfile
import shutil


@pytest.fixture
def project_root() -> Path:
    """プロジェクトのルートディレクトリを返す"""
    return Path(__file__).parent.parent


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """テスト用の一時ディレクトリを提供"""
    return tmp_path


@pytest.fixture
def sample_image(temp_dir: Path) -> Path:
    """テスト用のサンプル画像を作成して返す"""
    import numpy as np
    import cv2

    img = np.zeros((480, 640, 3), dtype=np.uint8)
    # 白い正方形を描画
    img[100:200, 100:200] = 255
    path = temp_dir / "sample.jpg"
    cv2.imwrite(str(path), img)
    return path


@pytest.fixture
def sample_mask() -> "np.ndarray":
    """テスト用のバイナリマスクを作成して返す"""
    import numpy as np

    mask = np.zeros((480, 640), dtype=bool)
    mask[100:200, 100:200] = True
    return mask


@pytest.fixture
def sample_rgb_image() -> "np.ndarray":
    """テスト用のRGB画像を作成して返す"""
    import numpy as np

    img = np.zeros((480, 640, 3), dtype=np.uint8)
    # 赤い正方形
    img[100:200, 100:200, 0] = 255
    return img


@pytest.fixture
def sample_object_classes_json(temp_dir: Path) -> Path:
    """テスト用のobject_classes.jsonを作成"""
    import json

    config = {
        "categories": ["food", "container"],
        "objects": [
            {"id": 0, "name": "apple", "category": "food"},
            {"id": 1, "name": "cup", "category": "container"},
        ],
    }
    path = temp_dir / "object_classes.json"
    with open(path, "w") as f:
        json.dump(config, f)
    return path


@pytest.fixture
def sample_dataset_yaml(temp_dir: Path) -> Path:
    """テスト用のdata.yamlを作成"""
    import yaml

    # データセット構造を作成
    train_dir = temp_dir / "train" / "images"
    val_dir = temp_dir / "val" / "images"
    train_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)

    config = {
        "path": str(temp_dir),
        "train": "train/images",
        "val": "val/images",
        "names": {0: "apple", 1: "cup"},
    }
    path = temp_dir / "data.yaml"
    with open(path, "w") as f:
        yaml.dump(config, f)
    return path


@pytest.fixture
def sample_yolo_labels(temp_dir: Path) -> Path:
    """テスト用のYOLOラベルファイルを作成"""
    labels_dir = temp_dir / "labels"
    labels_dir.mkdir(parents=True)

    # 正常なラベルファイル
    label_path = labels_dir / "image001.txt"
    with open(label_path, "w") as f:
        f.write("0 0.5 0.5 0.2 0.2\n")
        f.write("1 0.3 0.7 0.1 0.15\n")

    return labels_dir
