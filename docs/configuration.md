# Configuration Guide / 設定ガイド

HSR Perception Pipeline configuration reference.

## Directory Structure / ディレクトリ構造

```
hsr-perception-robocup/
├── config/
│   └── object_classes.json      # Class definitions / クラス定義
├── datasets/
│   ├── raw_captures/            # Raw images by class / 生画像
│   ├── annotated/               # Annotated datasets / アノテーション済み
│   └── backgrounds/             # Background images / 背景画像
├── models/
│   ├── pretrained/              # Pre-trained models / 事前学習モデル
│   └── finetuned/               # Fine-tuned models / ファインチューニング済み
├── app/
│   └── data/
│       ├── collected_images/    # App collected images / アプリ収集画像
│       ├── reference_images/    # Reference images / 参照画像
│       ├── object_registry.json # App object registry / オブジェクト登録情報
│       └── tasks/               # Task status files / タスク状態ファイル
└── scripts/                     # Pipeline scripts / パイプラインスクリプト
```

---

## Object Classes Configuration / オブジェクトクラス設定

**File / ファイル:** `config/object_classes.json`

This is the main configuration file for object classes used by annotation, training, and evaluation scripts.

### Schema / スキーマ

```json
{
  "version": "1.0.0",
  "competition": "RoboCup@Home 2025",
  "created_at": "2025-01-15T10:00:00Z",

  "categories": [
    {
      "id": 0,
      "name": "food",
      "display_name": "Food Items",
      "color": "#FF6B6B"
    }
  ],

  "objects": [
    {
      "class_id": 0,
      "class_name": "apple",
      "category_id": 0,
      "object_type": "standard",
      "target_samples": 100,
      "collected_samples": 0,
      "last_updated": null,
      "notes": ""
    }
  ],

  "settings": {
    "default_target_samples": 100,
    "min_samples_for_training": 50,
    "train_val_split": 0.85,
    "image_format": "jpg",
    "jpeg_quality": 95
  }
}
```

### Field Descriptions / フィールド説明

#### Root Fields / ルートフィールド

| Field | Type | Description (EN) | 説明 (JP) |
|-------|------|------------------|-----------|
| `version` | string | Config version | 設定バージョン |
| `competition` | string | Competition name | 競技名 |
| `created_at` | string/null | Creation timestamp | 作成日時 |

#### Category Fields / カテゴリフィールド

| Field | Type | Description (EN) | 説明 (JP) |
|-------|------|------------------|-----------|
| `id` | int | Category ID (0-indexed) | カテゴリID |
| `name` | string | Internal name (lowercase) | 内部名（小文字） |
| `display_name` | string | Display name | 表示名 |
| `color` | string | Hex color code | カラーコード |

#### Object Fields / オブジェクトフィールド

| Field | Type | Description (EN) | 説明 (JP) |
|-------|------|------------------|-----------|
| `class_id` | int | Class ID (0-indexed, used in YOLO) | クラスID（YOLOで使用） |
| `class_name` | string | Class name (lowercase, no spaces) | クラス名 |
| `category_id` | int | Parent category ID | 親カテゴリID |
| `object_type` | string | "standard" / "consistent" / "unknown" | 物体タイプ |
| `target_samples` | int | Target number of samples | 目標サンプル数 |
| `collected_samples` | int | Current collected count | 収集済み数 |
| `last_updated` | string/null | Last update timestamp | 最終更新日時 |
| `notes` | string | Additional notes | メモ |

#### Settings / 設定

| Field | Type | Default | Description (EN) | 説明 (JP) |
|-------|------|---------|------------------|-----------|
| `default_target_samples` | int | 100 | Default target per class | デフォルト目標数 |
| `min_samples_for_training` | int | 50 | Minimum for training | 学習最小数 |
| `train_val_split` | float | 0.85 | Train/val split ratio | 学習/検証分割比 |
| `image_format` | string | "jpg" | Output format | 出力形式 |
| `jpeg_quality` | int | 95 | JPEG quality | JPEG品質 |

### Example: Competition Day Update / 大会当日の更新例

```json
{
  "version": "1.0.0",
  "competition": "iHR-C3 2025",
  "created_at": "2025-03-15T09:00:00+09:00",

  "categories": [
    {"id": 0, "name": "food", "display_name": "Food", "color": "#FF6B6B"},
    {"id": 1, "name": "drink", "display_name": "Drink", "color": "#4ECDC4"},
    {"id": 2, "name": "kitchen_item", "display_name": "Kitchen Item", "color": "#45B7D1"},
    {"id": 3, "name": "task_item", "display_name": "Task Item", "color": "#96CEB4"},
    {"id": 4, "name": "bag", "display_name": "Bag", "color": "#FFEAA7"}
  ],

  "objects": [
    {"class_id": 0, "class_name": "noodles", "category_id": 0, "object_type": "consistent", "target_samples": 100, "collected_samples": 0, "notes": ""},
    {"class_id": 1, "class_name": "tea_bag", "category_id": 0, "object_type": "consistent", "target_samples": 100, "collected_samples": 0, "notes": ""},
    {"class_id": 2, "class_name": "potato_chips", "category_id": 0, "object_type": "consistent", "target_samples": 100, "collected_samples": 0, "notes": ""},
    {"class_id": 3, "class_name": "gummy", "category_id": 0, "object_type": "consistent", "target_samples": 100, "collected_samples": 0, "notes": ""},
    {"class_id": 4, "class_name": "redbull", "category_id": 1, "object_type": "consistent", "target_samples": 100, "collected_samples": 0, "notes": "Heavy Item, Liquid"},
    {"class_id": 5, "class_name": "aquarius", "category_id": 1, "object_type": "consistent", "target_samples": 100, "collected_samples": 0, "notes": "Heavy Item, Liquid"},
    {"class_id": 6, "class_name": "lychee", "category_id": 1, "object_type": "consistent", "target_samples": 100, "collected_samples": 0, "notes": "Heavy Item, Liquid"},
    {"class_id": 7, "class_name": "coffee", "category_id": 1, "object_type": "consistent", "target_samples": 100, "collected_samples": 0, "notes": "Heavy Item, Liquid"},
    {"class_id": 8, "class_name": "detergent", "category_id": 2, "object_type": "consistent", "target_samples": 100, "collected_samples": 0, "notes": "Without content"},
    {"class_id": 9, "class_name": "cup", "category_id": 2, "object_type": "consistent", "target_samples": 100, "collected_samples": 0, "notes": ""},
    {"class_id": 10, "class_name": "lunch_box", "category_id": 2, "object_type": "consistent", "target_samples": 100, "collected_samples": 0, "notes": ""},
    {"class_id": 11, "class_name": "bowl", "category_id": 2, "object_type": "consistent", "target_samples": 100, "collected_samples": 0, "notes": ""},
    {"class_id": 12, "class_name": "dice", "category_id": 3, "object_type": "consistent", "target_samples": 100, "collected_samples": 0, "notes": "Tiny Item, 1.6x1.6x1.6cm"},
    {"class_id": 13, "class_name": "light_bulb", "category_id": 3, "object_type": "consistent", "target_samples": 100, "collected_samples": 0, "notes": "Without content"},
    {"class_id": 14, "class_name": "block", "category_id": 3, "object_type": "consistent", "target_samples": 100, "collected_samples": 0, "notes": ""},
    {"class_id": 15, "class_name": "glue_gun", "category_id": 3, "object_type": "consistent", "target_samples": 100, "collected_samples": 0, "notes": "Without plastic container"},
    {"class_id": 16, "class_name": "shopping_bag", "category_id": 4, "object_type": "consistent", "target_samples": 100, "collected_samples": 0, "notes": ""}
  ],

  "settings": {
    "default_target_samples": 100,
    "min_samples_for_training": 50,
    "train_val_split": 0.85,
    "image_format": "jpg",
    "jpeg_quality": 95
  }
}
```

---

## Environment Variables / 環境変数

Configure the app behavior through environment variables.

| Variable | Default | Description (EN) | 説明 (JP) |
|----------|---------|------------------|-----------|
| `HSR_ENV` | `local` | Environment: local/docker | 環境: local/docker |
| `HSR_ROS2_ENABLED` | `true` | Enable ROS2 features | ROS2機能の有効化 |
| `HSR_GPU_ENABLED` | `true` | Enable GPU features | GPU機能の有効化 |
| `ROS2_SOURCE_SCRIPT` | `/opt/ros/humble/setup.bash` | ROS2 setup script path | ROS2セットアップスクリプトパス |

### Example / 使用例

```bash
# Disable ROS2 features (local camera only)
HSR_ROS2_ENABLED=false streamlit run app/main.py

# Docker environment
HSR_ENV=docker streamlit run app/main.py
```

---

## ROS2 Configuration / ROS2設定

### Capture Node Services / 撮影ノードサービス

| Service | Type | Description |
|---------|------|-------------|
| `/continuous_capture/set_class` | `hsr_perception/srv/SetClass` | Set current capture class |
| `/continuous_capture/start_burst` | `hsr_perception/srv/StartBurst` | Start burst capture |
| `/continuous_capture/get_status` | `hsr_perception/srv/GetStatus` | Get capture status |

### Default Image Topics / デフォルト画像トピック

HSR Topics:
- `/hsrb/head_rgbd_sensor/rgb/image_rect_color` (Recommended)
- `/hsrb/head_rgbd_sensor/rgb/image_raw`
- `/hsrb/hand_camera/image_raw`
- `/hsrb/head_l_stereo_camera/image_rect_color`
- `/hsrb/head_r_stereo_camera/image_rect_color`

Generic Topics:
- `/camera/color/image_raw` (RealSense)
- `/camera/rgb/image_raw`
- `/usb_cam/image_raw`
- `/image_raw`

---

## YOLO Dataset Format / YOLOデータセット形式

Auto-annotation produces datasets in YOLO format:

```
datasets/annotated/{session_name}/
├── data.yaml                  # Dataset configuration
├── images/
│   ├── train/                 # Training images
│   │   ├── apple_001.jpg
│   │   └── ...
│   └── val/                   # Validation images
│       ├── apple_050.jpg
│       └── ...
└── labels/
    ├── train/                 # Training labels (YOLO format)
    │   ├── apple_001.txt
    │   └── ...
    └── val/                   # Validation labels
        ├── apple_050.txt
        └── ...
```

### data.yaml Structure / data.yaml構造

```yaml
path: /absolute/path/to/dataset
train: images/train
val: images/val

nc: 17  # Number of classes
names:
  0: noodles
  1: tea_bag
  2: potato_chips
  # ...
```

### Label Format / ラベル形式

Each `.txt` file contains one line per object:

```
<class_id> <x_center> <y_center> <width> <height>
```

Values are normalized (0-1) relative to image dimensions.

Example:
```
0 0.5 0.5 0.3 0.4
```

---

## Training Configuration / 学習設定

### Recommended Settings / 推奨設定

| Parameter | Competition | Fast Testing |
|-----------|-------------|--------------|
| Base Model | yolov8m.pt | yolov8n.pt |
| Epochs | 50 | 10 |
| Batch Size | 16 | 32 |
| Image Size | 640 | 640 |
| Workers | 8 | 4 |

### Competition Targets / 競技目標

| Metric | Target |
|--------|--------|
| mAP@50 | ≥ 85% |
| mAP@50-95 | ≥ 60% |
| Inference Time | ≤ 100ms |

---

## File Paths / ファイルパス

### App Internal Paths / アプリ内部パス

| Purpose | Path |
|---------|------|
| App collected images | `app/data/collected_images/{object_name}/` |
| Reference images | `app/data/reference_images/{object_id}/` |
| Object registry | `app/data/object_registry.json` |
| Task status | `app/data/tasks/{task_id}.json` |

### Pipeline Paths / パイプラインパス

| Purpose | Path |
|---------|------|
| Raw captures | `datasets/raw_captures/{class_name}/` |
| Backgrounds | `datasets/backgrounds/` |
| Annotated datasets | `datasets/annotated/{session_name}/` |
| Pre-trained models | `models/pretrained/` |
| Fine-tuned models | `models/finetuned/{run_name}/` |
| Class config | `config/object_classes.json` |

---

## Troubleshooting / トラブルシューティング

### Path Sync Issues / パス同期の問題

If data doesn't appear in annotation:

1. Check app collected images: `app/data/collected_images/`
2. Run sync from Collection page: "Sync to Datasets Directory"
3. Verify data in: `datasets/raw_captures/`

### Config Sync Issues / 設定同期の問題

If classes don't match between app and scripts:

1. Go to Settings page
2. Click "Export to YOLO Config"
3. This updates `config/object_classes.json`

### Task Files / タスクファイル

Task status files are stored in `app/data/tasks/`. To reset stuck tasks:

```bash
# View task status
cat app/data/tasks/{task_id}.json

# Remove stale tasks
rm app/data/tasks/*.json
```
