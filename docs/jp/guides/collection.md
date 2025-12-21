# Collection - データ収集

ROS2カメラトピックからの画像収集。単発撮影、バースト撮影、動画録画に対応。

---

## 関連ファイル

| ファイル | 説明 |
|---------|------|
| `scripts/capture/capture_app.py` | Tkinter GUI（バースト/単発撮影） |
| `scripts/capture/record_app.py` | 動画録画アプリ |
| `scripts/capture/burst_capture.py` | CLIバースト撮影 |
| `scripts/capture/capture_frame.py` | 単発フレーム撮影 |
| `scripts/capture/preview_window.py` | プレビューウィンドウ |
| `src/hsr_perception/hsr_perception/continuous_capture_node.py` | ROS2ノード |
| `src/hsr_perception/srv/SetClass.srv` | クラス設定サービス |
| `src/hsr_perception/srv/StartBurst.srv` | バースト開始サービス |
| `src/hsr_perception/srv/GetStatus.srv` | 状態取得サービス |

---

## 使用技術

- **ROS2 Humble** - ロボットミドルウェア
- **OpenCV** - 画像処理
- **Tkinter** - GUI
- **NumPy** - 配列操作

---

## キャプチャモード

### 1. 単発撮影
現在のフレームを1枚保存

### 2. バースト撮影
指定間隔で連続撮影（デフォルト: 0.2秒間隔、50枚）

### 3. 動画録画
MP4動画を録画し、後から均等にフレーム抽出

---

## Docker実行方法

### Xtionカメラノード起動

```bash
# OpenNI2カメラノード起動（Xtion用）
docker compose run --rm hsr-perception ros2-camera
```

### HSRキャプチャノード起動

```bash
# HSR向けキャプチャノード起動
docker compose run --rm hsr-perception ros2-capture
```

### start.shを使用（推奨）

```bash
# Xtionカメラ接続時は自動でカメラノードを起動
./start.sh
```

`start.sh`は以下を自動で行います：
- Xtionカメラの検出
- udevルール設定
- カメラノードのバックグラウンド起動

---

## GUIアプリ（Streamlit UI推奨）

Streamlit UIからデータ収集を行う場合は、Collectionページを使用してください。

### 収集方法

| 方法 | 説明 |
|-----|------|
| ROS2 Camera | ROS2トピックからリアルタイム収集 |
| Local Camera | Webカメラで撮影 |
| File Upload | 画像ファイルをアップロード |
| Folder Import | フォルダから一括インポート |

---

## ROS2ノード

### サービス（Docker内から実行）

```bash
# Dockerコンテナにアクセス
docker compose run --rm hsr-perception bash

# コンテナ内でROS2コマンド実行
ros2 service call /continuous_capture/set_class \
    hsr_perception/srv/SetClass "{class_id: 0}"

ros2 service call /continuous_capture/start_burst \
    hsr_perception/srv/StartBurst "{num_images: 100, interval_seconds: 0.2}"

ros2 service call /continuous_capture/stop_burst std_srvs/srv/Empty

ros2 service call /continuous_capture/get_status \
    hsr_perception/srv/GetStatus
```

---

## 設定パラメータ

### バースト撮影

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `num_images` | 50 | 撮影枚数 |
| `interval_seconds` | 0.2 | 撮影間隔（秒） |

### 画像保存

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `jpeg_quality` | 95 | JPEG圧縮品質 |

---

## 出力形式

### ファイル名形式

```
<class_name>_YYYYMMDD_HHMMSS_ffffff.jpg
```

- `class_name`: オブジェクト名
- `YYYYMMDD_HHMMSS`: 日時
- `ffffff`: マイクロ秒（一意性確保）

### ディレクトリ構造

```
raw_captures/
└── <class_name>/
    ├── <class_name>_20241208_103045_123456.jpg
    ├── <class_name>_20241208_103045_323456.jpg
    └── ...

videos/
└── <class_name>_20241208-10-30.mp4
```

---

## 画像エンコーディング対応

ROS2 Imageメッセージの以下のエンコーディングをサポート:

- `rgb8` - RGB 8bit
- `bgr8` - BGR 8bit（OpenCVネイティブ）
- `mono8` - グレースケール 8bit
- `16UC1` - 深度画像 16bit
- `32FC1` - 深度画像 32bit float

---

## トピックプリセット

| プリセット | トピック |
|-----------|---------|
| HSR | `/hsrb/head_rgbd_sensor/rgb/image_rect_color` |
| 汎用USB | `/usb_cam/image_raw` |
| RealSense | `/camera/color/image_raw` |
| Xtion | `/camera/rgb/image_raw` |
