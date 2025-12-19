# HSR Perception Pipeline for RoboCup@Home

RoboCup@Home大会向けのHSR（Human Support Robot）用物体認識パイプライン。
大会当日2〜3時間の制約内でデータ収集からモデルデプロイまでを完了させるためのツール群を提供します。

**開発環境**: Ubuntu 22.04 / ROS2 Humble / Python

---

## 大会当日ワークフロー

```mermaid
flowchart LR
    A[Data Collection<br/>~40min] --> B[Auto-Annotation<br/>~25min]
    B --> C[Fine-tuning<br/>~45min]
    C --> D[Evaluation<br/>~15min]
    D --> E[Deploy]

    A -.- A1[ROS2 node]
    B -.- B1[背景差分/SAM2]
    C -.- C1[YOLOv8m]
    D -.- D1[mAP check]
    E -.- E1[HSR]
```

---

## Dockerでの実行（推奨）

Docker環境を使用すると、依存関係のインストールなしですぐに利用できます。

### 前提条件

| 項目 | 要件 |
|------|------|
| Docker | 24.0以上 |
| Docker Compose | v2.0以上 |
| NVIDIA Driver | 525以上 |
| NVIDIA Container Toolkit | インストール済み |

### クイックスタート

**推奨: start.shを使用**

```bash
# 起動スクリプトを実行（初回はイメージビルド、udevルール設定を自動実行）
./start.sh

# ブラウザで http://localhost:8501 を開く
```

`start.sh`は以下を自動で行います：
- 初回起動時: Dockerイメージのビルド、Xtion用udevルールのインストール
- X11アクセスの設定（GUIアプリ用）
- Xtionカメラの検出と自動起動
- **Ctrl+Cで終了時に`docker compose down`を自動実行**

**オプション:**
```bash
./start.sh --build        # イメージを強制再ビルド
./start.sh --tensorboard  # TensorBoard付きで起動（ポート6006）
./start.sh -d             # バックグラウンド起動
./start.sh --help         # ヘルプを表示
```

**手動で起動する場合:**

```bash
# 1. イメージをビルド（初回のみ、約10-15分）
docker compose build

# 2. X11アクセスを許可（GUIアプリを使用する場合）
xhost +local:docker

# 3. Streamlit UIを起動
docker compose up

# ブラウザで http://localhost:8501 を開く

# 4. 停止時
docker compose down
```

### GUIアプリケーション（tkinter）を使用する場合

Docker内からGUIアプリ（Xtion Test App等）を起動する場合は、以下の設定が必要です：

```bash
# ホスト側でX11アクセスを許可（Docker起動前に毎回実行）
xhost +local:docker

# その後、通常通りDocker Composeを起動
docker compose up
```

**注意**: `xhost +local:docker`はセキュリティを緩和するコマンドです。使用後に`xhost -local:docker`で元に戻せます。

### Docker Composeコマンド

| コマンド | 説明 |
|---------|------|
| `docker compose up` | Streamlit UI起動 |
| `docker compose up -d` | バックグラウンド起動 |
| `docker compose down` | 停止 |
| `docker compose run --rm hsr-perception bash` | シェルアクセス |
| `docker compose run --rm hsr-perception train --dataset /workspace/datasets/data.yaml` | 学習実行 |
| `docker compose run --rm hsr-perception annotate --help` | アノテーション |
| `docker compose run --rm hsr-perception evaluate --help` | モデル評価 |

### Xtionカメラ使用時

1. udevルールをインストール（初回のみ）:
```bash
sudo cp docker/99-xtion.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
sudo udevadm trigger
```

2. カメラノード起動:
```bash
docker compose run --rm hsr-perception ros2-camera
```

### 含まれるコンポーネント

| コンポーネント | バージョン | 備考 |
|---------------|-----------|------|
| Python | 3.10 | |
| PyTorch | 2.x | CUDA 12.1対応 |
| Ultralytics | >=8.3.0 | YOLOv8 |
| SAM2 | latest | Segment Anything 2 |
| ROS2 | Humble | OpenNI2対応 |
| Streamlit | >=1.28.0 | Web UI |

**事前ダウンロード済みモデル**: yolov8m.pt, yolov8n.pt

---

## ローカルセットアップ

### 必要環境

| 項目 | 要件 |
|------|------|
| OS | Ubuntu 22.04 |
| Python | 3.10+ |
| GPU | CUDA対応GPU（VRAM 6GB以上推奨） |
| ROS2 | Humble（データ収集機能使用時） |

### 1. Python依存パッケージ

```bash
# 基本パッケージのインストール
pip install -r requirements.txt

# GPU使用時: PyTorch CUDA版をインストール（推奨）
# https://pytorch.org/get-started/locally/ で環境に合ったコマンドを確認
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 2. SAM2のセットアップ（自動アノテーションで使用）

```bash
# SAM2パッケージのインストール
pip install git+https://github.com/facebookresearch/segment-anything-2.git

# SAM2モデルのダウンロード（Base Plusを推奨）
# models/ ディレクトリに配置してください
wget -P models https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt
```

**利用可能なSAM2モデル:**
| モデル | サイズ | 用途 |
|--------|--------|------|
| sam2.1_hiera_tiny.pt | 最小 | 高速処理優先 |
| sam2.1_hiera_small.pt | 小 | バランス型 |
| sam2.1_hiera_base_plus.pt | 中（推奨） | 精度と速度のバランス |
| sam2.1_hiera_large.pt | 大 | 高精度優先 |

### 3. YOLOモデルの事前取得（オプション）

Ultralyticsは初回実行時に自動ダウンロードしますが、オフライン環境に備えて事前取得を推奨します。

```bash
python -c "from ultralytics import YOLO; YOLO('yolov8m.pt')"
```

**利用可能なYOLOv8モデル:**
| モデル | パラメータ | VRAM目安 | 推奨バッチサイズ | mAP (COCO) | 用途 |
|--------|-----------|---------|----------------|-----------|------|
| yolov8n.pt | 3.2M | ~2GB | 32 | 37.3 | 最速・エッジデバイス向け |
| yolov8s.pt | 11.2M | ~4GB | 16-32 | 44.9 | 高速・リソース制限時 |
| yolov8m.pt | 25.9M | ~6GB | 16 | 50.2 | バランス型（推奨） |
| yolov8l.pt | 43.7M | ~10GB | 16-32 | 52.9 | 高精度優先 |
| yolov8x.pt | 68.2M | ~14GB | 32-64 | 53.9 | 最高精度 |

※学習時間の目安（50エポック、640x640、データ約1000枚の場合）:
- yolov8n/s: 約20-40分
- yolov8m: 約45-60分（競技設定）
- yolov8l/x: 約60-90分以上

### 4. ROS2パッケージのビルド（データ収集機能使用時）

```bash
cd src
colcon build --packages-select hsr_perception
source install/setup.bash
```

---

## 使い方

### GUIアプリ（推奨）

Streamlitベースの統合GUIで、全工程を操作できます。

```bash
./run_app.sh
# または
streamlit run app/main.py
```

ブラウザで http://localhost:8501 を開いてください。

**機能:**
- Dashboard - パイプライン全体の進捗・状態可視化
- Registry - オブジェクト登録・参照画像管理
- Collection - データ収集（ROS2/ローカルカメラ/ファイル）
- Annotation - 自動アノテーション実行
- Training - YOLOv8 fine-tuning・進捗監視
- Evaluation - モデル評価・可視化テスト

### CLIスクリプト

#### Step 1: クラス設定
`config/object_classes.json` を大会で配布されたオブジェクトリストに合わせて編集

#### Step 2: データ収集（ROS2使用時）
```bash
# 連続撮影ノードの起動
ros2 launch hsr_perception capture.launch.py

# クラス設定・バースト撮影
ros2 service call /continuous_capture/set_class hsr_perception/srv/SetClass "{class_id: 0}"
ros2 service call /continuous_capture/start_burst hsr_perception/srv/StartBurst "{num_images: 100, interval_seconds: 0.2}"
```

#### Step 3: 自動アノテーション
```bash
# 背景差分方式（推奨・高速）
python scripts/annotation/auto_annotate.py \
    --method background \
    --background datasets/backgrounds/white_sheet.jpg \
    --input-dir datasets/raw_captures \
    --output-dir datasets/competition_day \
    --class-config config/object_classes.json

# SAM2方式（フォールバック・GPU必要）
python scripts/annotation/auto_annotate.py \
    --method sam2 \
    --input-dir datasets/raw_captures \
    --output-dir datasets/competition_day \
    --class-config config/object_classes.json
```

#### Step 4: Fine-tuning
```bash
# 競技用設定でYOLOv8mをfine-tuning
python scripts/training/quick_finetune.py \
    --dataset datasets/competition_day/data.yaml \
    --model yolov8m.pt \
    --output models/finetuned

# 高速モード（小さいモデル）
python scripts/training/quick_finetune.py \
    --dataset datasets/competition_day/data.yaml \
    --fast
```

#### Step 5: 評価
```bash
# モデル評価（mAP、推論時間）
python scripts/evaluation/evaluate_model.py \
    --model models/finetuned/competition_*/weights/best.pt \
    --dataset datasets/competition_day/data.yaml

# 可視化検証
python scripts/evaluation/visual_verification.py \
    --model models/finetuned/competition_*/weights/best.pt \
    --batch-dir datasets/competition_day/images/val
```

---

## ディレクトリ構成

```
hsr-perception-robocup/
├── app/                    # Streamlit GUIアプリ
│   ├── main.py             # メインエントリーポイント
│   ├── pages/              # 各ページ（Registry, Collection, etc.）
│   ├── services/           # バックエンドサービス
│   └── components/         # 共有UIコンポーネント
│
├── scripts/                # CLIスクリプト
│   ├── common/             # 共通ユーティリティ
│   ├── capture/            # キャプチャツール
│   ├── annotation/         # 自動アノテーション
│   ├── training/           # 学習パイプライン
│   └── evaluation/         # 評価ツール
│
├── src/hsr_perception/     # ROS2パッケージ
│   ├── hsr_perception/     # ノード実装
│   ├── srv/                # サービス定義
│   └── launch/             # Launchファイル
│
├── config/                 # 設定ファイル
│   └── object_classes.json # クラス定義
│
├── docs/                   # ドキュメント
├── models/                 # 学習済みモデル（.gitignore）
├── datasets/               # データセット（.gitignore）
└── profiles/               # 競技プロファイル
```

各ファイルの詳細は [docs/implementation.md](docs/implementation.md) を参照してください。

---

## 技術スタック

| カテゴリ | 技術 |
|---------|------|
| 物体検出 | YOLOv8 (Ultralytics) |
| セグメンテーション | SAM2 (Meta) |
| GUI | Streamlit |
| ロボティクス | ROS2 Humble, cv_bridge |

---

## テスト

### テスト構成

| カテゴリ | フレームワーク | 対象 |
|---------|--------------|------|
| Backend Unit | pytest | scripts/, src/hsr_perception/ |
| Frontend Unit | pytest + mock | app/services/, app/components/ |
| Frontend E2E | Playwright | ブラウザUIテスト |

### テスト実行

```bash
# 全テスト（Unit）
pytest tests/ -v

# Backendテスト
pytest tests/backend/ -v

# Frontendユニットテスト
pytest tests/frontend/ -v

# E2Eテスト（要Docker起動）
cd tests/e2e && npm test

# カバレッジレポート
pytest tests/ --cov=app --cov=scripts --cov-report=html
```

### E2Eテスト（Playwright）

Playwright を使用したブラウザUIテストです。Streamlit UIの動作を自動検証します。

**前提条件:**
- Node.js 18+
- Docker起動済み（`docker compose up -d`）

**セットアップ:**
```bash
cd tests/e2e
npm install
npx playwright install chromium
```

**実行:**
```bash
# ヘッドレスモード（CI向け）
npm test

# UIモード（デバッグ用）
npm run test:ui

# 特定テストのみ
npx playwright test dashboard.spec.ts

# テストレポート表示
npx playwright show-report
```

**テスト構成:**
- `specs/smoke/` - 起動確認・ページ遷移
- `specs/pages/` - 各ページの機能テスト
- `page-objects/` - Page Objectパターン実装

---

## ドキュメント

- [アプリガイド（日本語）](docs/app_guide.md)
- [App Guide (English)](docs/app_guide_en.md)
- [実装リファレンス](docs/implementation.md)
- [設定リファレンス](docs/configuration.md)

---

## 参考資料

- [RoboCup@Home Rulebook](https://github.com/RoboCupAtHome/RuleBook)
- [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/)
- [Segment Anything 2](https://github.com/facebookresearch/sam2)
