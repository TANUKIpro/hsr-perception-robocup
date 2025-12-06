# HSR Object Manager アプリガイド

RoboCup@Home競技用物体認識パイプラインのGUIアプリケーション使用ガイド。

## 概要

HSR Object ManagerはStreamlitベースのWebアプリケーションで、物体認識モデルの作成から評価までの全工程をGUIで操作できます。

**主要機能:**
- 📊 Dashboard: 収集進捗とパイプライン状態の可視化
- 📋 Registry: オブジェクト登録・参照画像管理
- 📸 Collection: ROS2/カメラ/ファイルからのデータ収集
- 🏷️ Annotation: 自動アノテーションパイプライン実行
- 🎓 Training: YOLOv8 fine-tuning実行
- 📈 Evaluation: モデル評価・競技要件チェック
- ⚙️ Settings: システム設定・状態確認

---

## 起動方法

```bash
# 方法1: シェルスクリプト
./run_app.sh

# 方法2: 直接起動
streamlit run app/main.py

# 方法3: カスタムポート
streamlit run app/main.py --server.port 8502
```

ブラウザで http://localhost:8501 を開きます。

---

## 各ページの使い方

### 📊 Dashboard

パイプライン全体の状態を一目で確認できます。

**表示内容:**
- 登録オブジェクト数
- 収集済み画像数・目標達成率
- カテゴリ別進捗
- アノテーション済みデータセット数
- 学習済みモデル数
- 実行中タスク数

**アクション:**
- 「Export to YOLO Config」: `config/object_classes.json` を更新

### 📋 Registry

オブジェクトの登録・管理を行います。

**View Objects タブ:**
- 登録済みオブジェクト一覧表示
- カテゴリでフィルタリング
- 参照画像の追加
- 「📸 Collect」でCollectionページへ移動
- オブジェクトの削除

**Add New Object タブ:**
- ID（自動採番）
- 名前（英小文字、スペースなし）
- 表示名
- カテゴリ選択
- 目標サンプル数
- プロパティ（重い/小さい/液体含む/サイズ）

**Quick Import:**
- 「Import iHR Standard Objects」でiHR標準オブジェクトを一括登録

### 📸 Collection

画像データの収集を行います。4つの方法が利用可能です。

#### 🤖 ROS2 Camera タブ

HSRやROS2対応カメラから直接画像を収集します。

**必要条件:**
- ROS2 Humbleがインストール・ソース済み
- 連続撮影ノードが起動中

```bash
# ノード起動コマンド
ros2 launch hsr_perception capture.launch.py
```

**操作手順:**
1. 対象オブジェクトを選択
2. 画像トピックを選択（自動検出 or プリセット）
3. 「Set Capture Class」でクラスを設定
4. 撮影枚数・間隔を指定
5. 「Start Burst Capture」で連続撮影開始

**トピックプリセット:**
- HSR: `/hsrb/head_rgbd_sensor/rgb/image_rect_color`
- 汎用USB: `/usb_cam/image_raw`
- RealSense: `/camera/color/image_raw`

#### 📷 Local Camera タブ

Webカメラやデバイス内蔵カメラで撮影します。

1. オブジェクトを選択
2. 「Take a photo」で撮影
3. 自動保存

#### 📁 File Upload タブ

画像ファイルをアップロードします。

1. オブジェクトを選択
2. 複数画像を選択してアップロード
3. 自動保存・カウント更新

#### 📂 Folder Import タブ

フォルダパスを指定して一括インポートします。

1. オブジェクトを選択
2. フォルダパスを入力
3. 「Import from Folder」で実行

**データ同期:**
収集後、「Sync to Datasets Directory」で `datasets/raw_captures/` にデータを同期します。

### 🏷️ Annotation

自動アノテーションパイプラインを実行します。

**前提条件:**
- `datasets/raw_captures/` に画像データが存在
- クラス設定が `config/object_classes.json` に定義済み

**設定項目:**

| 項目 | 説明 | 推奨値 |
|------|------|--------|
| Annotation Method | background / sam2 | background（高速） |
| Train/Val Split | 学習/検証分割比率 | 0.85 |
| Background Image | 背景画像（background方式のみ） | 白背景推奨 |
| Minimum Contour Area | 最小輪郭面積（背景方式） | 500 |

**実行手順:**
1. 方式を選択
2. 背景画像を選択/アップロード（background方式の場合）
3. 分割比率を設定
4. 「Start Annotation」で実行

**推定時間:**
- 100画像あたり約5分

**出力:**
- `datasets/annotated/{session_name}/`
  - `images/train/`, `images/val/`
  - `labels/train/`, `labels/val/`
  - `data.yaml`

### 🎓 Training

YOLOv8 fine-tuningを実行します。

**設定項目:**

| 項目 | 説明 | 推奨値 |
|------|------|--------|
| Dataset | アノテーション済みデータセット | - |
| Base Model | ベースモデル | yolov8m.pt |
| Epochs | エポック数 | 50 |
| Batch Size | バッチサイズ | 16 |
| Fast Mode | 高速モード（テスト用） | OFF |

**モデル選択ガイド:**
- `yolov8m.pt`: 精度重視（競技推奨）、学習時間: 約45分
- `yolov8s.pt`: バランス型、学習時間: 約30分
- `yolov8n.pt`: 高速、学習時間: 約20分

**GPU確認:**
ページ内で自動的にGPU利用可能性をチェックします。

**実行手順:**
1. データセットを選択
2. ベースモデル・エポック数を設定
3. 「Start Training」で実行

**出力:**
- `models/finetuned/{run_name}/weights/best.pt`
- `models/finetuned/{run_name}/weights/last.pt`
- `models/finetuned/{run_name}/training_result.json`

### 📈 Evaluation

学習済みモデルを評価します。

**Run Evaluation タブ:**

1. モデルを選択
2. データセットを選択
3. 信頼度閾値を設定（デフォルト: 0.25）
4. 「Start Evaluation」で実行

**評価指標:**
- mAP@50: 目標 ≥ 85%
- mAP@50-95
- Precision / Recall
- 推論時間: 目標 ≤ 100ms
- 競技要件チェック（自動判定）

**Results タブ:**
過去の評価結果一覧と詳細表示。

**Visual Test タブ:**
単一画像での予測テスト。

1. モデルを選択
2. 画像をアップロード or データセットから選択
3. 「Run Prediction」で実行
4. 検出結果を可視化

### ⚙️ Settings

システム設定と状態確認を行います。

**Data Management:**
- Export to YOLO Config: クラス設定をエクスポート
- Update All Collection Counts: 収集数を再集計
- Sync All to Datasets: 全データを同期

**Category Management:**
- カテゴリの追加

**System Status:**
- ROS2接続状態
- GPU利用可能性
- トピック数・ノード数

**Data Paths:**
- 各種パスの確認

---

## 競技当日ワークフロー

### 準備（競技開始前）

1. アプリを起動: `./run_app.sh`
2. Registry → Quick Import で標準オブジェクトを登録
3. 背景画像を準備（白いシートなど）

### Phase 1: データ収集（40分）

1. Collection → ROS2 Camera タブ
2. 連続撮影ノードを起動（別ターミナル）
3. 各オブジェクトについて:
   - オブジェクトを選択
   - クラス設定
   - バースト撮影（50-100枚）
   - オブジェクトを回転させながら複数回撮影

### Phase 2: アノテーション（25分）

1. Annotation ページへ移動
2. Background方式を選択
3. 背景画像を選択
4. 「Start Annotation」
5. 完了まで待機

### Phase 3: 学習（45分）

1. Training ページへ移動
2. 作成したデータセットを選択
3. `yolov8m.pt` / 50エポックを設定
4. 「Start Training」
5. 進捗バーで確認

### Phase 4: 評価（15分）

1. Evaluation ページへ移動
2. 学習済みモデルを選択
3. 「Start Evaluation」
4. 競技要件チェック:
   - mAP@50 ≥ 85% ✓
   - 推論時間 ≤ 100ms ✓
5. Visual Testで最終確認

### デプロイ

学習済みモデルのパスをコピー:
```
models/finetuned/{run_name}/weights/best.pt
```

---

## トラブルシューティング

### ROS2が接続できない

1. ROS2環境をソース:
   ```bash
   source /opt/ros/humble/setup.bash
   ```
2. 撮影ノードを起動:
   ```bash
   ros2 launch hsr_perception capture.launch.py
   ```

### アノテーションが失敗する

1. 画像データが `datasets/raw_captures/` に存在するか確認
2. 背景画像が正しく選択されているか確認
3. `config/object_classes.json` にクラスが定義されているか確認

### 学習が遅い

1. GPUが利用可能か確認（Settingsページ）
2. バッチサイズを下げる
3. Fast Modeを有効にする（テスト用）

### 評価でmAPが低い

1. データ量が十分か確認（クラスあたり50枚以上推奨）
2. アノテーション品質を確認
3. エポック数を増やして再学習

---

## 関連コマンド

### CLIでの操作（GUIの代替）

```bash
# アノテーション
python scripts/annotation/auto_annotate.py \
    --method background \
    --background datasets/backgrounds/bg.jpg \
    --input-dir datasets/raw_captures \
    --output-dir datasets/annotated/session1

# 学習
python scripts/training/quick_finetune.py \
    --dataset datasets/annotated/session1/data.yaml

# 評価
python scripts/evaluation/evaluate_model.py \
    --model models/finetuned/run1/weights/best.pt \
    --dataset datasets/annotated/session1/data.yaml
```

---

## 設定ファイル

詳細は [configuration.md](./configuration.md) を参照してください。
