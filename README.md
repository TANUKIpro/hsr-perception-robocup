# HSR Perception Pipeline for RoboCup@Home

RoboCup@Home（iHR-C3等）大会に向けた、HSR用物体認識パイプラインの開発リポジトリ

## 📋 プロジェクト概要

### 背景
- **使用ロボット**: Toyota HSR (Human Support Robot)
- **対象大会**: RoboCup@Home / Intelligent Home Robotics Challenge
- **開発環境**: Ubuntu 22.04 / ROS2 Humble / Python

### 解決すべき課題
1. **時間制約下での物体認識モデル構築**
   - 大会当日、実物オブジェクト配布から競技開始まで2〜3時間
   - この時間内でYOLO等のfine-tuningを完了させる必要がある

2. **3種類の物体への対応**
   | 物体タイプ | 説明 | 対応方針 |
   |-----------|------|---------|
   | Standard Objects | 数ヶ月前に公開 | 事前に十分な学習 |
   | Consistent Objects | Setup Dayに公開 | 当日の効率的な学習 |
   | Unknown Objects | リストにない物体 | ゼロショット/2段階検出 |

---

## 🎯 目指すべき指標

### 物体認識性能
| 指標 | 目標値 | 備考 |
|------|--------|------|
| Known Objects mAP | ≥ 85% | YOLO fine-tuning |
| Known Objects 推論速度 | ≤ 100ms | リアルタイム性確保 |
| Unknown Objects 検出率 | ≥ 60% | 2段階検出 |
| 誤検出率 (False Positive) | ≤ 10% | 背景誤検出の抑制 |

### 当日オペレーション
| 指標 | 目標値 | 備考 |
|------|--------|------|
| データ収集時間 | ≤ 40分 | 全クラス合計 |
| アノテーション時間 | ≤ 30分 | 半自動化 |
| 学習時間 | ≤ 60分 | GPU使用 |
| 検証・デプロイ時間 | ≤ 20分 | - |
| **合計** | **≤ 2.5時間** | バッファ含む |

---

## 🗓️ マイルストーン

### Phase 1: 基盤構築（〜2週間）
- [ ] 開発環境セットアップ（ROS2 Humble, YOLO, SAM等）
- [ ] HSRカメラからの画像取得ノード作成
- [ ] 基本的な物体検出ノード（YOLO）の実装
- [ ] シミュレーション環境での動作確認

### Phase 2: Known Objects対応（〜4週間）
- [ ] Standard Objectsリスト入手・データセット準備
- [ ] YOLOv8 fine-tuningパイプライン構築
- [ ] データ拡張戦略の検討・実装
- [ ] 精度評価・チューニング

### Phase 3: 当日用パイプライン（〜6週間）
- [ ] 効率的データ収集スクリプト（連続撮影）
- [ ] 半自動アノテーションツール
- [ ] 高速fine-tuningスクリプト
- [ ] 検証・デプロイ自動化

### Phase 4: Unknown Objects対応（〜8週間）
- [ ] クラス非依存検出（SAM）の統合
- [ ] ゼロショット分類（DINOv2/CLIP）の統合
- [ ] ハイブリッド検出パイプライン構築
- [ ] フォールバック（対話確認）の実装

### Phase 5: 統合・実機テスト（〜大会）
- [ ] 全パイプラインの統合
- [ ] HSR実機での動作確認
- [ ] 競技シナリオでのE2Eテスト
- [ ] ドキュメント整備・チームトレーニング

---

## 🔬 技術的検討事項

### 1. 学習手法の選択

#### Fine-tuning（メイン手法）
- **用途**: Known Objects（Standard / Consistent）
- **モデル**: YOLOv8s（精度と速度のバランス）
- **データ量目安**: クラスあたり50〜100枚
- **想定精度**: mAP 85-95%

#### ゼロショット学習（バックアップ）
- **用途**: Unknown Objects、緊急時のフォールバック
- **モデル候補**: OWL-ViT, Grounding DINO
- **特徴**: 事前学習不要、テキストプロンプトで検出
- **想定精度**: mAP 50-70%（物体による）

#### フューショット学習（オプション）
- **用途**: 当日追加クラスの即座登録
- **手法**: DINOv2特徴量 + 最近傍法
- **データ量**: クラスあたり3〜10枚
- **特徴**: 学習時間ゼロ、精度は中程度

### 2. 検出アーキテクチャの比較

#### 1段階検出（YOLO）
```
画像 → YOLO → 既知物体の検出結果
```
- **メリット**: 高精度（+10-15%）、高速、シンプル
- **デメリット**: 未知物体は検出不可

#### 2段階検出（SAM + 分類器）
```
画像 → SAM（物体候補）→ 分類器 → 既知/未知判定
```
- **メリット**: 未知物体も検出可能
- **デメリット**: 精度低下、処理時間増加、誤検出リスク

#### 採用方針: ハイブリッドアプローチ
```
画像 ─┬→ YOLO（既知物体）───────────┬→ 統合結果
      │                             │
      └→ SAM+分類（未知物体）─────────┘
         ※必要な時のみ起動
```

### 3. 当日ワークフロー設計

```
00:00 - 00:10  セットアップ確認、オブジェクト受取
               ↓
              【並列作業開始】
               ├─ 作業者A: データ収集（撮影）
               └─ 作業者B: フューショット用サンプル登録（バックアップ）
               ↓
00:10 - 00:50  Phase 1: データ収集
               ・回転台での多角度撮影
               ・背景バリエーション（白背景 + 実環境）
               ・各クラス50-100枚目標
               ↓
00:50 - 01:20  Phase 2: アノテーション
               ・半自動アノテーション実行
               ・目視確認・修正
               ↓
01:20 - 02:00  Phase 3: Fine-tuning
               ・YOLOv8s, 30エポック
               ・データ拡張有効
               ↓
02:00 - 02:20  Phase 4: 検証・デプロイ
               ・テスト画像での精度確認
               ・HSRへのモデルデプロイ
               ↓
02:20 - 02:30  バッファ・最終確認
```

### 4. Unknown Objects対応戦略

#### 優先度順の対応方法
1. **タスク指示からの推測**
   - 音声指示に含まれる物体名でオープンボキャブラリ検出
   - 例:「スナック菓子を取って」→ "snack", "chips"で検出

2. **クラス非依存検出 + 特徴量比較**
   - SAMで物体候補を検出
   - DINOv2特徴量でKnown Objectsとの類似度を計算
   - 類似度が低い → Unknown Objectと判定

3. **対話による確認（最終手段）**
   - 「この物体は何ですか？」と人間に質問
   - Deus ex Machinaの枠組みで許容（減点あり）

---

## 📁 ディレクトリ構成

```
hsr-perception-robocup/
├── README.md
├── CLAUDE.md                      # Claude Code用ガイド
├── run_app.sh                     # アプリ起動スクリプト
│
├── app/                           # Streamlit GUIアプリ
│   ├── main.py                    # メインエントリーポイント
│   ├── config.py                  # アプリ設定
│   ├── object_registry.py         # オブジェクト管理
│   ├── services/                  # サービス層
│   │   ├── path_coordinator.py    # パス管理
│   │   ├── task_manager.py        # 長時間タスク管理
│   │   ├── ros2_bridge.py         # ROS2連携
│   │   └── task_runners/          # タスク実行スクリプト
│   ├── pages/                     # ページモジュール
│   │   ├── 4_Annotation.py        # アノテーションページ
│   │   ├── 5_Training.py          # 学習ページ
│   │   └── 6_Evaluation.py        # 評価ページ
│   ├── components/                # 共有UIコンポーネント
│   │   └── progress_display.py    # 進捗表示
│   └── data/                      # アプリデータ
│       ├── collected_images/      # 収集画像
│       ├── reference_images/      # 参照画像
│       ├── object_registry.json   # オブジェクト登録情報
│       └── tasks/                 # タスク状態ファイル
│
├── docs/                          # ドキュメント
│   ├── app_guide.md               # アプリガイド（日本語）
│   ├── app_guide_en.md            # アプリガイド（英語）
│   └── configuration.md           # 設定リファレンス
│
├── config/                        # 設定ファイル
│   └── object_classes.json        # クラス定義（メイン設定）
│
├── datasets/                      # データセット（.gitignore）
│   ├── raw_captures/              # 生画像（クラス別）
│   ├── annotated/                 # アノテーション済みデータセット
│   └── backgrounds/               # 背景画像
│
├── models/                        # 学習済みモデル（.gitignore）
│   ├── pretrained/                # 事前学習済みモデル
│   └── finetuned/                 # Fine-tuned モデル
│
├── scripts/                       # パイプラインスクリプト
│   ├── annotation/                # アノテーション
│   │   ├── auto_annotate.py       # メインパイプライン
│   │   ├── annotation_utils.py    # ユーティリティ
│   │   ├── background_subtraction.py  # 背景差分
│   │   └── sam2_annotator.py      # SAM2アノテーション
│   ├── training/                  # 学習
│   │   └── quick_finetune.py      # 競技用fine-tuning
│   └── evaluation/                # 評価
│       ├── evaluate_model.py      # モデル評価
│       └── visual_verification.py # 可視化検証
│
├── src/                           # ROS2パッケージソース
│   └── hsr_perception/            # メインパッケージ
│       ├── hsr_perception/
│       │   └── continuous_capture_node.py  # 連続撮影ノード
│       ├── srv/                   # サービス定義
│       │   ├── SetClass.srv
│       │   ├── StartBurst.srv
│       │   └── GetStatus.srv
│       └── launch/
│           └── capture.launch.py  # 撮影ノード起動
│
├── tests/                         # テスト
├── notebooks/                     # 実験・分析用Notebook
├── docker/                        # Docker環境
└── requirements.txt               # Python依存パッケージ
```

---

## 🛠️ 技術スタック

### 物体検出
- **YOLOv8** (Ultralytics): Known Objects検出のメイン
- **Segment Anything (SAM)**: クラス非依存物体候補検出
- **OWL-ViT / Grounding DINO**: ゼロショット検出

### 特徴抽出・分類
- **DINOv2**: 汎用特徴抽出（フューショット用）
- **CLIP**: テキスト-画像対応（オープンボキャブラリ用）

### ロボティクス
- **ROS2 Humble**: ミドルウェア
- **cv_bridge**: ROS-OpenCV変換
- **vision_msgs**: 検出結果メッセージ

### 開発ツール
- **Docker**: 環境の再現性確保
- **CVAT / Label Studio**: アノテーション（オプション）

---

## 📊 評価計画

### オフライン評価
- [ ] Standard Objectsでのmeasurement
- [ ] 未知物体を混ぜたテストセットでの評価
- [ ] 処理速度ベンチマーク

### オンライン評価（シミュレーション）
- [ ] Gazebo上でのE2Eテスト
- [ ] 把持タスクとの統合テスト

### 実機評価
- [ ] HSR実機での認識精度
- [ ] 照明条件変化への耐性
- [ ] 競技環境模擬テスト

---

## 📚 参考資料

### ルール・レギュレーション
- [RoboCup@Home Rulebook](https://github.com/RoboCupAtHome/RuleBook)
- [iHR-C3 General Rules and Regulations](./docs/grr_en.md)

### 技術資料
- [HSR ROS2 Documentation](https://github.com/hsr-project/hsr_ros2_doc)
- [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/)
- [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything)

---

## 👥 コントリビューション

### ブランチ戦略
- `main`: 安定版
- `develop`: 開発版
- `feature/*`: 機能開発
- `competition/*`: 大会固有の調整

### コミットメッセージ規約
```
feat: 新機能追加
fix: バグ修正
docs: ドキュメント
refactor: リファクタリング
test: テスト追加・修正
```

---

## 🚀 クイックスタート

### 環境構築

```bash
# Python依存パッケージのインストール
pip install -r requirements.txt

# ROS2パッケージのビルド（ROS2 Humbleが必要）
cd src
colcon build --packages-select hsr_perception
source install/setup.bash
```

### Object Manager アプリ（GUI）

物体認識パイプライン全体をGUIで操作できるStreamlitベースのWebアプリケーション。
データ収集からモデル評価まで、競技当日の全工程をサポートします。

```bash
# 起動
./run_app.sh
# または
streamlit run app/main.py
```

**機能:**
- 📊 **Dashboard**: パイプライン全体の進捗・状態可視化
- 📋 **Registry**: オブジェクト登録・参照画像管理
- 📸 **Collection**: ROS2カメラ/ローカルカメラ/ファイルアップロードでデータ収集
- 🏷️ **Annotation**: 自動アノテーションパイプライン実行（背景差分/SAM2）
- 🎓 **Training**: YOLOv8 fine-tuning実行・進捗監視
- 📈 **Evaluation**: モデル評価・競技要件チェック・可視化テスト
- ⚙️ **Settings**: システム状態確認・データ同期

**ドキュメント:**
- [アプリガイド（日本語）](docs/app_guide.md)
- [App Guide (English)](docs/app_guide_en.md)
- [設定リファレンス](docs/configuration.md)

ブラウザで http://localhost:8501 を開くとアプリが表示されます。

### 大会当日ワークフロー

#### Step 1: クラス設定の更新
`config/object_classes.json` を大会で配布されたオブジェクトリストに合わせて更新

#### Step 2: データ収集（ROS2ノード使用）
```bash
# 連続撮影ノードの起動
ros2 launch hsr_perception capture.launch.py

# 別ターミナルでクラス設定・バースト撮影
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
# 競技用設定でYOLOv8mをfine-tuning（約45分）
python scripts/training/quick_finetune.py \
    --dataset datasets/competition_day/data.yaml \
    --model yolov8m.pt \
    --output models/finetuned

# 高速モード（小さいモデル、約20分）
python scripts/training/quick_finetune.py \
    --dataset datasets/competition_day/data.yaml \
    --fast
```

#### Step 5: 評価・検証
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

## 📦 実装済みコンポーネント

### 自動アノテーション (`scripts/annotation/`)
| ファイル | 説明 |
|---------|------|
| `annotation_utils.py` | YOLO形式変換、データセット分割ユーティリティ |
| `background_subtraction.py` | 背景差分による自動アノテーション（高速・シンプル） |
| `sam2_annotator.py` | SAM2による自動アノテーション（高精度・GPU必要） |
| `auto_annotate.py` | メインパイプライン（上記手法を統合） |

### 学習パイプライン (`scripts/training/`)
| ファイル | 説明 |
|---------|------|
| `quick_finetune.py` | 競技用最適化されたYOLOv8 fine-tuningスクリプト |

### 評価ツール (`scripts/evaluation/`)
| ファイル | 説明 |
|---------|------|
| `evaluate_model.py` | mAP計算、推論時間測定、競技要件チェック |
| `visual_verification.py` | 対話的な検出結果可視化ツール |

### ROS2パッケージ (`src/hsr_perception/`)
| コンポーネント | 説明 |
|--------------|------|
| `continuous_capture_node.py` | HSRカメラからの連続撮影ROS2ノード |
| `srv/SetClass.srv` | クラス選択サービス |
| `srv/StartBurst.srv` | バースト撮影開始サービス |
| `srv/GetStatus.srv` | 撮影状態取得サービス |
| `launch/capture.launch.py` | 撮影ノード起動用launchファイル |

---

## 📝 TODO（直近）

- [x] 自動アノテーションパイプライン（背景差分 + SAM2）
- [x] 連続撮影ROS2ノード
- [x] YOLOv8 fine-tuningスクリプト
- [x] モデル評価・可視化ツール
- [ ] 開発環境のDockerfile作成
- [ ] HSR実機での動作確認
- [ ] Standard Objectsリストの入手・事前学習

---

## ライセンス

TBD（チームで決定）