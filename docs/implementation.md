# 実装リファレンス

このドキュメントでは、HSR Perception Pipelineの各コンポーネントの詳細を説明します。

---

## scripts/

### common/ - 共通ユーティリティ

| ファイル | 説明 |
|---------|------|
| `constants.py` | 共通定数（画像拡張子、競技目標値、デフォルトパラメータ） |
| `image_utils.py` | 画像処理ユーティリティ（マスク変換、BBox描画） |
| `config_utils.py` | 設定管理（Annotator/Training Config、SAM2設定検出） |
| `device_utils.py` | CUDA/GPUデバイス検出 |
| `validation.py` | バリデーション・エラー処理ユーティリティ |

### capture/ - キャプチャツール

| ファイル | 説明 |
|---------|------|
| `burst_capture.py` | ROS2トピックからのバースト撮影 |
| `capture_app.py` | スタンドアロンキャプチャGUI |
| `capture_frame.py` | 単一フレーム撮影ユーティリティ |
| `record_app.py` | 動画録画アプリ |
| `preview_window.py` | プレビューウィンドウ |

### annotation/ - 自動アノテーション

| ファイル | 説明 |
|---------|------|
| `auto_annotate.py` | メインパイプライン（背景差分/SAM2を統合） |
| `annotation_utils.py` | YOLO形式変換、データセット分割ユーティリティ |
| `background_subtraction.py` | 背景差分による自動アノテーション（高速・シンプル） |
| `sam2_annotator.py` | SAM2による自動アノテーション（高精度・GPU必要） |
| `sam2_interactive_app.py` | SAM2対話的アノテーションGUI |
| `video_tracking_predictor.py` | 動画トラッキング予測器 |
| `prepare_dataset.py` | データセット準備スクリプト |
| `colab_utils.py` | Google Colab用ユーティリティ |

### training/ - 学習パイプライン

| ファイル | 説明 |
|---------|------|
| `quick_finetune.py` | 競技用最適化されたYOLOv8 fine-tuningスクリプト |
| `training_config.py` | 学習設定管理 |
| `gpu_scaler.py` | GPUメモリに応じた自動スケーリング |
| `tensorboard_monitor.py` | TensorBoard監視ユーティリティ |

### evaluation/ - 評価ツール

| ファイル | 説明 |
|---------|------|
| `evaluate_model.py` | mAP計算、推論時間測定、競技要件チェック |
| `visual_verification.py` | 対話的な検出結果可視化ツール |
| `xtion_test_app.py` | Xtionカメラでのリアルタイムテスト |

---

## app/ - Streamlit GUIアプリ

### エントリーポイント

| ファイル | 説明 |
|---------|------|
| `main.py` | メインアプリケーション（Dashboard含む） |
| `config.py` | アプリ設定 |
| `object_registry.py` | オブジェクト登録管理 |

### pages/ - ページモジュール

| ファイル | 説明 |
|---------|------|
| `2_Registry.py` | オブジェクト登録ページ |
| `3_Collection.py` | データ収集ページ |
| `4_Annotation.py` | アノテーション実行ページ |
| `5_Training.py` | 学習実行・監視ページ |
| `6_Evaluation.py` | モデル評価ページ |
| `7_Settings.py` | 設定ページ |

### services/ - サービス層

| ファイル | 説明 |
|---------|------|
| `path_coordinator.py` | パス管理・データディレクトリ調整 |
| `task_manager.py` | 長時間タスク（学習・アノテーション）管理 |
| `ros2_bridge.py` | ROS2連携ブリッジ |
| `profile_manager.py` | プロファイル管理 |
| `dataset_preparer.py` | データセット準備サービス |

### services/task_runners/ - タスク実行スクリプト

| ファイル | 説明 |
|---------|------|
| `run_annotation.py` | アノテーションタスク実行 |
| `run_training.py` | 学習タスク実行 |
| `run_evaluation.py` | 評価タスク実行 |

### components/ - 共有UIコンポーネント

| ファイル | 説明 |
|---------|------|
| `progress_display.py` | 進捗表示コンポーネント |
| `dataset_status.py` | データセット状態表示 |
| `config_preview.py` | 設定プレビュー |
| `training_charts.py` | 学習グラフ表示 |
| `training_styles.py` | 学習ページスタイル |
| `tensorboard_embed.py` | TensorBoard埋め込み |
| `robustness_test.py` | ロバストネステスト |
| `robustness_augmentation.py` | ロバストネス拡張 |
| `common_sidebar.py` | 共通サイドバー |

---

## src/hsr_perception/ - ROS2パッケージ

### ノード

| ファイル | 説明 |
|---------|------|
| `hsr_perception/continuous_capture_node.py` | HSRカメラからの連続撮影ROS2ノード |

### サービス定義 (srv/)

| ファイル | 説明 |
|---------|------|
| `SetClass.srv` | クラス選択サービス |
| `StartBurst.srv` | バースト撮影開始サービス |
| `GetStatus.srv` | 撮影状態取得サービス |

### ランチファイル (launch/)

| ファイル | 説明 |
|---------|------|
| `capture.launch.py` | 撮影ノード起動用launchファイル |

---

## config/

| ファイル | 説明 |
|---------|------|
| `object_classes.json` | クラス定義（大会ごとに編集） |
