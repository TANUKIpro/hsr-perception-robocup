# フロントエンドテスト項目一覧

このドキュメントは、フロントエンド（`app/`）のテスト項目を網羅的に列挙しています。
別セッションでテストを実装する際の参照資料として使用してください。

## 凡例

- ✅ 実装済み
- ⬜ 未実装
- 🔶 優先度: 高
- 🔷 優先度: 中
- ⬜ 優先度: 低

---

## 1. Services モジュール (`app/services/`)

### 1.1 test_profile_manager.py ⬜

**ソースファイル**: `app/services/profile_manager.py`

**モック要件**: ファイルシステム、`zipfile`

**優先度**: 🔶 高

| 状態 | テストクラス | テストメソッド | 説明 |
|------|-------------|---------------|------|
| ⬜ | `TestProfileMetadata` | `test_to_dict()` | 辞書変換 |
| ⬜ | `TestProfileMetadata` | `test_from_dict()` | 辞書から作成 |
| ⬜ | `TestProfileMetadata` | `test_default_timestamps()` | デフォルトタイムスタンプ |
| ⬜ | `TestProfileManager` | `test_initialization()` | 初期化 |
| ⬜ | `TestProfileManager` | `test_initialization_creates_default_profile()` | デフォルトプロファイル作成 |
| ⬜ | `TestProfileManager` | `test_create_profile()` | プロファイル作成 |
| ⬜ | `TestProfileManager` | `test_create_profile_with_custom_id()` | カスタムID付きプロファイル作成 |
| ⬜ | `TestProfileManager` | `test_get_profile()` | プロファイル取得 |
| ⬜ | `TestProfileManager` | `test_get_profile_not_found()` | プロファイル未発見 |
| ⬜ | `TestProfileManager` | `test_get_all_profiles()` | 全プロファイル取得 |
| ⬜ | `TestProfileManager` | `test_set_active_profile()` | アクティブプロファイル設定 |
| ⬜ | `TestProfileManager` | `test_set_active_profile_not_found()` | 存在しないプロファイルをアクティブに |
| ⬜ | `TestProfileManager` | `test_get_active_profile_id()` | アクティブプロファイルID取得 |
| ⬜ | `TestProfileManager` | `test_update_profile()` | プロファイル更新 |
| ⬜ | `TestProfileManager` | `test_update_profile_name()` | プロファイル名更新 |
| ⬜ | `TestProfileManager` | `test_delete_profile()` | プロファイル削除 |
| ⬜ | `TestProfileManager` | `test_cannot_delete_last_profile()` | 最後のプロファイル削除不可 |
| ⬜ | `TestProfileManager` | `test_cannot_delete_active_profile()` | アクティブプロファイル削除不可 |
| ⬜ | `TestProfileManager` | `test_duplicate_profile()` | プロファイル複製 |
| ⬜ | `TestProfileManager` | `test_generate_profile_id()` | プロファイルID生成 |
| ⬜ | `TestProfileManager` | `test_get_profile_path()` | プロファイルパス取得 |
| ⬜ | `TestProfileExportImport` | `test_export_profile()` | プロファイルエクスポート |
| ⬜ | `TestProfileExportImport` | `test_export_profile_to_bytes()` | バイトへエクスポート |
| ⬜ | `TestProfileExportImport` | `test_import_profile()` | プロファイルインポート |
| ⬜ | `TestProfileExportImport` | `test_safe_extract_zip_prevents_path_traversal()` | パストラバーサル防止 |
| ⬜ | `TestProfileExportImport` | `test_resolve_duplicate_name()` | 重複名解決 |
| ⬜ | `TestProfileExportImport` | `test_import_invalid_zip()` | 無効なZIPインポート |
| ⬜ | `TestProfileExportImport` | `test_import_too_large_zip()` | 大きすぎるZIPインポート |

---

### 1.2 test_task_manager.py ⬜

**ソースファイル**: `app/services/task_manager.py`

**モック要件**: `subprocess`, ファイルシステム

**優先度**: 🔶 高

| 状態 | テストクラス | テストメソッド | 説明 |
|------|-------------|---------------|------|
| ⬜ | `TestTaskStatus` | `test_enum_values()` | Enum値の確認 |
| ⬜ | `TestTaskInfo` | `test_to_dict()` | 辞書変換 |
| ⬜ | `TestTaskInfo` | `test_from_dict()` | 辞書から作成 |
| ⬜ | `TestTaskInfo` | `test_is_active_property()` | アクティブ状態プロパティ |
| ⬜ | `TestTaskInfo` | `test_is_finished_property()` | 完了状態プロパティ |
| ⬜ | `TestTaskInfo` | `test_elapsed_time_calculation()` | 経過時間計算 |
| ⬜ | `TestTaskInfo` | `test_elapsed_time_str_format()` | 経過時間文字列フォーマット |
| ⬜ | `TestTaskManager` | `test_initialization()` | 初期化 |
| ⬜ | `TestTaskManager` | `test_generate_task_id()` | タスクID生成 |
| ⬜ | `TestTaskManager` | `test_save_and_load_status()` | ステータス保存と読み込み |
| ⬜ | `TestTaskManager` | `test_get_task()` | タスク取得 |
| ⬜ | `TestTaskManager` | `test_get_task_not_found()` | タスク未発見 |
| ⬜ | `TestTaskManager` | `test_get_all_tasks()` | 全タスク取得 |
| ⬜ | `TestTaskManager` | `test_get_active_tasks()` | アクティブタスク取得 |
| ⬜ | `TestTaskManager` | `test_get_recent_tasks()` | 最近のタスク取得 |
| ⬜ | `TestTaskManager` | `test_cancel_task()` | タスクキャンセル |
| ⬜ | `TestTaskManager` | `test_cancel_nonexistent_task()` | 存在しないタスクキャンセル |
| ⬜ | `TestTaskManager` | `test_delete_task()` | タスク削除 |
| ⬜ | `TestTaskManager` | `test_cleanup_old_tasks()` | 古いタスクのクリーンアップ |
| ⬜ | `TestTaskLaunchers` | `test_start_annotation()` | アノテーション開始 |
| ⬜ | `TestTaskLaunchers` | `test_start_annotation_requires_background()` | アノテーションには背景が必要 |
| ⬜ | `TestTaskLaunchers` | `test_start_training()` | 訓練開始 |
| ⬜ | `TestTaskLaunchers` | `test_start_training_requires_dataset()` | 訓練にはデータセットが必要 |
| ⬜ | `TestTaskLaunchers` | `test_start_evaluation()` | 評価開始 |
| ⬜ | `TestUpdateTaskStatus` | `test_update_progress()` | 進捗更新 |
| ⬜ | `TestUpdateTaskStatus` | `test_update_status()` | ステータス更新 |
| ⬜ | `TestUpdateTaskStatus` | `test_update_error_message()` | エラーメッセージ更新 |

---

### 1.3 test_path_coordinator.py ⬜

**ソースファイル**: `app/services/path_coordinator.py`

**モック要件**: ファイルシステム、`streamlit.session_state`

**優先度**: 🔶 高

| 状態 | テストクラス | テストメソッド | 説明 |
|------|-------------|---------------|------|
| ⬜ | `TestPathConfig` | `test_default_paths()` | デフォルトパス |
| ⬜ | `TestPathConfig` | `test_custom_paths()` | カスタムパス |
| ⬜ | `TestPathCoordinator` | `test_initialization()` | 初期化 |
| ⬜ | `TestPathCoordinator` | `test_initialization_with_profile_manager()` | ProfileManager付き初期化 |
| ⬜ | `TestPathCoordinator` | `test_get_path_profile_specific()` | プロファイル固有パス取得 |
| ⬜ | `TestPathCoordinator` | `test_get_path_shared()` | 共有パス取得 |
| ⬜ | `TestPathCoordinator` | `test_resolve_path_absolute()` | 絶対パス解決 |
| ⬜ | `TestPathCoordinator` | `test_resolve_path_relative()` | 相対パス解決 |
| ⬜ | `TestPathCoordinator` | `test_create_annotation_session()` | アノテーションセッション作成 |
| ⬜ | `TestPathCoordinator` | `test_get_annotation_sessions()` | アノテーションセッション取得 |
| ⬜ | `TestPathCoordinator` | `test_get_training_paths()` | 訓練パス取得 |
| ⬜ | `TestPathCoordinator` | `test_get_trained_models()` | 訓練済みモデル取得 |
| ⬜ | `TestPathCoordinator` | `test_get_pretrained_models()` | 事前訓練モデル取得 |
| ⬜ | `TestPathCoordinator` | `test_get_background_images()` | 背景画像取得 |
| ⬜ | `TestPathCoordinator` | `test_add_background_image()` | 背景画像追加 |
| ⬜ | `TestPathCoordinator` | `test_validate_paths()` | パス検証 |
| ⬜ | `TestCachedFunctions` | `test_cached_get_annotation_sessions()` | キャッシュ付きセッション取得 |
| ⬜ | `TestCachedFunctions` | `test_cached_get_trained_models()` | キャッシュ付きモデル取得 |
| ⬜ | `TestCachedFunctions` | `test_cached_get_background_images()` | キャッシュ付き背景取得 |
| ⬜ | `TestCachedFunctions` | `test_cache_invalidation()` | キャッシュ無効化 |

---

### 1.4 test_ui_settings_manager.py ⬜

**ソースファイル**: `app/services/ui_settings_manager.py`

**モック要件**: ファイルシステム、`streamlit.session_state`

**優先度**: 🔷 中

| 状態 | テストクラス | テストメソッド | 説明 |
|------|-------------|---------------|------|
| ⬜ | `TestUISettingsManager` | `test_initialization()` | 初期化 |
| ⬜ | `TestUISettingsManager` | `test_load_settings()` | 設定読み込み |
| ⬜ | `TestUISettingsManager` | `test_load_settings_file_not_found()` | 設定ファイル未発見 |
| ⬜ | `TestUISettingsManager` | `test_save_settings()` | 設定保存 |
| ⬜ | `TestUISettingsManager` | `test_get_setting()` | 設定取得 |
| ⬜ | `TestUISettingsManager` | `test_get_setting_with_default()` | デフォルト付き設定取得 |
| ⬜ | `TestUISettingsManager` | `test_set_setting()` | 設定設定 |
| ⬜ | `TestUISettingsManager` | `test_delete_setting()` | 設定削除 |
| ⬜ | `TestUISettingsManager` | `test_nested_settings()` | ネストした設定 |
| ⬜ | `TestUISettingsManager` | `test_settings_persistence()` | 設定の永続化 |

---

### 1.5 test_dataset_preparer.py ⬜

**ソースファイル**: `app/services/dataset_preparer.py`

**モック要件**: ファイルシステム、`yaml`

**優先度**: 🔷 中

| 状態 | テストクラス | テストメソッド | 説明 |
|------|-------------|---------------|------|
| ⬜ | `TestDatasetPreparer` | `test_initialization()` | 初期化 |
| ⬜ | `TestDatasetPreparer` | `test_prepare_dataset()` | データセット準備 |
| ⬜ | `TestDatasetPreparer` | `test_prepare_dataset_with_split()` | 分割付きデータセット準備 |
| ⬜ | `TestDatasetPreparer` | `test_validate_annotations()` | アノテーション検証 |
| ⬜ | `TestDatasetPreparer` | `test_create_yaml()` | YAML作成 |
| ⬜ | `TestDatasetPreparer` | `test_copy_images()` | 画像コピー |
| ⬜ | `TestDatasetPreparer` | `test_symlink_images()` | 画像シンボリックリンク |
| ⬜ | `TestDatasetPreparer` | `test_get_dataset_stats()` | データセット統計取得 |

---

## 2. Core モジュール (`app/`)

### 2.1 test_object_registry.py ⬜

**ソースファイル**: `app/object_registry.py`

**モック要件**: ファイルシステム、`cv2`（サムネイル）

**優先度**: 🔶 高

| 状態 | テストクラス | テストメソッド | 説明 |
|------|-------------|---------------|------|
| ⬜ | `TestObjectVersion` | `test_default_values()` | デフォルト値 |
| ⬜ | `TestObjectVersion` | `test_custom_values()` | カスタム値 |
| ⬜ | `TestObjectProperties` | `test_default_values()` | デフォルト値 |
| ⬜ | `TestObjectProperties` | `test_to_dict()` | 辞書変換 |
| ⬜ | `TestObjectProperties` | `test_from_dict()` | 辞書から作成 |
| ⬜ | `TestRegisteredObject` | `test_to_dict()` | 辞書変換 |
| ⬜ | `TestRegisteredObject` | `test_from_dict()` | 辞書から作成 |
| ⬜ | `TestRegisteredObject` | `test_roundtrip_serialization()` | 往復シリアライズ |
| ⬜ | `TestObjectRegistry` | `test_initialization()` | 初期化 |
| ⬜ | `TestObjectRegistry` | `test_initialization_creates_file()` | 初期化でファイル作成 |
| ⬜ | `TestObjectRegistry` | `test_add_object()` | オブジェクト追加 |
| ⬜ | `TestObjectRegistry` | `test_add_object_duplicate_name()` | 重複名オブジェクト追加 |
| ⬜ | `TestObjectRegistry` | `test_remove_object()` | オブジェクト削除 |
| ⬜ | `TestObjectRegistry` | `test_get_object()` | オブジェクト取得 |
| ⬜ | `TestObjectRegistry` | `test_get_object_not_found()` | オブジェクト未発見 |
| ⬜ | `TestObjectRegistry` | `test_get_object_by_name()` | 名前でオブジェクト取得 |
| ⬜ | `TestObjectRegistry` | `test_get_all_objects()` | 全オブジェクト取得 |
| ⬜ | `TestObjectRegistry` | `test_get_objects_by_category()` | カテゴリでオブジェクト取得 |
| ⬜ | `TestObjectRegistry` | `test_get_next_id()` | 次のID取得 |
| ⬜ | `TestObjectRegistry` | `test_add_category()` | カテゴリ追加 |
| ⬜ | `TestObjectRegistry` | `test_add_duplicate_category()` | 重複カテゴリ追加 |
| ⬜ | `TestObjectRegistry` | `test_update_object()` | オブジェクト更新 |
| ⬜ | `TestObjectRegistry` | `test_update_object_name_renames_directories()` | 名前変更時ディレクトリリネーム |
| ⬜ | `TestThumbnailManagement` | `test_set_thumbnail()` | サムネイル設定 |
| ⬜ | `TestThumbnailManagement` | `test_save_thumbnail_from_bytes()` | バイトからサムネイル保存 |
| ⬜ | `TestThumbnailManagement` | `test_get_thumbnail_path()` | サムネイルパス取得 |
| ⬜ | `TestThumbnailManagement` | `test_get_thumbnail_path_not_found()` | サムネイルパス未発見 |
| ⬜ | `TestReferenceImageManagement` | `test_add_reference_image()` | 参照画像追加 |
| ⬜ | `TestReferenceImageManagement` | `test_get_reference_images()` | 参照画像取得 |
| ⬜ | `TestReferenceImageManagement` | `test_delete_reference_image()` | 参照画像削除 |
| ⬜ | `TestCollectionManagement` | `test_add_collected_image()` | 収集画像追加 |
| ⬜ | `TestCollectionManagement` | `test_save_collected_image()` | 収集画像保存 |
| ⬜ | `TestCollectionManagement` | `test_get_collected_images()` | 収集画像取得 |
| ⬜ | `TestCollectionManagement` | `test_update_collection_count()` | 収集カウント更新 |
| ⬜ | `TestCollectionManagement` | `test_update_all_collection_counts()` | 全収集カウント更新 |
| ⬜ | `TestStatistics` | `test_get_collection_stats()` | 収集統計取得 |
| ⬜ | `TestStatistics` | `test_get_category_progress()` | カテゴリ進捗取得 |
| ⬜ | `TestExport` | `test_export_to_yolo_config()` | YOLO設定エクスポート |

---

### 2.2 test_config.py ⬜

**ソースファイル**: `app/config.py`

**モック要件**: 環境変数、`torch`、`rclpy`

**優先度**: 🔷 中

| 状態 | テストクラス | テストメソッド | 説明 |
|------|-------------|---------------|------|
| ⬜ | `TestAppConfig` | `test_default_values()` | デフォルト値 |
| ⬜ | `TestAppConfig` | `test_environment_detection()` | 環境検出 |
| ⬜ | `TestAppConfig` | `test_docker_environment()` | Docker環境 |
| ⬜ | `TestAppConfig` | `test_property_paths()` | プロパティパス |
| ⬜ | `TestAppConfig` | `test_check_ros2_available()` | ROS2利用可能チェック |
| ⬜ | `TestAppConfig` | `test_check_ros2_not_available()` | ROS2利用不可チェック |
| ⬜ | `TestAppConfig` | `test_check_gpu_available()` | GPU利用可能チェック |
| ⬜ | `TestAppConfig` | `test_check_gpu_not_available()` | GPU利用不可チェック |
| ⬜ | `TestAppConfig` | `test_to_dict()` | 辞書変換 |
| ⬜ | `TestGetConfig` | `test_singleton_pattern()` | シングルトンパターン |
| ⬜ | `TestGetConfig` | `test_reload_config()` | 設定リロード |

---

## 3. Components モジュール (`app/components/`)

### 3.1 test_training_charts.py ⬜

**ソースファイル**: `app/components/training_charts.py`

**モック要件**: `plotly`

**優先度**: 🔷 中

| 状態 | テストクラス | テストメソッド | 説明 |
|------|-------------|---------------|------|
| ⬜ | `TestTrainingCharts` | `test_create_loss_chart()` | ロスチャート作成 |
| ⬜ | `TestTrainingCharts` | `test_create_metrics_chart()` | メトリクスチャート作成 |
| ⬜ | `TestTrainingCharts` | `test_create_lr_chart()` | 学習率チャート作成 |
| ⬜ | `TestTrainingCharts` | `test_parse_training_log()` | 訓練ログパース |
| ⬜ | `TestTrainingCharts` | `test_empty_data_handling()` | 空データ処理 |

---

### 3.2 test_progress_display.py ⬜

**ソースファイル**: `app/components/progress_display.py`

**モック要件**: `streamlit`

**優先度**: 🔷 中

| 状態 | テストクラス | テストメソッド | 説明 |
|------|-------------|---------------|------|
| ⬜ | `TestProgressDisplay` | `test_render_progress()` | 進捗レンダリング |
| ⬜ | `TestProgressDisplay` | `test_calculate_percentage()` | パーセンテージ計算 |
| ⬜ | `TestProgressDisplay` | `test_format_elapsed_time()` | 経過時間フォーマット |
| ⬜ | `TestProgressDisplay` | `test_estimate_remaining_time()` | 残り時間推定 |

---

### 3.3 test_captured_images_tree.py ⬜

**ソースファイル**: `app/components/captured_images_tree.py`

**モック要件**: ファイルシステム、`streamlit`

**優先度**: 🔷 中

| 状態 | テストクラス | テストメソッド | 説明 |
|------|-------------|---------------|------|
| ⬜ | `TestCapturedImagesTree` | `test_build_tree_structure()` | ツリー構造構築 |
| ⬜ | `TestCapturedImagesTree` | `test_count_images_per_class()` | クラス別画像数カウント |
| ⬜ | `TestCapturedImagesTree` | `test_empty_directory()` | 空ディレクトリ |
| ⬜ | `TestCapturedImagesTree` | `test_nested_directories()` | ネストディレクトリ |

---

### 3.4 test_robustness_test.py ⬜

**ソースファイル**: `app/components/robustness_test.py`

**モック要件**: `cv2`, `numpy`, モデル

**優先度**: ⬜ 低

| 状態 | テストクラス | テストメソッド | 説明 |
|------|-------------|---------------|------|
| ⬜ | `TestRobustnessTest` | `test_apply_brightness_augmentation()` | 明るさ拡張適用 |
| ⬜ | `TestRobustnessTest` | `test_apply_noise_augmentation()` | ノイズ拡張適用 |
| ⬜ | `TestRobustnessTest` | `test_apply_blur_augmentation()` | ブラー拡張適用 |
| ⬜ | `TestRobustnessTest` | `test_run_robustness_test()` | ロバスト性テスト実行 |
| ⬜ | `TestRobustnessTest` | `test_calculate_robustness_score()` | ロバスト性スコア計算 |

---

## 4. Pages モジュール (`app/pages/`)

### 4.1 test_dashboard_integration.py ⬜

**ソースファイル**: `app/pages/1_Dashboard.py`

**モック要件**: `streamlit`, サービス層

**優先度**: ⬜ 低

| 状態 | テストクラス | テストメソッド | 説明 |
|------|-------------|---------------|------|
| ⬜ | `TestDashboard` | `test_load_collection_stats()` | 収集統計読み込み |
| ⬜ | `TestDashboard` | `test_calculate_pipeline_status()` | パイプラインステータス計算 |
| ⬜ | `TestDashboard` | `test_category_progress_display()` | カテゴリ進捗表示 |
| ⬜ | `TestDashboard` | `test_training_readiness_check()` | 訓練準備チェック |
| ⬜ | `TestDashboard` | `test_active_task_display()` | アクティブタスク表示 |

---

### 4.2 test_registry_integration.py ⬜

**ソースファイル**: `app/pages/2_Registry.py`

**モック要件**: `streamlit`, `ObjectRegistry`

**優先度**: ⬜ 低

| 状態 | テストクラス | テストメソッド | 説明 |
|------|-------------|---------------|------|
| ⬜ | `TestRegistry` | `test_view_objects()` | オブジェクト表示 |
| ⬜ | `TestRegistry` | `test_add_object_form()` | オブジェクト追加フォーム |
| ⬜ | `TestRegistry` | `test_edit_object()` | オブジェクト編集 |
| ⬜ | `TestRegistry` | `test_delete_object()` | オブジェクト削除 |
| ⬜ | `TestRegistry` | `test_filter_by_category()` | カテゴリフィルタ |
| ⬜ | `TestRegistry` | `test_search_objects()` | オブジェクト検索 |

---

### 4.3 test_training_integration.py ⬜

**ソースファイル**: `app/pages/5_Training.py`

**モック要件**: `streamlit`, `TaskManager`, GPU

**優先度**: ⬜ 低

| 状態 | テストクラス | テストメソッド | 説明 |
|------|-------------|---------------|------|
| ⬜ | `TestTrainingPage` | `test_load_datasets()` | データセット読み込み |
| ⬜ | `TestTrainingPage` | `test_load_models()` | モデル読み込み |
| ⬜ | `TestTrainingPage` | `test_advanced_params_form()` | 詳細パラメータフォーム |
| ⬜ | `TestTrainingPage` | `test_start_training()` | 訓練開始 |
| ⬜ | `TestTrainingPage` | `test_monitor_training_progress()` | 訓練進捗監視 |
| ⬜ | `TestTrainingPage` | `test_cancel_training()` | 訓練キャンセル |
| ⬜ | `TestTrainingPage` | `test_gpu_detection_display()` | GPU検出表示 |
| ⬜ | `TestTrainingPage` | `test_tensorboard_embed()` | TensorBoard埋め込み |

---

## テスト実行方法

```bash
# 全フロントエンドテスト実行
pytest tests/frontend/ -v

# 特定モジュールのテスト
pytest tests/frontend/services/ -v
pytest tests/frontend/components/ -v
pytest tests/frontend/pages/ -v

# カバレッジレポート付き
pytest tests/frontend/ --cov=app --cov-report=html

# Streamlit関連テストをスキップ
pytest tests/frontend/ -v -m "not streamlit"
```

---

## 統計

| カテゴリ | 実装済み | 未実装 | 合計 |
|---------|---------|--------|------|
| Services | 0 | 5 | 5 |
| Core | 0 | 2 | 2 |
| Components | 0 | 4 | 4 |
| Pages | 0 | 3 | 3 |
| **合計** | **0** | **14** | **14** |

---

## 優先度別実装順序

### Phase 1 (高優先度)
1. `test_profile_manager.py` - プロファイル管理の基盤
2. `test_task_manager.py` - タスク管理の基盤
3. `test_path_coordinator.py` - パス管理の基盤
4. `test_object_registry.py` - オブジェクト管理の基盤

### Phase 2 (中優先度)
1. `test_ui_settings_manager.py`
2. `test_dataset_preparer.py`
3. `test_config.py`
4. `test_training_charts.py`
5. `test_progress_display.py`
6. `test_captured_images_tree.py`

### Phase 3 (低優先度)
1. `test_robustness_test.py`
2. `test_dashboard_integration.py`
3. `test_registry_integration.py`
4. `test_training_integration.py`

---

## Streamlitテストの注意点

### モック戦略

Streamlitのテストでは以下のモック戦略を推奨します：

```python
# tests/frontend/conftest.py のフィクスチャを活用
@pytest.fixture
def mock_streamlit():
    """Streamlitモジュールのモック"""
    st_mock = MagicMock()
    st_mock.session_state = {}
    st_mock.cache_data = lambda **kwargs: lambda f: f
    # ... 詳細は conftest.py 参照
```

### セッション状態のテスト

```python
def test_session_state_initialization(mock_streamlit):
    """セッション状態の初期化テスト"""
    # セッション状態をシミュレート
    mock_streamlit.session_state = {
        'profile_id': 'prof_1',
        'active_task': None,
    }
    # テスト実行
    # ...
```

### ページテストの注意

ページテスト（`app/pages/`）は統合テストとして扱い、
以下の点に注意してください：

1. **サービス層のモック**: 実際のファイルI/Oを避ける
2. **セッション状態のセットアップ**: 適切な初期状態を設定
3. **UI要素の検証**: レンダリングではなくロジックを検証

---

## テストデータの準備

### プロファイルディレクトリ構造

```python
@pytest.fixture
def temp_profile_dir(tmp_path):
    """テスト用プロファイルディレクトリ"""
    profile_dir = tmp_path / "profiles" / "prof_1"
    subdirs = [
        "app_data",
        "datasets",
        "models/trained",
        "models/pretrained",
        "raw_captures",
        "backgrounds",
        "annotation_sessions",
    ]
    for subdir in subdirs:
        (profile_dir / subdir).mkdir(parents=True)
    return profile_dir
```

### オブジェクトレジストリデータ

```python
@pytest.fixture
def sample_registry_data():
    """テスト用オブジェクトレジストリ"""
    return {
        "categories": ["food", "container"],
        "objects": [
            {"id": 0, "name": "apple", "category": "food"},
            {"id": 1, "name": "cup", "category": "container"},
        ]
    }
```

---

## 5. E2Eテスト（Playwright）

E2EテストはPlaywright（TypeScript）で実装され、`tests/e2e/`に配置されています。

### 5.1 テスト構成

```
tests/e2e/
├── playwright.config.ts    # Playwright設定
├── package.json            # Node.js依存
├── utils/
│   ├── streamlit-selectors.ts  # Streamlit用セレクタ
│   └── wait-helpers.ts         # 待機ユーティリティ
├── page-objects/           # Page Objectパターン
│   ├── base.page.ts
│   ├── sidebar.component.ts
│   ├── dashboard.page.ts
│   ├── registry.page.ts
│   ├── collection.page.ts
│   ├── annotation.page.ts
│   ├── training.page.ts
│   ├── evaluation.page.ts
│   └── settings.page.ts
└── specs/
    ├── smoke/              # 起動確認テスト
    │   ├── app-launch.spec.ts
    │   └── navigation.spec.ts
    └── pages/              # ページ別テスト
        ├── dashboard.spec.ts
        ├── registry.spec.ts
        ├── collection.spec.ts
        ├── annotation.spec.ts
        ├── training.spec.ts
        ├── evaluation.spec.ts
        └── settings.spec.ts
```

### 5.2 Smokeテスト

| テストファイル | テストケース | 説明 |
|--------------|-------------|------|
| app-launch.spec.ts | `should load the home page successfully` | ホームページ読み込み |
| app-launch.spec.ts | `should display the application title` | タイトル表示 |
| app-launch.spec.ts | `should display the sidebar` | サイドバー表示 |
| app-launch.spec.ts | `should display navigation links` | ナビゲーションリンク |
| app-launch.spec.ts | `should display profile selector` | プロファイル選択 |
| app-launch.spec.ts | `should have no console errors` | コンソールエラーなし |
| app-launch.spec.ts | `should respond within acceptable time` | 応答時間確認 |
| navigation.spec.ts | `should navigate to Dashboard page` | Dashboard遷移 |
| navigation.spec.ts | `should navigate to Registry page` | Registry遷移 |
| navigation.spec.ts | `should navigate to Collection page` | Collection遷移 |
| navigation.spec.ts | `should navigate to Annotation page` | Annotation遷移 |
| navigation.spec.ts | `should navigate to Training page` | Training遷移 |
| navigation.spec.ts | `should navigate to Evaluation page` | Evaluation遷移 |
| navigation.spec.ts | `should navigate to Settings page` | Settings遷移 |
| navigation.spec.ts | `should navigate between pages using sidebar` | サイドバーナビゲーション |
| navigation.spec.ts | `should handle browser back/forward` | ブラウザ履歴操作 |

### 5.3 ページ別テスト概要

| ページ | テスト数 | 主なテスト内容 |
|-------|---------|--------------|
| Dashboard | 13 | 統計表示、パイプラインステータス、カテゴリ進捗 |
| Registry | 15 | タブ切り替え、オブジェクト追加フォーム、フィルタリング |
| Collection | 12 | オブジェクト選択、収集方法タブ、ファイルアップロード |
| Annotation | 13 | タブ切り替え、クラス選択、デバイス選択、セッション管理 |
| Training | 20 | 設定タブ、データセット選択、GPU状態、モデル管理 |
| Evaluation | 16 | 評価実行、競技要件表示、ビジュアルテスト、ロバスト性テスト |
| Settings | 17 | プロファイル管理、データ管理、カテゴリ追加、システム状態 |

### 5.4 E2Eテスト実行方法

```bash
# セットアップ
cd tests/e2e
npm install
npx playwright install chromium

# Docker起動（別ターミナル）
docker compose up -d

# テスト実行
npm test                                    # 全テスト
npm run test:ui                             # UIモード
npx playwright test smoke/                  # Smokeテストのみ
npx playwright test pages/dashboard.spec.ts # 特定ページのみ

# レポート表示
npx playwright show-report
```

### 5.5 Streamlit E2Eテストの注意点

1. **セレクタ戦略**: Streamlitの`data-testid`属性を活用
   - `[data-testid="stAppViewContainer"]` - アプリコンテナ
   - `[data-testid="stSidebar"]` - サイドバー
   - `[data-testid="stSelectbox"]` - セレクトボックス
   - `[data-testid="stButton"]` - ボタン

2. **待機戦略**: Streamlitのrerun処理に対応
   - `waitForAppLoad()` - 初期読み込み待機
   - `waitForRerun()` - 状態変更後の再描画待機
   - `waitForSpinnerClear()` - ローディング完了待機

3. **並列実行**: Streamlitのセッション状態の都合上、単一ワーカーで順次実行

4. **タイムアウト設定**: Streamlitの応答に合わせて拡張
   - `actionTimeout: 15000` (15秒)
   - `navigationTimeout: 30000` (30秒)
   - `testTimeout: 60000` (60秒)
