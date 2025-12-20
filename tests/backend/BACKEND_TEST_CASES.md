# バックエンドテスト項目一覧

このドキュメントは、バックエンド（`scripts/`、`src/`）のテスト項目を網羅的に列挙しています。
別セッションでテストを実装する際の参照資料として使用してください。

## 凡例

- ✅ 実装済み
- ⬜ 未実装
- 🔶 優先度: 高
- 🔷 優先度: 中
- ⬜ 優先度: 低

---

## 1. Common モジュール (`scripts/common/`)

### 1.1 test_config_utils.py ✅

**ソースファイル**: `scripts/common/config_utils.py`

**モック要件**: なし（データクラスのみ）

**優先度**: 🔶 高

**状態**: 実装済み - `tests/backend/common/test_config_utils.py` (36テスト)

| 状態 | テストクラス | テストメソッド | 説明 |
|------|-------------|---------------|------|
| ✅ | `TestAnnotatorConfig` | `test_default_values()` | デフォルト値の確認 |
| ✅ | `TestAnnotatorConfig` | `test_custom_values()` | カスタム値の設定 |
| ✅ | `TestBackgroundSubtractionConfig` | `test_default_values()` | デフォルト値の確認 |
| ✅ | `TestBackgroundSubtractionConfig` | `test_blur_kernel_validation_odd()` | カーネルサイズが奇数であること |
| ✅ | `TestBackgroundSubtractionConfig` | `test_threshold_method_validation()` | 有効な閾値手法 |
| ✅ | `TestBackgroundSubtractionConfig` | `test_invalid_threshold_method_raises()` | 無効な手法でエラー |
| ✅ | `TestSAM2Config` | `test_default_values()` | デフォルト値の確認 |
| ✅ | `TestSAM2Config` | `test_custom_device()` | カスタムデバイス設定 |
| ✅ | `TestSAM2Config` | `test_model_type_validation()` | モデルタイプの検証 |
| ✅ | `TestGetSam2ModelConfig` | `test_sam21_base_model()` | SAM2.1 Baseの設定取得 |
| ✅ | `TestGetSam2ModelConfig` | `test_sam21_large_model()` | SAM2.1 Largeの設定取得 |
| ✅ | `TestGetSam2ModelConfig` | `test_sam2_fallback_to_sam21()` | SAM2→SAM2.1フォールバック |
| ✅ | `TestGetSam2ModelConfig` | `test_unknown_model_defaults_to_base()` | 不明モデルはbaseにフォールバック |
| ✅ | `TestTrainingConfig` | `test_default_configuration()` | デフォルト設定の確認 |
| ✅ | `TestTrainingConfig` | `test_to_dict_serialization()` | 辞書へのシリアライズ |
| ✅ | `TestTrainingConfig` | `test_competition_preset()` | 競技会プリセット |
| ✅ | `TestTrainingConfig` | `test_fast_preset()` | 高速テストプリセット |
| ✅ | `TestEvaluationConfig` | `test_default_values()` | デフォルト値の確認 |
| ✅ | `TestLoadClassConfig` | `test_load_existing_config()` | 既存設定ファイルの読み込み |
| ✅ | `TestLoadClassConfig` | `test_file_not_found_raises()` | ファイル未発見時のエラー |
| ✅ | `TestGetClassNames` | `test_extract_class_names()` | クラス名の抽出 |
| ✅ | `TestGetClassNames` | `test_empty_objects_list()` | 空のオブジェクトリスト |
| ✅ | `TestGetClassIdMap` | `test_create_mapping()` | ID→名前マッピングの作成 |

---

### 1.2 test_constants.py ✅

**ソースファイル**: `scripts/common/constants.py`

**モック要件**: なし

**優先度**: ⬜ 低

**状態**: 実装済み - `tests/backend/common/test_constants.py` (35テスト)

| 状態 | テストクラス | テストメソッド | 説明 |
|------|-------------|---------------|------|
| ✅ | `TestImageExtensions` | `test_image_extensions_is_tuple()` | IMAGE_EXTENSIONSがタプルであること |
| ✅ | `TestImageExtensions` | `test_contains_common_formats()` | 一般的なフォーマット含有 |
| ✅ | `TestImageExtensions` | `test_all_lowercase()` | すべて小文字 |
| ✅ | `TestImageExtensions` | `test_all_start_with_dot()` | すべてドットで始まる |
| ✅ | `TestCompetitionDefaults` | `test_default_target_samples()` | デフォルトサンプル数 |
| ✅ | `TestCompetitionDefaults` | `test_default_group_interval()` | グルーピング間隔 |
| ✅ | `TestCompetitionDefaults` | `test_default_burst_count()` | バースト数 |
| ✅ | `TestCompetitionDefaults` | `test_default_burst_interval()` | バースト間隔 |
| ✅ | `TestModelDefaults` | `test_target_map50()` | mAP50ターゲット |
| ✅ | `TestModelDefaults` | `test_target_inference_ms()` | 推論時間ターゲット |
| ✅ | `TestModelDefaults` | `test_default_confidence_threshold()` | 信頼度閾値 |
| ✅ | `TestModelDefaults` | `test_default_iou_threshold()` | IoU閾値 |
| ✅ | `TestTrainingDefaults` | `test_default_yolo_model()` | デフォルトモデル |
| ✅ | `TestTrainingDefaults` | `test_default_epochs()` | デフォルトエポック数 |
| ✅ | `TestTrainingDefaults` | `test_default_batch_size()` | デフォルトバッチサイズ |
| ✅ | `TestTrainingDefaults` | `test_default_image_size()` | デフォルト画像サイズ |
| ✅ | `TestTrainingDefaults` | `test_default_patience()` | デフォルトpatience |
| ✅ | `TestGPUScalingConstants` | `test_gpu_vram_thresholds_is_dict()` | VRAM閾値が辞書 |
| ✅ | `TestGPUScalingConstants` | `test_vram_thresholds_keys()` | VRAM閾値のキー |
| ✅ | `TestGPUScalingConstants` | `test_vram_thresholds_values_are_positive()` | 閾値が正の値 |
| ✅ | `TestGPUScalingConstants` | `test_batch_size_by_vram_is_dict()` | バッチサイズが辞書 |
| ✅ | `TestGPUScalingConstants` | `test_workers_by_vram_is_dict()` | ワーカー数が辞書 |
| ✅ | `TestPathConstants` | `test_default_output_dir()` | デフォルト出力ディレクトリ |
| ✅ | `TestPathConstants` | `test_default_dataset_dir()` | デフォルトデータセットディレクトリ |
| ✅ | `TestPathConstants` | `test_default_models_dir()` | デフォルトモデルディレクトリ |
| ✅ | `TestAnnotationConstants` | `test_annotation_defaults()` | アノテーションデフォルト値 |
| ✅ | `TestAnnotationConstants` | `test_sam2_defaults()` | SAM2デフォルト値 |
| ✅ | `TestROSConstants` | `test_ros_topic_defaults()` | ROSトピックデフォルト |
| ✅ | `TestROSConstants` | `test_ros_service_names()` | ROSサービス名 |
| ✅ | `TestConstantsIntegrity` | `test_no_mutable_defaults()` | ミュータブルデフォルトなし |
| ✅ | `TestConstantsIntegrity` | `test_constants_are_importable()` | インポート可能 |

---

### 1.3 test_validation.py ✅

**ソースファイル**: `scripts/common/validation.py`

**モック要件**: ファイルシステム（`tmp_path`フィクスチャ）

**優先度**: 🔶 高

**状態**: 実装済み - `tests/backend/common/test_validation.py` (46テスト)

| 状態 | テストクラス | テストメソッド | 説明 |
|------|-------------|---------------|------|
| ✅ | `TestErrorSeverity` | `test_enum_values()` | Enumの値確認 |
| ✅ | `TestPipelineError` | `test_creation()` | エラーオブジェクト作成 |
| ✅ | `TestPipelineError` | `test_format_with_color()` | カラー付きフォーマット |
| ✅ | `TestPipelineError` | `test_format_without_color()` | カラーなしフォーマット |
| ✅ | `TestValidationResult` | `test_default_is_valid()` | デフォルトは有効 |
| ✅ | `TestValidationResult` | `test_add_error_invalidates()` | エラー追加で無効化 |
| ✅ | `TestValidationResult` | `test_add_warning_keeps_valid()` | 警告追加は有効維持 |
| ✅ | `TestValidationResult` | `test_merge_results()` | 結果のマージ |
| ✅ | `TestValidationResult` | `test_format_all()` | 全エラーのフォーマット |
| ✅ | `TestValidateDatasetYaml` | `test_file_not_found()` | ファイル未発見 |
| ✅ | `TestValidateDatasetYaml` | `test_invalid_yaml_format()` | 不正なYAML形式 |
| ✅ | `TestValidateDatasetYaml` | `test_missing_required_fields()` | 必須フィールド欠落 |
| ✅ | `TestValidateDatasetYaml` | `test_train_path_not_found()` | 訓練パス未発見 |
| ✅ | `TestValidateDatasetYaml` | `test_no_classes_defined()` | クラス未定義 |
| ✅ | `TestValidateDatasetYaml` | `test_warning_single_class()` | 単一クラス警告 |
| ✅ | `TestValidateDatasetYaml` | `test_warning_few_images()` | 画像不足警告 |
| ✅ | `TestValidateDatasetYaml` | `test_valid_dataset()` | 有効なデータセット |
| ✅ | `TestValidateYoloAnnotation` | `test_file_not_found()` | ファイル未発見 |
| ✅ | `TestValidateYoloAnnotation` | `test_empty_file_is_valid()` | 空ファイルは有効 |
| ✅ | `TestValidateYoloAnnotation` | `test_wrong_number_of_values()` | 値の数が不正 |
| ✅ | `TestValidateYoloAnnotation` | `test_invalid_number_format()` | 数値形式が不正 |
| ✅ | `TestValidateYoloAnnotation` | `test_negative_class_id()` | 負のクラスID |
| ✅ | `TestValidateYoloAnnotation` | `test_out_of_range_coordinates()` | 座標が範囲外 |
| ✅ | `TestValidateYoloAnnotation` | `test_valid_annotation()` | 有効なアノテーション |
| ✅ | `TestValidateModelPath` | `test_model_not_found()` | モデル未発見 |
| ✅ | `TestValidateModelPath` | `test_yolo_model_auto_download_info()` | YOLO自動DL情報 |
| ✅ | `TestValidateModelPath` | `test_unexpected_extension_warning()` | 予期しない拡張子警告 |
| ✅ | `TestValidateModelPath` | `test_valid_model()` | 有効なモデル |

---

### 1.4 test_device_utils.py ✅

**ソースファイル**: `scripts/common/device_utils.py`

**モック要件**: `torch`モジュールのモック

**優先度**: 🔷 中

**状態**: 実装済み - `tests/backend/common/test_device_utils.py` (18テスト)

| 状態 | テストクラス | テストメソッド | 説明 |
|------|-------------|---------------|------|
| ✅ | `TestCheckCudaAvailable` | `test_cuda_available()` | CUDAが利用可能 |
| ✅ | `TestCheckCudaAvailable` | `test_cuda_not_available()` | CUDAが利用不可 |
| ✅ | `TestCheckCudaAvailable` | `test_torch_import_error()` | torchインポートエラー |
| ✅ | `TestGetDefaultDevice` | `test_returns_cuda_when_available()` | CUDA利用可能時cuda返却 |
| ✅ | `TestGetDefaultDevice` | `test_returns_cpu_when_unavailable()` | CUDA利用不可時cpu返却 |
| ✅ | `TestLogGpuStatus` | `test_verbose_cuda_available()` | 詳細ログCUDA有効 |
| ✅ | `TestLogGpuStatus` | `test_verbose_cuda_not_available()` | 詳細ログCUDA無効 |
| ✅ | `TestLogGpuStatus` | `test_silent_mode()` | サイレントモード |
| ✅ | `TestGetGpuInfo` | `test_multi_gpu_info()` | マルチGPU情報 |
| ✅ | `TestGetGpuInfo` | `test_no_gpu_info()` | GPU無し情報 |
| ✅ | `TestGetOptimalBatchSize` | `test_scale_up_large_memory()` | 大メモリでスケールアップ |
| ✅ | `TestGetOptimalBatchSize` | `test_scale_down_small_memory()` | 小メモリでスケールダウン |
| ✅ | `TestGetOptimalBatchSize` | `test_cpu_fallback()` | CPUフォールバック |

---

### 1.5 test_image_utils.py ✅

**ソースファイル**: `scripts/common/image_utils.py`

**モック要件**: `cv2`, `numpy`（実際の画像データ）

**優先度**: 🔷 中

**状態**: 実装済み - `tests/backend/common/test_image_utils.py` (29テスト)

| 状態 | テストクラス | テストメソッド | 説明 |
|------|-------------|---------------|------|
| ✅ | `TestMaskToBbox` | `test_basic_mask_to_bbox()` | 基本的なマスク→BBox変換 |
| ✅ | `TestMaskToBbox` | `test_empty_mask_returns_none()` | 空マスクはNone返却 |
| ✅ | `TestMaskToBbox` | `test_with_margin()` | マージン付き変換 |
| ✅ | `TestMaskToBbox` | `test_clamp_to_image_bounds()` | 画像境界にクランプ |
| ✅ | `TestFindObjectBbox` | `test_find_object()` | オブジェクト検出 |
| ✅ | `TestFindObjectBbox` | `test_filter_by_min_area()` | 最小面積フィルタ |
| ✅ | `TestFindObjectBbox` | `test_filter_by_max_area_ratio()` | 最大面積比フィルタ |
| ✅ | `TestFindObjectBbox` | `test_no_valid_contours()` | 有効輪郭なし |
| ✅ | `TestDrawBbox` | `test_draw_rectangle()` | 矩形描画 |
| ✅ | `TestDrawBbox` | `test_draw_with_label()` | ラベル付き描画 |
| ✅ | `TestDrawMaskOverlay` | `test_overlay_application()` | オーバーレイ適用 |
| ✅ | `TestDrawMaskOverlay` | `test_alpha_blending()` | アルファブレンディング |
| ✅ | `TestDrawDetections` | `test_multiple_detections()` | 複数検出の描画 |
| ✅ | `TestDrawDetections` | `test_color_map_usage()` | カラーマップ使用 |
| ✅ | `TestDrawDetections` | `test_confidence_display()` | 信頼度表示 |
| ✅ | `TestListImageFiles` | `test_find_jpg_png()` | JPG/PNG検出 |
| ✅ | `TestListImageFiles` | `test_recursive_search()` | 再帰検索 |
| ✅ | `TestListImageFiles` | `test_custom_extensions()` | カスタム拡張子 |
| ✅ | `TestListImageFiles` | `test_nonexistent_directory()` | 存在しないディレクトリ |
| ✅ | `TestLoadImage` | `test_load_bgr()` | BGR読み込み |
| ✅ | `TestLoadImage` | `test_load_rgb()` | RGB読み込み |
| ✅ | `TestLoadImage` | `test_load_gray()` | グレースケール読み込み |
| ✅ | `TestLoadImage` | `test_load_nonexistent()` | 存在しないファイル |
| ✅ | `TestSaveImage` | `test_save_jpg()` | JPG保存 |
| ✅ | `TestSaveImage` | `test_save_png()` | PNG保存 |
| ✅ | `TestSaveImage` | `test_quality_parameter()` | 品質パラメータ |

---

## 2. Training モジュール (`scripts/training/`)

### 2.1 test_memory_utils.py ✅

**ソースファイル**: `scripts/training/memory_utils.py`

**モック要件**: `torch`, `gc`

**優先度**: 🔶 高

**状態**: 実装済み - `tests/backend/training/test_memory_utils.py`

| 状態 | テストクラス | テストメソッド | 説明 |
|------|-------------|---------------|------|
| ✅ | `TestMemoryStats` | `test_dataclass_creation()` | データクラス作成 |
| ✅ | `TestMemoryStats` | `test_to_dict()` | 辞書変換 |
| ✅ | `TestMemoryTracker` | `test_initialization()` | 初期化 |
| ✅ | `TestMemoryTracker` | `test_log_memory()` | メモリログ |
| ✅ | `TestMemoryTracker` | `test_get_peak_memory()` | ピークメモリ取得 |
| ✅ | `TestCleanupCudaMemory` | `test_cleanup_success()` | クリーンアップ成功 |
| ✅ | `TestCleanupCudaMemory` | `test_cleanup_no_cuda()` | CUDA無しクリーンアップ |
| ✅ | `TestCleanupModel` | `test_cleanup_model()` | モデルクリーンアップ |
| ✅ | `TestCleanupModel` | `test_cleanup_none_model()` | Noneモデル |
| ✅ | `TestCleanupOptimizer` | `test_cleanup_optimizer()` | オプティマイザクリーンアップ |
| ✅ | `TestCleanupSwaModel` | `test_cleanup_swa_model()` | SWAモデルクリーンアップ |
| ✅ | `TestCleanupTensorboard` | `test_cleanup_tensorboard()` | TensorBoardクリーンアップ |
| ✅ | `TestFullTrainingCleanup` | `test_full_cleanup()` | 完全クリーンアップ |
| ✅ | `TestLogMemorySnapshot` | `test_log_memory_snapshot()` | メモリスナップショットログ |

---

### 2.2 test_llrd_trainer.py ✅

**ソースファイル**: `scripts/training/llrd_trainer.py`

**モック要件**: `torch`, `ultralytics`

**優先度**: 🔷 中

**状態**: 実装済み - `tests/backend/training/test_llrd_trainer.py`

| 状態 | テストクラス | テストメソッド | 説明 |
|------|-------------|---------------|------|
| ✅ | `TestLLRDConfig` | `test_default_values()` | デフォルト値 |
| ✅ | `TestLLRDConfig` | `test_custom_values()` | カスタム値 |
| ✅ | `TestLLRDConfig` | `test_validation()` | バリデーション |
| ✅ | `TestLayerDepthCalculation` | `test_backbone_layers()` | バックボーンレイヤー深度 |
| ✅ | `TestLayerDepthCalculation` | `test_neck_layers()` | ネックレイヤー深度 |
| ✅ | `TestLayerDepthCalculation` | `test_head_layers()` | ヘッドレイヤー深度 |
| ✅ | `TestLearningRateFormula` | `test_lr_decay_formula()` | LR減衰計算式 |
| ✅ | `TestLearningRateFormula` | `test_lr_at_different_depths()` | 異なる深度でのLR |
| ✅ | `TestFreezeAndLLRDInteraction` | `test_freeze_excludes_llrd()` | フリーズとLLRDの相互作用 |
| ✅ | `TestLayerCategorization` | `test_categorize_backbone()` | バックボーン分類 |
| ✅ | `TestLayerCategorization` | `test_categorize_neck()` | ネック分類 |
| ✅ | `TestLayerCategorization` | `test_categorize_head()` | ヘッド分類 |

---

### 2.3 test_swa_trainer.py ✅

**ソースファイル**: `scripts/training/swa_trainer.py`

**モック要件**: `torch`, `ultralytics`

**優先度**: 🔷 中

**状態**: 実装済み - `tests/backend/training/test_swa_trainer.py`

| 状態 | テストクラス | テストメソッド | 説明 |
|------|-------------|---------------|------|
| ✅ | `TestAdaptiveSWAStartEpoch` | `test_short_training()` | 短期訓練開始エポック |
| ✅ | `TestAdaptiveSWAStartEpoch` | `test_standard_training()` | 標準訓練開始エポック |
| ✅ | `TestAdaptiveSWAStartEpoch` | `test_long_training()` | 長期訓練開始エポック |
| ✅ | `TestSWAConfig` | `test_default_values()` | デフォルト値 |
| ✅ | `TestSWAConfig` | `test_custom_values()` | カスタム値 |
| ✅ | `TestSWAConfig` | `test_to_dict()` | 辞書変換 |
| ✅ | `TestSWAConfig` | `test_from_dict()` | 辞書から作成 |
| ✅ | `TestSWAConfig` | `test_get_swa_lr()` | SWA学習率取得 |
| ✅ | `TestSWAConfig` | `test_get_swa_start_epoch()` | SWA開始エポック取得 |

---

### 2.4 test_training_config.py ✅

**ソースファイル**: `scripts/training/training_config.py`

**モック要件**: `torch`（GPU検出用）

**優先度**: 🔶 高

**状態**: 実装済み - `tests/backend/training/test_training_config.py` (30テスト)

| 状態 | テストクラス | テストメソッド | 説明 |
|------|-------------|---------------|------|
| ✅ | `TestAugmentationConfig` | `test_default_values()` | デフォルト値 |
| ✅ | `TestAugmentationConfig` | `test_hsv_ranges()` | HSV値範囲 |
| ✅ | `TestAugmentationConfig` | `test_geometric_transforms()` | 幾何変換パラメータ |
| ✅ | `TestAugmentationConfig` | `test_mosaic_mixup()` | Mosaic/Mixup設定 |
| ✅ | `TestOptimizerConfig` | `test_default_values()` | デフォルト値 |
| ✅ | `TestOptimizerConfig` | `test_llrd_settings()` | LLRD設定 |
| ✅ | `TestOptimizerConfig` | `test_swa_settings()` | SWA設定 |
| ✅ | `TestPerformanceConfig` | `test_default_values()` | デフォルト値 |
| ✅ | `TestPerformanceConfig` | `test_worker_count()` | ワーカー数 |
| ✅ | `TestPerformanceConfig` | `test_amp_setting()` | AMP設定 |
| ✅ | `TestCheckpointConfig` | `test_default_values()` | デフォルト値 |
| ✅ | `TestCheckpointConfig` | `test_save_period()` | 保存間隔 |
| ✅ | `TestTrainingConfig` | `test_default_configuration()` | デフォルト設定 |
| ✅ | `TestTrainingConfig` | `test_competition_default()` | 競技会デフォルト |
| ✅ | `TestTrainingConfig` | `test_fast_test()` | 高速テスト設定 |
| ✅ | `TestTrainingConfig` | `test_from_gpu_profile()` | GPUプロファイルから作成 |
| ✅ | `TestTrainingConfig` | `test_auto_detect()` | 自動検出 |
| ✅ | `TestTrainingConfig` | `test_to_dict()` | 辞書変換 |
| ✅ | `TestTrainingConfig` | `test_to_yolo_args()` | YOLOArgs変換 |

---

### 2.5 test_gpu_scaler.py ✅

**ソースファイル**: `scripts/training/gpu_scaler.py`

**モック要件**: `torch`, `subprocess`（nvidia-smi用）

**優先度**: 🔶 高

**状態**: 実装済み - `tests/backend/training/test_gpu_scaler.py` (42テスト)

| 状態 | テストクラス | テストメソッド | 説明 |
|------|-------------|---------------|------|
| ✅ | `TestGPUTier` | `test_enum_values()` | Enum値の確認 |
| ✅ | `TestGPUProfile` | `test_dataclass_creation()` | データクラス作成 |
| ✅ | `TestGPUProfile` | `test_to_dict()` | 辞書変換 |
| ✅ | `TestGPUScaler` | `test_initialization()` | 初期化 |
| ✅ | `TestGPUScaler` | `test_detect_gpu_no_cuda()` | CUDA無し検出 |
| ✅ | `TestGPUScaler` | `test_detect_gpu_with_cuda()` | CUDA有り検出 |
| ✅ | `TestGPUScaler` | `test_get_optimal_config_low_tier()` | Low Tier設定取得 |
| ✅ | `TestGPUScaler` | `test_get_optimal_config_medium_tier()` | Medium Tier設定取得 |
| ✅ | `TestGPUScaler` | `test_get_optimal_config_high_tier()` | High Tier設定取得 |
| ✅ | `TestGPUScaler` | `test_get_optimal_config_workstation()` | Workstation設定取得 |
| ✅ | `TestGPUScaler` | `test_calculate_batch_size_8gb()` | 8GB VRAMバッチサイズ |
| ✅ | `TestGPUScaler` | `test_calculate_batch_size_12gb()` | 12GB VRAMバッチサイズ |
| ✅ | `TestGPUScaler` | `test_calculate_batch_size_24gb()` | 24GB VRAMバッチサイズ |
| ✅ | `TestGPUScaler` | `test_estimate_training_time()` | 訓練時間推定 |
| ✅ | `TestOOMRecoveryStrategy` | `test_get_recovery_config_attempt1()` | リカバリー試行1 |
| ✅ | `TestOOMRecoveryStrategy` | `test_get_recovery_config_attempt2()` | リカバリー試行2 |
| ✅ | `TestOOMRecoveryStrategy` | `test_get_recovery_config_attempt3()` | リカバリー試行3 |
| ✅ | `TestOOMRecoveryStrategy` | `test_max_attempts_exceeded()` | 最大試行回数超過 |

---

### 2.6 test_quick_finetune.py ✅

**ソースファイル**: `scripts/training/quick_finetune.py`

**モック要件**: `torch`, `ultralytics`, `colorama`（sys.modules事前モック）

**優先度**: 🔷 中

**状態**: 実装済み - `tests/backend/training/test_quick_finetune.py` (49テスト)

| 状態 | テストクラス | テストメソッド | 説明 |
|------|-------------|---------------|------|
| ✅ | `TestSyntheticConfigKeys` | `test_contains_expected_keys()` | 期待されるキーが含まれている |
| ✅ | `TestSyntheticConfigKeys` | `test_keys_are_strings()` | 全キーが文字列型 |
| ✅ | `TestSyntheticConfigKeys` | `test_is_set()` | setであることを確認 |
| ✅ | `TestCompetitionConfig` | `test_contains_model_settings()` | モデル設定の確認 |
| ✅ | `TestCompetitionConfig` | `test_contains_training_settings()` | 訓練設定の確認 |
| ✅ | `TestCompetitionConfig` | `test_contains_optimizer_settings()` | オプティマイザ設定の確認 |
| ✅ | `TestCompetitionConfig` | `test_contains_augmentation_settings()` | 拡張設定の確認 |
| ✅ | `TestCompetitionConfig` | `test_contains_llrd_settings()` | LLRD設定の確認 |
| ✅ | `TestCompetitionConfig` | `test_contains_synthetic_settings()` | 合成設定の確認 |
| ✅ | `TestFastConfig` | `test_smaller_model_than_competition()` | より小さなモデル |
| ✅ | `TestFastConfig` | `test_fewer_epochs()` | より少ないエポック数 |
| ✅ | `TestFastConfig` | `test_smaller_image_size()` | より小さな画像サイズ |
| ✅ | `TestFastConfig` | `test_inherits_from_competition()` | COMPETITION_CONFIGからの継承 |
| ✅ | `TestTrainingResult` | `test_creation()` | インスタンス作成 |
| ✅ | `TestTrainingResult` | `test_summary_generation()` | サマリ文字列生成 |
| ✅ | `TestTrainingResult` | `test_meets_target_pass()` | ターゲット達成時True |
| ✅ | `TestTrainingResult` | `test_meets_target_fail()` | ターゲット未達時False |
| ✅ | `TestTrainingResult` | `test_meets_target_custom_threshold()` | カスタム閾値 |
| ✅ | `TestTrainingResult` | `test_meets_target_missing_metric()` | メトリクス欠落時 |
| ✅ | `TestTrainingResult` | `test_to_dict()` | 辞書変換 |
| ✅ | `TestTrainingResult` | `test_timestamp_auto_generated()` | タイムスタンプ自動生成 |
| ✅ | `TestCompetitionTrainerInit` | `test_init_default()` | デフォルト初期化 |
| ✅ | `TestCompetitionTrainerInit` | `test_init_with_custom_output()` | カスタム出力ディレクトリ |
| ✅ | `TestCompetitionTrainerInit` | `test_init_with_config()` | カスタム設定で初期化 |
| ✅ | `TestCompetitionTrainerInit` | `test_init_auto_scale_disabled()` | auto_scale無効 |
| ✅ | `TestCompetitionTrainerInit` | `test_init_tensorboard_disabled()` | TensorBoard無効 |
| ✅ | `TestCompetitionTrainerInit` | `test_init_no_gpu_raises()` | GPU無しでエラー |
| ✅ | `TestCompetitionTrainerInit` | `test_init_no_gpu_with_allow_cpu()` | allow_cpuフラグ |
| ✅ | `TestCompetitionTrainerInit` | `test_init_base_model_override()` | base_modelオーバーライド |
| ✅ | `TestValidateDataset` | `test_valid_dataset()` | 有効なデータセット |
| ✅ | `TestValidateDataset` | `test_dataset_not_found()` | ファイル未発見 |
| ✅ | `TestValidateDataset` | `test_missing_required_field()` | 必須フィールド欠落 |
| ✅ | `TestValidateDataset` | `test_train_path_not_found()` | 訓練パス未発見 |
| ✅ | `TestValidateDataset` | `test_val_path_not_found()` | 検証パス未発見 |
| ✅ | `TestSyntheticConfigFiltering` | `test_filter_synthetic_keys_from_config()` | 合成キーフィルタ |
| ✅ | `TestSyntheticConfigFiltering` | `test_yolo_compatible_config_only()` | YOLO互換設定のみ残る |
| ✅ | `TestArgumentParsing` | `test_required_dataset_arg()` | 必須引数 |
| ✅ | `TestArgumentParsing` | `test_dataset_arg_provided()` | データセット引数 |
| ✅ | `TestArgumentParsing` | `test_optional_model_arg()` | オプションモデル引数 |
| ✅ | `TestArgumentParsing` | `test_default_output_dir()` | デフォルト出力ディレクトリ |
| ✅ | `TestArgumentParsing` | `test_fast_flag()` | --fastフラグ |
| ✅ | `TestArgumentParsing` | `test_llrd_flags()` | --llrd, --llrd-decay-rate |
| ✅ | `TestArgumentParsing` | `test_tensorboard_flags()` | TensorBoardフラグ |
| ✅ | `TestArgumentParsing` | `test_gpu_tier_choices()` | GPU tier選択 |
| ✅ | `TestArgumentParsing` | `test_resume_flag()` | --resumeフラグ |
| ✅ | `TestArgumentParsing` | `test_export_choices()` | --export選択 |
| ✅ | `TestArgumentParsing` | `test_dynamic_synthetic_flags()` | 動的合成フラグ |
| ✅ | `TestRunNameGeneration` | `test_run_name_contains_competition()` | run名にcompetition含む |
| ✅ | `TestRunNameGeneration` | `test_run_name_contains_timestamp()` | run名にタイムスタンプ含む |

---

### 2.7 test_tensorboard_monitor.py ✅

**ソースファイル**: `scripts/training/tensorboard_monitor.py`

**モック要件**: `torch.utils.tensorboard`, `subprocess`

**優先度**: ⬜ 低

**状態**: 実装済み - `tests/backend/training/test_tensorboard_monitor.py` (41テスト)

| 状態 | テストクラス | テストメソッド | 説明 |
|------|-------------|---------------|------|
| ✅ | `TestTensorBoardConfig` | `test_default_values()` | デフォルト値確認 |
| ✅ | `TestTensorBoardConfig` | `test_custom_values()` | カスタム値設定 |
| ✅ | `TestTensorBoardConfig` | `test_to_dict()` | 辞書変換 |
| ✅ | `TestTensorBoardConfig` | `test_from_dict()` | 辞書から作成 |
| ✅ | `TestTensorBoardServer` | `test_init()` | 初期化 |
| ✅ | `TestTensorBoardServer` | `test_start()` | サーバー起動 |
| ✅ | `TestTensorBoardServer` | `test_stop()` | サーバー停止 |
| ✅ | `TestTensorBoardServer` | `test_get_url()` | URL取得 |
| ✅ | `TestTensorBoardServer` | `test_is_running()` | 実行状態確認 |
| ✅ | `TestTensorBoardServer` | `test_find_available_port()` | 利用可能ポート検索 |
| ✅ | `TestTensorBoardServer` | `test_start_already_running()` | 既に起動中 |
| ✅ | `TestTensorBoardServer` | `test_stop_not_running()` | 未起動時の停止 |
| ✅ | `TestCompetitionTensorBoardCallback` | `test_init()` | 初期化 |
| ✅ | `TestCompetitionTensorBoardCallback` | `test_on_train_epoch_end()` | エポック終了時 |
| ✅ | `TestCompetitionTensorBoardCallback` | `test_on_fit_epoch_end()` | フィット終了時 |
| ✅ | `TestCompetitionTensorBoardCallback` | `test_log_scalar()` | スカラーログ |
| ✅ | `TestCompetitionTensorBoardCallback` | `test_log_scalars()` | 複数スカラーログ |
| ✅ | `TestCompetitionTensorBoardCallback` | `test_log_image()` | 画像ログ |
| ✅ | `TestCompetitionTensorBoardCallback` | `test_close()` | クローズ |
| ✅ | `TestGPUMonitorCallback` | `test_init()` | 初期化 |
| ✅ | `TestGPUMonitorCallback` | `test_log_gpu_stats()` | GPU統計ログ |
| ✅ | `TestGPUMonitorCallback` | `test_on_train_batch_end()` | バッチ終了時 |
| ✅ | `TestGPUMonitorCallback` | `test_no_cuda()` | CUDA無し時 |
| ✅ | `TestTensorBoardManager` | `test_init()` | 初期化 |
| ✅ | `TestTensorBoardManager` | `test_setup()` | セットアップ |
| ✅ | `TestTensorBoardManager` | `test_cleanup()` | クリーンアップ |
| ✅ | `TestTensorBoardManager` | `test_context_manager()` | コンテキストマネージャ |
| ✅ | `TestTensorBoardManager` | `test_get_callback()` | コールバック取得 |
| ✅ | `TestTensorBoardManager` | `test_get_server_url()` | サーバーURL取得 |

---

## 3. Annotation モジュール (`scripts/annotation/`)

### 3.1 test_annotation_utils.py ✅

**ソースファイル**: `scripts/annotation/annotation_utils.py`

**モック要件**: ファイルシステム、`yaml`

**優先度**: 🔶 高

**状態**: 実装済み - `tests/backend/annotation/test_annotation_utils.py` (53テスト)

| 状態 | テストクラス | テストメソッド | 説明 |
|------|-------------|---------------|------|
| ✅ | `TestAnnotationResult` | `test_default_values()` | デフォルト値 |
| ✅ | `TestAnnotationResult` | `test_success_rate_calculation()` | 成功率計算 |
| ✅ | `TestAnnotationResult` | `test_summary_format()` | サマリフォーマット |
| ✅ | `TestBboxToYolo` | `test_basic_conversion()` | 基本変換 |
| ✅ | `TestBboxToYolo` | `test_clamp_values()` | 値のクランプ |
| ✅ | `TestBboxToYolo` | `test_edge_cases()` | エッジケース |
| ✅ | `TestYoloToBbox` | `test_basic_conversion()` | 基本変換 |
| ✅ | `TestYoloToBbox` | `test_roundtrip_conversion()` | 往復変換 |
| ✅ | `TestYoloToBbox` | `test_clamp_to_image()` | 画像サイズにクランプ |
| ✅ | `TestWriteYoloLabel` | `test_write_single_label()` | 単一ラベル書き込み |
| ✅ | `TestWriteYoloLabel` | `test_append_mode()` | 追記モード |
| ✅ | `TestWriteYoloLabel` | `test_precision()` | 精度確認 |
| ✅ | `TestReadYoloLabel` | `test_read_valid_labels()` | 有効ラベル読み込み |
| ✅ | `TestReadYoloLabel` | `test_empty_file()` | 空ファイル |
| ✅ | `TestReadYoloLabel` | `test_file_not_found()` | ファイル未発見 |
| ✅ | `TestValidateYoloAnnotation` | `test_valid_annotation()` | 有効アノテーション |
| ✅ | `TestValidateYoloAnnotation` | `test_invalid_field_count()` | フィールド数不正 |
| ✅ | `TestValidateYoloAnnotation` | `test_out_of_range_values()` | 範囲外の値 |
| ✅ | `TestCreateDatasetYaml` | `test_create_yaml()` | YAML作成 |
| ✅ | `TestCreateDatasetYaml` | `test_with_test_path()` | テストパス付き |
| ✅ | `TestSplitDataset` | `test_basic_split()` | 基本分割 |
| ✅ | `TestSplitDataset` | `test_group_continuous_frames()` | 連続フレームグルーピング |
| ✅ | `TestSplitDataset` | `test_symlink_mode()` | シンボリックリンクモード |
| ✅ | `TestSplitDataset` | `test_seed_reproducibility()` | シード再現性 |
| ✅ | `TestExtractTimestamp` | `test_valid_timestamp_format()` | 有効タイムスタンプ |
| ✅ | `TestExtractTimestamp` | `test_invalid_format()` | 無効フォーマット |
| ✅ | `TestGroupByTimestamp` | `test_group_nearby_frames()` | 近接フレームグルーピング |
| ✅ | `TestGroupByTimestamp` | `test_separate_distant_frames()` | 離れたフレームの分離 |
| ✅ | `TestMaskToBbox` | `test_numpy_detection()` | NumPy検出 |
| ✅ | `TestMaskToBbox` | `test_contour_detection()` | 輪郭検出 |
| ✅ | `TestMaskToBbox` | `test_margin_expansion()` | マージン拡張 |

---

### 3.2 test_base_annotator.py ✅

**ソースファイル**: `scripts/annotation/base_annotator.py`

**モック要件**: 抽象クラステスト

**優先度**: 🔷 中

**状態**: 実装済み - `tests/backend/annotation/test_base_annotator.py` (11テスト)

| 状態 | テストクラス | テストメソッド | 説明 |
|------|-------------|---------------|------|
| ✅ | `TestBaseAnnotator` | `test_abstract_method_enforcement()` | 抽象メソッド強制 |
| ✅ | `TestBaseAnnotator` | `test_cannot_instantiate()` | インスタンス化不可 |
| ✅ | `TestConcreteAnnotator` | `test_annotate_batch()` | バッチアノテーション |
| ✅ | `TestConcreteAnnotator` | `test_progress_callback()` | 進捗コールバック |
| ✅ | `TestConcreteAnnotator` | `test_visualize_annotation()` | アノテーション可視化 |

---

### 3.3 test_background_subtraction.py ✅

**ソースファイル**: `scripts/annotation/background_subtraction.py`

**モック要件**: `cv2`、テスト画像

**優先度**: 🔶 高

**状態**: 実装済み - `tests/backend/annotation/test_background_subtraction.py` (28テスト)

| 状態 | テストクラス | テストメソッド | 説明 |
|------|-------------|---------------|------|
| ✅ | `TestBackgroundSubtractionAnnotator` | `test_init_with_config()` | 設定付き初期化 |
| ✅ | `TestBackgroundSubtractionAnnotator` | `test_init_with_background_image()` | 背景画像付き初期化 |
| ✅ | `TestBackgroundSubtractionAnnotator` | `test_annotate_single_image()` | 単一画像アノテーション |
| ✅ | `TestBackgroundSubtractionAnnotator` | `test_no_object_detected()` | オブジェクト未検出 |
| ✅ | `TestBackgroundSubtractionAnnotator` | `test_threshold_method_otsu()` | Otsu閾値法 |
| ✅ | `TestBackgroundSubtractionAnnotator` | `test_threshold_method_adaptive()` | 適応的閾値法 |
| ✅ | `TestBackgroundSubtractionAnnotator` | `test_threshold_method_fixed()` | 固定閾値法 |
| ✅ | `TestBackgroundSubtractionAnnotator` | `test_morphological_operations()` | モルフォロジー演算 |
| ✅ | `TestBackgroundSubtractionAnnotator` | `test_batch_annotation()` | バッチアノテーション |
| ✅ | `TestBackgroundSubtractionAnnotator` | `test_min_contour_area_filter()` | 最小輪郭面積フィルタ |
| ✅ | `TestBackgroundSubtractionAnnotator` | `test_max_area_ratio_filter()` | 最大面積比フィルタ |

---

### 3.4 test_sam2_annotator.py ✅

**ソースファイル**: `scripts/annotation/sam2_annotator.py`

**モック要件**: `sam2`モジュール、`torch`

**優先度**: 🔷 中

**状態**: 実装済み - `tests/backend/annotation/test_sam2_annotator.py` (12テスト)

| 状態 | テストクラス | テストメソッド | 説明 |
|------|-------------|---------------|------|
| ✅ | `TestSAM2Annotator` | `test_init_with_config()` | 設定付き初期化 |
| ✅ | `TestSAM2Annotator` | `test_init_model_loading()` | モデル読み込み |
| ✅ | `TestSAM2Annotator` | `test_annotate_single_image()` | 単一画像アノテーション |
| ✅ | `TestSAM2Annotator` | `test_mask_generation()` | マスク生成 |
| ✅ | `TestSAM2Annotator` | `test_mask_to_bbox_conversion()` | マスク→BBox変換 |
| ✅ | `TestSAM2Annotator` | `test_gpu_device_usage()` | GPUデバイス使用 |
| ✅ | `TestSAM2Annotator` | `test_cpu_fallback()` | CPUフォールバック |
| ✅ | `TestSAM2Annotator` | `test_batch_annotation()` | バッチアノテーション |

---

### 3.5 test_auto_annotate.py ✅

**ソースファイル**: `scripts/annotation/auto_annotate.py`

**モック要件**: アノテータクラスのモック

**優先度**: 🔷 中

**状態**: 実装済み - `tests/backend/annotation/test_auto_annotate.py` (14テスト)

| 状態 | テストクラス | テストメソッド | 説明 |
|------|-------------|---------------|------|
| ✅ | `TestAutoAnnotatePipeline` | `test_background_subtraction_method()` | 背景差分法 |
| ✅ | `TestAutoAnnotatePipeline` | `test_sam2_method()` | SAM2法 |
| ✅ | `TestAutoAnnotatePipeline` | `test_invalid_method()` | 無効な手法 |
| ✅ | `TestAutoAnnotatePipeline` | `test_output_directory_creation()` | 出力ディレクトリ作成 |
| ✅ | `TestAutoAnnotatePipeline` | `test_progress_tracking()` | 進捗追跡 |
| ✅ | `TestAutoAnnotatePipeline` | `test_error_handling()` | エラーハンドリング |

---

## 4. Augmentation モジュール (`scripts/augmentation/`)

### 4.1 test_object_extractor.py ✅

**ソースファイル**: `scripts/augmentation/object_extractor.py`

**モック要件**: `cv2`, `numpy`

**優先度**: 🔶 高

**状態**: 実装済み - `tests/backend/augmentation/test_object_extractor.py`

| 状態 | テストクラス | テストメソッド | 説明 |
|------|-------------|---------------|------|
| ✅ | `TestExtractedObject` | `test_dataclass_creation()` | データクラス作成 |
| ✅ | `TestExtractedObject` | `test_save_and_load()` | 保存と読み込み |
| ✅ | `TestObjectExtractor` | `test_initialization()` | 初期化 |
| ✅ | `TestObjectExtractor` | `test_extract_object()` | オブジェクト抽出 |
| ✅ | `TestObjectExtractor` | `test_extract_with_soft_alpha()` | ソフトアルファ抽出 |
| ✅ | `TestObjectExtractor` | `test_extract_batch()` | バッチ抽出 |
| ✅ | `TestCreateSoftAlpha` | `test_gaussian_blur_edge()` | ガウシアンブラーエッジ |
| ✅ | `TestSaveAndLoadObject` | `test_roundtrip()` | 往復テスト |

---

### 4.2 test_copy_paste_augmentor.py ✅

**ソースファイル**: `scripts/augmentation/copy_paste_augmentor.py`

**モック要件**: `cv2`, `numpy`, テスト画像

**優先度**: 🔶 高

**状態**: 実装済み - `tests/backend/augmentation/test_copy_paste_augmentor.py`

| 状態 | テストクラス | テストメソッド | 説明 |
|------|-------------|---------------|------|
| ✅ | `TestCopyPasteConfig` | `test_default_values()` | デフォルト値 |
| ✅ | `TestCopyPasteConfig` | `test_custom_values()` | カスタム値 |
| ✅ | `TestCopyPasteConfig` | `test_validation()` | バリデーション |
| ✅ | `TestCopyPasteAugmentor` | `test_initialization()` | 初期化 |
| ✅ | `TestCopyPasteAugmentor` | `test_synthesize_image()` | 画像合成 |
| ✅ | `TestCopyPasteAugmentor` | `test_object_placement()` | オブジェクト配置 |
| ✅ | `TestCopyPasteAugmentor` | `test_color_correction()` | 色補正 |
| ✅ | `TestCopyPasteAugmentor` | `test_scale_range()` | スケール範囲 |
| ✅ | `TestCopyPasteAugmentor` | `test_rotation_range()` | 回転範囲 |
| ✅ | `TestCopyPasteAugmentor` | `test_overlap_prevention()` | オーバーラップ防止 |
| ✅ | `TestRotation` | `test_rotation_0_degrees()` | 0度回転 |
| ✅ | `TestRotation` | `test_rotation_90_degrees()` | 90度回転 |
| ✅ | `TestRotation` | `test_rotation_180_degrees()` | 180度回転 |
| ✅ | `TestBatchExtraction` | `test_batch_synthesis()` | バッチ合成 |

---

## 5. Evaluation モジュール (`scripts/evaluation/`)

### 5.1 test_evaluate_model.py ✅

**ソースファイル**: `scripts/evaluation/evaluate_model.py`

**モック要件**: `ultralytics`, `torch`

**優先度**: 🔷 中

**状態**: 実装済み - `tests/backend/evaluation/test_evaluate_model.py` (16テスト)

| 状態 | テストクラス | テストメソッド | 説明 |
|------|-------------|---------------|------|
| ✅ | `TestClassMetrics` | `test_dataclass_creation()` | データクラス作成 |
| ✅ | `TestClassMetrics` | `test_to_dict()` | 辞書変換 |
| ✅ | `TestEvaluationReport` | `test_dataclass_creation()` | データクラス作成 |
| ✅ | `TestEvaluationReport` | `test_meets_competition_requirements()` | 競技会要件確認 |
| ✅ | `TestEvaluationReport` | `test_to_dict()` | 辞書変換 |
| ✅ | `TestEvaluateModel` | `test_evaluate_valid_model()` | 有効モデル評価 |
| ✅ | `TestEvaluateModel` | `test_model_not_found()` | モデル未発見 |
| ✅ | `TestEvaluateModel` | `test_dataset_not_found()` | データセット未発見 |
| ✅ | `TestEvaluateModel` | `test_inference_time_measurement()` | 推論時間測定 |
| ✅ | `TestEvaluateModel` | `test_per_class_metrics()` | クラス別メトリクス |
| ✅ | `TestCompetitionRequirements` | `test_map_threshold()` | mAP閾値 |
| ✅ | `TestCompetitionRequirements` | `test_inference_time_threshold()` | 推論時間閾値 |

---

### 5.2 test_visual_verification.py ✅

**ソースファイル**: `scripts/evaluation/visual_verification.py`

**モック要件**: `cv2`, `ultralytics`

**優先度**: ⬜ 低

**状態**: 実装済み - `tests/backend/evaluation/test_visual_verification.py` (16テスト)

| 状態 | テストクラス | テストメソッド | 説明 |
|------|-------------|---------------|------|
| ✅ | `TestVisualVerifierInit` | `test_init_with_model_path()` | モデルパスで初期化 |
| ✅ | `TestVisualVerifierInit` | `test_init_with_class_config()` | クラス設定付き初期化 |
| ✅ | `TestVisualVerifierInit` | `test_init_without_class_config()` | クラス設定なし初期化 |
| ✅ | `TestGenerateColors` | `test_generate_colors_for_each_class()` | 各クラスに色生成 |
| ✅ | `TestGenerateColors` | `test_color_format_is_tuple()` | 色がタプル形式 |
| ✅ | `TestDrawDetections` | `test_draw_detections_returns_image()` | 検出描画が画像を返す |
| ✅ | `TestDrawDetections` | `test_draw_detections_preserves_original()` | 元画像を保持 |
| ✅ | `TestDrawDetections` | `test_draw_detections_with_color_override()` | 色オーバーライド |
| ✅ | `TestDrawDetections` | `test_draw_empty_detections()` | 空の検出リスト |
| ✅ | `TestPredictImage` | `test_predict_returns_tuple()` | 予測結果がタプル |
| ✅ | `TestPredictImage` | `test_predict_detection_format()` | 検出フォーマット確認 |
| ✅ | `TestVerifyBatch` | `test_verify_batch_empty_dir()` | 空ディレクトリのバッチ検証 |
| ✅ | `TestCreateComparisonGrid` | `test_create_grid_returns_array()` | グリッドがnumpy配列 |
| ✅ | `TestCreateComparisonGrid` | `test_create_grid_with_output()` | グリッドをファイル保存 |
| ✅ | `TestGenerateReportSamples` | `test_generate_samples_creates_files()` | サンプルファイル生成 |
| ✅ | `TestGenerateReportSamples` | `test_generate_samples_empty_dir()` | 空ディレクトリテスト |

---

## 6. Capture モジュール (`scripts/capture/`)

### 6.1 test_burst_capture.py ✅

**ソースファイル**: `scripts/capture/burst_capture.py`

**モック要件**: `rclpy`, `cv2`

**優先度**: ⬜ 低

**状態**: 実装済み - `tests/backend/capture/test_burst_capture.py` (17テスト)

| 状態 | テストクラス | テストメソッド | 説明 |
|------|-------------|---------------|------|
| ✅ | `TestImgmsgToCv2` | `test_rgb8_encoding()` | RGB8エンコーディング変換 |
| ✅ | `TestImgmsgToCv2` | `test_bgr8_encoding()` | BGR8エンコーディング変換 |
| ✅ | `TestImgmsgToCv2` | `test_mono8_encoding()` | Mono8エンコーディング変換 |
| ✅ | `TestImgmsgToCv2` | `test_mono16_encoding()` | Mono16エンコーディング変換 |
| ✅ | `TestImgmsgToCv2` | `test_rgba8_encoding()` | RGBA8エンコーディング変換 |
| ✅ | `TestImgmsgToCv2` | `test_16uc1_encoding()` | 16UC1エンコーディング変換 |
| ✅ | `TestBurstCaptureInit` | `test_initialization()` | 初期化 |
| ✅ | `TestBurstCaptureInit` | `test_initialization_creates_output_dir()` | 出力ディレクトリ作成 |
| ✅ | `TestBurstCaptureCallback` | `test_callback_stores_latest_frame()` | 最新フレーム保存 |
| ✅ | `TestBurstCaptureCallback` | `test_callback_handles_conversion_error()` | 変換エラー処理 |
| ✅ | `TestBurstCaptureCaptureIfReady` | `test_capture_if_ready_no_frame()` | フレームなし |
| ✅ | `TestBurstCaptureCaptureIfReady` | `test_capture_if_ready_interval_not_passed()` | インターバル未経過 |
| ✅ | `TestBurstCaptureCaptureIfReady` | `test_capture_if_ready_captures_image()` | キャプチャ実行 |
| ✅ | `TestBurstCaptureCaptureIfReady` | `test_capture_if_ready_returns_true_when_done()` | 完了時True返却 |
| ✅ | `TestBurstCaptureIsDone` | `test_is_done_false()` | 未完了 |
| ✅ | `TestBurstCaptureIsDone` | `test_is_done_true()` | 完了 |
| ✅ | `TestBurstCaptureIsDone` | `test_is_done_true_exceeded()` | ターゲット超過 |

---

### 6.2 test_capture_frame.py ✅

**ソースファイル**: `scripts/capture/capture_frame.py`

**モック要件**: `rclpy`, `cv_bridge`, `cv2`

**優先度**: ⬜ 低

**状態**: 実装済み - `tests/backend/capture/test_capture_frame.py` (11テスト)

| 状態 | テストクラス | テストメソッド | 説明 |
|------|-------------|---------------|------|
| ✅ | `TestFrameCaptureInit` | `test_initialization()` | 初期化 |
| ✅ | `TestFrameCaptureInit` | `test_initialization_attributes()` | 属性値確認 |
| ✅ | `TestFrameCaptureCallback` | `test_callback_rgb8_encoding()` | RGB8コールバック |
| ✅ | `TestFrameCaptureCallback` | `test_callback_bgr8_encoding()` | BGR8コールバック |
| ✅ | `TestFrameCaptureCallback` | `test_callback_mono8_encoding()` | Mono8コールバック |
| ✅ | `TestFrameCaptureCallback` | `test_callback_depth_16uc1_encoding()` | 深度16UC1コールバック |
| ✅ | `TestFrameCaptureCallback` | `test_callback_already_received()` | 受信済みスキップ |
| ✅ | `TestFrameCaptureCallback` | `test_callback_conversion_error()` | 変換エラー処理 |
| ✅ | `TestFrameCaptureCallback` | `test_callback_unknown_encoding()` | 不明エンコーディング |
| ✅ | `TestFrameCaptureEncoding` | `test_encoding_list_rgb()` | RGBエンコーディングリスト |
| ✅ | `TestFrameCaptureEncoding` | `test_encoding_depth_list()` | 深度エンコーディングリスト |

---

## 7. GUI Framework モジュール (`scripts/gui_framework/`)

### 7.1 test_base_app.py ✅

**ソースファイル**: `scripts/gui_framework/base_app.py`

**モック要件**: `tkinter`, `AppTheme`

**優先度**: ⬜ 低

**状態**: 実装済み - `tests/backend/gui_framework/test_base_app.py` (17テスト)

| 状態 | テストクラス | テストメソッド | 説明 |
|------|-------------|---------------|------|
| ✅ | `TestBaseAppAbstract` | `test_cannot_instantiate_directly()` | 直接インスタンス化不可 |
| ✅ | `TestBaseAppAbstract` | `test_is_abstract_class()` | ABC継承確認 |
| ✅ | `TestBaseAppAbstract` | `test_abstract_methods_defined()` | 抽象メソッド定義確認 |
| ✅ | `TestBaseAppSubclass` | `test_subclass_can_instantiate()` | サブクラスインスタンス化 |
| ✅ | `TestBaseAppSubclass` | `test_subclass_sets_title()` | タイトル設定 |
| ✅ | `TestBaseAppSubclass` | `test_subclass_sets_geometry()` | ジオメトリ設定 |
| ✅ | `TestBaseAppSubclass` | `test_subclass_sets_min_size()` | 最小サイズ設定 |
| ✅ | `TestBaseAppKeyboardShortcuts` | `test_q_key_binds_to_close()` | qキーバインド |
| ✅ | `TestBaseAppKeyboardShortcuts` | `test_escape_handler_default()` | Escapeハンドラ |
| ✅ | `TestBaseAppDialogs` | `test_show_error_dialog()` | エラーダイアログ |
| ✅ | `TestBaseAppDialogs` | `test_show_warning_dialog()` | 警告ダイアログ |
| ✅ | `TestBaseAppDialogs` | `test_show_info_dialog()` | 情報ダイアログ |
| ✅ | `TestBaseAppDialogs` | `test_ask_yes_no_returns_true()` | Yes/No True返却 |
| ✅ | `TestBaseAppDialogs` | `test_ask_yes_no_returns_false()` | Yes/No False返却 |
| ✅ | `TestBaseAppRun` | `test_run_calls_mainloop()` | runがmainloop呼出 |
| ✅ | `TestBaseAppProtocol` | `test_close_protocol_bound()` | クローズプロトコル |
| ✅ | `TestBaseAppSetStatus` | `test_set_status_default()` | ステータス設定 |

---

### 7.2 test_ros2_app.py ✅

**ソースファイル**: `scripts/gui_framework/ros2_app.py`

**モック要件**: `rclpy`, `tkinter`, `ROS2ImageSubscriber`

**優先度**: ⬜ 低

**状態**: 実装済み - `tests/backend/gui_framework/test_ros2_app.py` (14テスト)

| 状態 | テストクラス | テストメソッド | 説明 |
|------|-------------|---------------|------|
| ✅ | `TestROS2AppAbstract` | `test_cannot_instantiate_directly()` | 直接インスタンス化不可 |
| ✅ | `TestROS2AppAbstract` | `test_is_subclass_of_base_app()` | BaseAppサブクラス確認 |
| ✅ | `TestROS2AppAbstract` | `test_abstract_methods_defined()` | 抽象メソッド定義確認 |
| ✅ | `TestROS2AppInit` | `test_init_initializes_ros2()` | ROS2初期化確認 |
| ✅ | `TestROS2AppInit` | `test_init_starts_ros_thread()` | ROSスレッド開始確認 |
| ✅ | `TestROS2AppInit` | `test_init_ros2_failure()` | ROS2初期化失敗時 |
| ✅ | `TestROS2AppShutdown` | `test_shutdown_stops_ros_thread()` | シャットダウン時スレッド停止 |
| ✅ | `TestROS2AppShutdown` | `test_shutdown_calls_rclpy_shutdown()` | rclpy.shutdown呼出 |
| ✅ | `TestROS2AppGetFrame` | `test_get_frame_returns_frame()` | フレーム返却 |
| ✅ | `TestROS2AppGetFrame` | `test_get_frame_returns_none_when_no_node()` | ノードなしでNone返却 |
| ✅ | `TestROS2AppGetImageTopics` | `test_get_image_topics_returns_list()` | トピックリスト返却 |
| ✅ | `TestROS2AppGetImageTopics` | `test_get_image_topics_returns_empty_when_no_node()` | ノードなしで空リスト |
| ✅ | `TestROS2AppSubscribeToTopic` | `test_subscribe_to_topic_calls_node()` | トピック購読呼出 |
| ✅ | `TestROS2AppSubscribeToTopic` | `test_subscribe_to_topic_no_node()` | ノードなしでエラーなし |

---

## 8. ROS2 パッケージ (`src/hsr_perception/`)

### 8.1 test_continuous_capture_node.py ✅

**ソースファイル**: `src/hsr_perception/hsr_perception/continuous_capture_node.py`

**モック要件**: `rclpy`, `sensor_msgs`, `cv_bridge`

**優先度**: 🔷 中

**状態**: 実装済み - `tests/backend/ros2/test_continuous_capture_node.py` (15テスト)

| 状態 | テストクラス | テストメソッド | 説明 |
|------|-------------|---------------|------|
| ✅ | `TestContinuousCaptureNode` | `test_node_initialization()` | ノード初期化 |
| ✅ | `TestContinuousCaptureNode` | `test_image_subscription()` | 画像購読 |
| ✅ | `TestContinuousCaptureNode` | `test_set_class_service()` | SetClassサービス |
| ✅ | `TestContinuousCaptureNode` | `test_start_burst_service()` | StartBurstサービス |
| ✅ | `TestContinuousCaptureNode` | `test_get_status_service()` | GetStatusサービス |
| ✅ | `TestContinuousCaptureNode` | `test_burst_capture_execution()` | バースト撮影実行 |
| ✅ | `TestContinuousCaptureNode` | `test_image_saving()` | 画像保存 |
| ✅ | `TestContinuousCaptureNode` | `test_jpeg_quality_setting()` | JPEG品質設定 |

---

## テスト実行方法

```bash
# 全バックエンドテスト実行
pytest tests/backend/ -v

# 特定モジュールのテスト
pytest tests/backend/training/ -v
pytest tests/backend/annotation/ -v
pytest tests/backend/augmentation/ -v

# カバレッジレポート付き
pytest tests/backend/ --cov=scripts --cov-report=html

# GPU必須テストをスキップ
pytest tests/backend/ -v -m "not gpu"

# 遅いテストをスキップ
pytest tests/backend/ -v -m "not slow"
```

---

## 統計

| カテゴリ | 実装済み | 未実装 | 合計 |
|---------|---------|--------|------|
| Common | 5 | 0 | 5 |
| Training | 7 | 0 | 7 |
| Annotation | 5 | 0 | 5 |
| Augmentation | 2 | 0 | 2 |
| Evaluation | 2 | 0 | 2 |
| Capture | 2 | 0 | 2 |
| GUI Framework | 2 | 0 | 2 |
| ROS2 | 1 | 0 | 1 |
| **合計** | **26** | **0** | **26** |

---

## 優先度別実装順序

### Phase 1 (高優先度) ✅ 完了
1. ✅ `test_validation.py` - パイプライン全体で使用
2. ✅ `test_config_utils.py` - 設定管理の基盤
3. ✅ `test_annotation_utils.py` - アノテーションの中核
4. ✅ `test_gpu_scaler.py` - 訓練の自動最適化
5. ✅ `test_background_subtraction.py` - 主要アノテーション手法

### Phase 2 (中優先度) ✅ 完了
1. ✅ `test_device_utils.py` (18テスト)
2. ✅ `test_image_utils.py` (29テスト)
3. ✅ `test_training_config.py` (30テスト)
4. ✅ `test_base_annotator.py` (11テスト)
5. ✅ `test_sam2_annotator.py` (12テスト)
6. ✅ `test_auto_annotate.py` (14テスト)
7. ✅ `test_evaluate_model.py` (16テスト)
8. ✅ `test_continuous_capture_node.py` (15テスト)

### Phase 3 (低優先度) ✅ 完了
1. ✅ `test_constants.py` (35テスト)
2. ✅ `test_quick_finetune.py` (49テスト)
3. ✅ `test_tensorboard_monitor.py` (41テスト)
4. ✅ `test_visual_verification.py` (16テスト)
5. ✅ `test_burst_capture.py` (17テスト)
6. ✅ `test_capture_frame.py` (11テスト)
7. ✅ `test_base_app.py` (17テスト)
8. ✅ `test_ros2_app.py` (14テスト)
