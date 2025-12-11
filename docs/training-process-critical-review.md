# YOLOv8 ファインチューニング プロセス批判的レビュー

**作成日**: 2025-12-11
**対象**: HSR Perception RoboCup トレーニングパイプライン
**レビュー観点**: YOLO Best Practices / Few-shot Learning / 最新Fine-tuning技術 (2024-2025)

---

## 1. エグゼクティブサマリー

### 1.1 現状評価の総括

現在のトレーニングパイプラインは、競技当日の制約（45分以内のファインチューニング）を考慮した実用的な設計となっている。GPU自動スケーリング、OOMリカバリー、TensorBoard統合など、プロダクション品質の機能が実装されている。

しかし、**50-200サンプル/クラス**という少量データ環境において、以下の課題が確認された：

| 評価項目 | 現状 | 評価 |
|----------|------|------|
| モデルサイズ選択 | YOLOv8m (25.9M params) | **要改善** - 小データセットには過大 |
| 正則化強度 | weight_decay=0.001, freeze=10 | **良好** - 最近の改善で適切 |
| データ拡張 | mosaic=1.0, mixup=0.1 | **要改善** - 小データセットには過剰 |
| 学習率戦略 | 単一LR、cosine decay | **改善余地あり** - LLRD未実装 |
| 過学習対策 | Early stopping (patience=10) | **良好** - 適切な設定 |

### 1.2 重要な発見事項

1. **過学習リスク**: YOLOv8m (25.9M params) と 50-200 samples/class の組み合わせでは、パラメータ/サンプル比が約1600:1となり、過学習リスクが高い

2. **早期収束**: 過去のトレーニングログでは、mAP50が3-4エポックで99.5%に到達しており、検証セット汚染またはモデル記憶の兆候がある

3. **未活用の最新技術**: Layer-wise LR Decay (LLRD)、Stochastic Weight Averaging (SWA)、Multi-scale Training など、効果的な手法が未実装

### 1.3 優先度別改善提案

| 優先度 | 改善項目 | 期待効果 | 実装工数 |
|--------|----------|----------|----------|
| **Tier 1** | 小データセット時のYOLOv8s自動選択 | +2-3% 汎化性能 | 低 |
| **Tier 1** | mosaic を 0.5-0.8 に削減 | 過学習軽減 | 低 |
| **Tier 1** | multi_scale: True 追加 | +1-2% mAP | 低 |
| **Tier 2** | Layer-wise LR Decay (LLRD) | +1-3% mAP | 中 |
| **Tier 2** | Stochastic Weight Averaging (SWA) | +1-2% mAP | 中 |
| **Tier 3** | SAM Optimizer | +1-2% mAP | 高 |

---

## 2. 現状実装の詳細分析

### 2.1 トレーニング設定の評価

#### 基本設定 (`scripts/training/quick_finetune.py`)

```python
COMPETITION_CONFIG = {
    "model": "yolov8m.pt",
    "imgsz": 640,
    "epochs": 50,
    "batch": 16,
    "patience": 10,
    "optimizer": "AdamW",
    "lr0": 0.001,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.001,
    "warmup_epochs": 3,
    "freeze": 10,
    # Augmentation
    "mosaic": 1.0,
    "mixup": 0.1,
    "close_mosaic": 10,
    # ...
}
```

#### パラメータ別評価

| パラメータ | 現在値 | 評価 | コメント |
|-----------|--------|------|----------|
| `model` | yolov8m.pt | **△** | 小データセット（<500 images）にはyolov8s.pt推奨 |
| `imgsz` | 640 | **○** | 適切。競技環境に合致 |
| `epochs` | 50 | **○** | Fine-tuning に適切 |
| `batch` | 16 | **○** | GPU メモリに応じて適切にスケール |
| `patience` | 10 | **○** | 適切。早すぎず遅すぎず |
| `optimizer` | AdamW | **○** | Fine-tuning の標準選択 |
| `lr0` | 0.001 | **○** | AdamW での Fine-tuning に適切 |
| `lrf` | 0.01 | **○** | 終端LR = 0.00001、適切な減衰 |
| `weight_decay` | 0.001 | **○** | 最近改善済み。小データセット向け |
| `warmup_epochs` | 3 | **○** | 50エポックに対して適切 |
| `freeze` | 10 | **○** | バックボーン保護に効果的 |
| `mosaic` | 1.0 | **△** | 小データセットには過剰。0.5-0.8 推奨 |
| `mixup` | 0.1 | **○** | 適切。少量データでは0.2-0.3も検討 |
| `close_mosaic` | 10 | **△** | 小データセットでは15推奨 |

### 2.2 強み

#### 2.2.1 GPU自動スケーリング (`scripts/training/gpu_scaler.py`)

```python
GPUTier.LOW: {"model": "yolov8s.pt", "batch": 8, "imgsz": 480}
GPUTier.MEDIUM: {"model": "yolov8m.pt", "batch": 16, "imgsz": 640}
GPUTier.HIGH: {"model": "yolov8l.pt", "batch": 32, "imgsz": 640}
```

**評価**: ハードウェアに応じた自動最適化は優れた設計。ただし、データセットサイズに基づくモデル選択は未実装。

#### 2.2.2 OOMリカバリー機構

```python
OOM_BATCH_REDUCTION_FACTOR = 0.5  # 50%に削減
MAX_OOM_RETRIES = 3
```

**評価**: プロダクション品質。競技当日の安定性を確保。

#### 2.2.3 EMA (Exponential Moving Average)

Ultralytics が自動的に有効化。decay=0.9999 で最先端の設定。

#### 2.2.4 TensorBoard統合

リアルタイムモニタリングにより、過学習の早期検出が可能。

### 2.3 課題

#### 2.3.1 モデルサイズとデータセットサイズの不均衡

| モデル | パラメータ数 | 推奨データセットサイズ |
|--------|-------------|----------------------|
| YOLOv8n | 3.2M | < 500 images |
| YOLOv8s | 11.2M | 500-2000 images |
| **YOLOv8m** | **25.9M** | **2000-10000 images** |
| YOLOv8l | 43.7M | > 10000 images |

**問題**: 現在のデフォルト（YOLOv8m）は、50-200 samples/class の環境では過大。

#### 2.3.2 Layer-wise LR Decay 未実装

現在は全層に同一学習率を適用。研究によると、LLRDで1-3%のmAP改善が可能。

#### 2.3.3 Validation戦略の脆弱性

20%のValidationセット（約100サンプル）では、クラスあたり約5-10サンプルとなり、メトリクスの信頼性が低い。

---

## 3. 専門家視点からの批判的レビュー

### 3.1 YOLO Best Practices視点

#### 3.1.1 学習率設定

| 設定 | 現在値 | 推奨値 | 根拠 |
|------|--------|--------|------|
| lr0 | 0.001 | 0.001 (維持) | AdamWでのFine-tuningに適切 |
| lrf | 0.01 | 0.01 (維持) | Cosine decayで適切な終端値 |
| warmup_epochs | 3 | 3 (維持) | 50エポックに対して適切 |
| cos_lr | False | **True** | 明示的にCosine LRを有効化推奨 |

#### 3.1.2 データ拡張戦略

**問題点**: mosaic=1.0 は小データセットで同一画像の組み合わせが頻発

**推奨変更**:
```python
# データセットサイズに応じた動的設定
def get_augmentation_config(dataset_size: int) -> dict:
    if dataset_size < 500:
        return {"mosaic": 0.5, "mixup": 0.2, "close_mosaic": 20}
    elif dataset_size < 2000:
        return {"mosaic": 0.8, "mixup": 0.15, "close_mosaic": 15}
    else:
        return {"mosaic": 1.0, "mixup": 0.1, "close_mosaic": 10}
```

#### 3.1.3 モデル選択

**推奨**: データセットサイズに基づく自動選択

```python
def select_model(dataset_size: int, num_classes: int) -> str:
    images_per_class = dataset_size / num_classes
    if images_per_class < 100:
        return "yolov8s.pt"  # 小データセット向け
    elif images_per_class < 500:
        return "yolov8m.pt"
    else:
        return "yolov8l.pt"
```

#### 3.1.4 追加推奨パラメータ

```python
# 現在未設定だが推奨するパラメータ
"multi_scale": True,      # スケール不変性向上
"nbs": 16,                # バッチサイズに合わせたloss正規化
"cos_lr": True,           # 明示的Cosine LR
"label_smoothing": 0.1,   # 過信防止
```

### 3.2 Few-shot Learning視点

#### 3.2.1 モデル容量 vs データ量

**深刻な問題**: パラメータ/サンプル比 ≈ 1600:1

| 指標 | 現在 | 推奨 |
|------|------|------|
| モデル | YOLOv8m (25.9M) | YOLOv8s (11.2M) |
| Params/Sample比 | ~1600:1 | < 500:1 |

**影響**: 高い過学習リスク、汎化性能の低下

#### 3.2.2 Few-shot向け正則化強化

```python
# 現在の設定
"weight_decay": 0.001,
"freeze": 10,

# Few-shot最適化設定
"weight_decay": 0.002,     # 強化
"freeze": 15,              # より多くのバックボーン凍結
"label_smoothing": 0.1,    # 追加
"mixup": 0.3,              # 増加
```

#### 3.2.3 Two-Stage Fine-Tuning (TFA) アプローチ

**現状**: 全トレーニング期間で freeze=10 固定

**推奨**: Progressive Unfreezing
```python
# Epoch 1-15:  freeze=20 (head のみ学習)
# Epoch 16-35: freeze=10 (部分的unfreeze)
# Epoch 36-50: freeze=0  (全層学習、低LR)
```

#### 3.2.4 Validation戦略の改善

**問題**: 20% validation = 約100サンプル → クラスあたり5-10サンプル

**推奨**:
1. Stratified Group Split (クラス均衡を保証)
2. 可能であれば 2-fold Cross Validation
3. Test-Time Augmentation (TTA) で評価の堅牢性向上

```python
# TTA を評価時に有効化
results = model.val(data=dataset_yaml, augment=True)
```

### 3.3 最新Fine-tuning技術視点 (2024-2025)

#### 3.3.1 適用可能な最新技術

| 技術 | 適用可否 | 期待効果 | 実装難易度 |
|------|----------|----------|------------|
| LoRA/QLoRA | **不可** | - | CNNには適用不可 |
| Adapter Layers | 低優先 | +0.5% | 推論遅延発生 |
| LLRD | **推奨** | +1-3% | 中 |
| SWA | **推奨** | +1-2% | 中 |
| SAM Optimizer | 検討 | +1-2% | 高（時間2倍） |
| Progressive Unfreezing | **推奨** | +0.5-1% | 中 |
| Multi-scale Training | **推奨** | +1-2% | 低 |
| Model Soups | 検討 | +0.5-1% | 低（後処理） |

#### 3.3.2 Layer-wise Learning Rate Decay (LLRD)

**概要**: 深い層ほど低い学習率を適用

```python
def build_optimizer_with_llrd(model, base_lr=0.001, llrd_rate=0.9):
    """
    Detection head: base_lr
    Backbone layer n: base_lr * (llrd_rate^n)
    """
    param_groups = []

    # Detection head: 最高LR
    param_groups.append({
        'params': model.model[-1].parameters(),
        'lr': base_lr
    })

    # Backbone: 減衰LR
    for i, layer in enumerate(reversed(list(model.model[:-1]))):
        lr = base_lr * (llrd_rate ** (i + 1))
        param_groups.append({'params': layer.parameters(), 'lr': lr})

    return AdamW(param_groups, weight_decay=0.001)
```

**期待効果**: +1-3% mAP改善

#### 3.3.3 Stochastic Weight Averaging (SWA)

**概要**: トレーニング終盤の重みを平均化し、より平坦な最小値を発見

```python
from torch.optim.swa_utils import AveragedModel, SWALR

# 最後10エポックでSWA適用
swa_model = AveragedModel(model)
swa_scheduler = SWALR(optimizer, swa_lr=0.0005)

if epoch >= (epochs - 10):
    swa_model.update_parameters(model)
    swa_scheduler.step()
```

**期待効果**: +1-2% mAP改善、汎化性能向上

#### 3.3.4 Model Soups

**概要**: 異なるシードで学習した複数モデルの重みを平均化

```python
def model_soup(model_paths: List[str]) -> dict:
    state_dicts = [torch.load(p)['model'].state_dict() for p in model_paths]
    averaged = {}
    for key in state_dicts[0].keys():
        averaged[key] = torch.stack([sd[key].float() for sd in state_dicts]).mean(0)
    return averaged
```

**メリット**: 推論時間増加なし、追加学習不要

---

## 4. 改善提案（優先度別）

### 4.1 Tier 1: 即時実装推奨（高インパクト・低工数）

#### 4.1.1 データセットサイズに基づくモデル自動選択

**対象ファイル**: `scripts/training/quick_finetune.py`

```python
def get_optimal_model(dataset_size: int, num_classes: int) -> str:
    """データセットサイズに基づいてモデルを選択"""
    images_per_class = dataset_size / max(num_classes, 1)

    if images_per_class < 100:
        return "yolov8s.pt"  # 過学習防止
    elif images_per_class < 500:
        return "yolov8m.pt"
    else:
        return "yolov8l.pt"
```

#### 4.1.2 Mosaic削減（小データセット向け）

**対象ファイル**: `app/components/training_advanced_params.py`

```python
# Competition プリセットの変更
"Competition": {
    "mosaic": 0.8,       # 1.0 → 0.8
    "close_mosaic": 15,  # 10 → 15
    # ...
}
```

#### 4.1.3 Multi-scale Training有効化

**対象ファイル**: `scripts/training/quick_finetune.py`

```python
COMPETITION_CONFIG = {
    # ...
    "multi_scale": True,  # 追加
}
```

**注意**: VRAMが6GB未満のGPUでは無効化を推奨

#### 4.1.4 Label Smoothing追加

```python
"label_smoothing": 0.1,  # 追加
```

### 4.2 Tier 2: 中期実装推奨（高インパクト・中工数）

#### 4.2.1 Layer-wise Learning Rate Decay (LLRD)

**実装場所**: `scripts/training/quick_finetune.py` のオプティマイザ構築部分

**工数**: 2-3時間

**期待効果**: +1-3% mAP

#### 4.2.2 Stochastic Weight Averaging (SWA)

**実装場所**: トレーニングループにSWAロジック追加

**工数**: 2-3時間

**期待効果**: +1-2% mAP

#### 4.2.3 Progressive Unfreezing

**実装場所**: カスタムコールバックでエポックごとにfreeze値を変更

```python
class ProgressiveUnfreezeCallback:
    def on_epoch_end(self, trainer):
        if trainer.epoch == 15:
            trainer.model.freeze = 10
        elif trainer.epoch == 35:
            trainer.model.freeze = 0
```

**工数**: 2時間

### 4.3 Tier 3: 将来検討（中インパクト・高工数）

#### 4.3.1 SAM Optimizer

**トレードオフ**: 学習時間が2倍になる

**推奨**: 45分制約がない場合のみ検討

#### 4.3.2 Model Soups

**前提条件**: 複数の学習ランを実行可能な準備時間がある場合

---

## 5. 推奨設定値

### 5.1 Few-shot最適化設定（50-200 samples/class）

```python
FEWSHOT_OPTIMIZED_CONFIG = {
    # Model - 小データセット向け
    "model": "yolov8s.pt",
    "imgsz": 640,

    # Training - 早期停止を積極的に
    "epochs": 50,
    "batch": 16,
    "patience": 5,          # 10 → 5

    # Optimizer - 安定性重視
    "optimizer": "AdamW",
    "lr0": 0.001,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.002,  # 0.001 → 0.002
    "cos_lr": True,

    # Regularization - 強化
    "freeze": 15,           # 10 → 15
    "warmup_epochs": 3,
    "label_smoothing": 0.1, # 追加

    # Augmentation - 適度に
    "mosaic": 0.5,          # 1.0 → 0.5
    "mixup": 0.3,           # 0.1 → 0.3
    "close_mosaic": 20,     # 10 → 20
    "hsv_h": 0.015,
    "hsv_s": 0.6,           # 0.7 → 0.6
    "hsv_v": 0.4,
    "degrees": 10.0,
    "translate": 0.1,
    "scale": 0.4,           # 0.5 → 0.4

    # Performance
    "multi_scale": True,    # 追加
    "workers": 8,
    "cache": True,
    "amp": True,
}
```

### 5.2 Competition（標準）設定の改善案

```python
IMPROVED_COMPETITION_CONFIG = {
    # 現在の良好な設定を維持しつつ改善
    "model": "yolov8m.pt",  # または自動選択
    "imgsz": 640,
    "epochs": 50,
    "batch": 16,
    "patience": 10,

    # 改善項目
    "mosaic": 0.8,          # 1.0 → 0.8
    "close_mosaic": 15,     # 10 → 15
    "multi_scale": True,    # 追加
    "cos_lr": True,         # 明示的に有効化
    "label_smoothing": 0.1, # 追加

    # その他は現状維持
    # ...
}
```

### 5.3 High Accuracy設定の改善案

```python
IMPROVED_HIGH_ACCURACY_CONFIG = {
    "model": "yolov8m.pt",
    "imgsz": 640,
    "epochs": 80,
    "patience": 15,

    # SWA有効化（最後15エポック）
    "swa_epochs": 15,
    "swa_lr": 0.0005,

    # 強い正則化
    "weight_decay": 0.002,
    "freeze": 15,
    "label_smoothing": 0.1,

    # 適度な拡張
    "mosaic": 0.6,
    "mixup": 0.3,
    "close_mosaic": 20,
}
```

---

## 6. 結論と次のアクション

### 6.1 総合評価

現在のトレーニングパイプラインは、競技当日の制約を考慮した**実用的で安定した設計**となっている。特に以下の点が優れている：

- GPU自動スケーリングによるハードウェア適応
- OOMリカバリーによる安定性
- TensorBoard統合によるモニタリング
- 最近実装された freeze=10, weight_decay=0.001 などの過学習対策

ただし、**50-200 samples/class** という少量データ環境に対して、以下の最適化余地がある：

1. モデルサイズの自動選択（YOLOv8s推奨）
2. Mosaic強度の削減
3. Layer-wise LR Decay の実装
4. Multi-scale Training の有効化

### 6.2 実装優先順位

| 順位 | 改善項目 | 工数 | 期待効果 | 推奨時期 |
|------|----------|------|----------|----------|
| 1 | Mosaic削減 (0.8) | 5分 | 過学習軽減 | 即時 |
| 2 | multi_scale: True | 5分 | +1-2% mAP | 即時 |
| 3 | label_smoothing: 0.1 | 5分 | 汎化向上 | 即時 |
| 4 | モデル自動選択 | 30分 | +2-3% | 今週中 |
| 5 | LLRD実装 | 3時間 | +1-3% mAP | 次スプリント |
| 6 | SWA実装 | 3時間 | +1-2% mAP | 次スプリント |

### 6.3 リスクと注意事項

1. **multi_scale**: VRAMが6GB未満のGPUではOOMリスク。GPU Tierに応じた条件分岐が必要

2. **モデル変更**: YOLOv8s への変更は推論性能に影響なし（むしろ高速化）だが、大規模データセットでは精度低下の可能性

3. **LLRD/SWA**: Ultralytics標準APIでは直接サポートされないため、カスタム実装が必要

### 6.4 参考文献

- [Ultralytics Configuration Documentation](https://docs.ultralytics.com/usage/cfg/)
- [YOLOv8 Hyperparameter Tuning Guide](https://docs.ultralytics.com/guides/hyperparameter-tuning/)
- [Fine-Tuning Without Forgetting (arXiv)](https://arxiv.org/html/2505.01016v1)
- [Sharpness-Aware Minimization (arXiv)](https://arxiv.org/abs/2010.01412)
- [Stochastic Weight Averaging (PyTorch)](https://pytorch.org/blog/stochastic-weight-averaging-in-pytorch/)

---

*このレポートは、YOLO Best Practices、Few-shot Learning、最新Fine-tuning技術の3つの専門家視点から、HSR Perception RoboCupトレーニングパイプラインを批判的にレビューした結果をまとめたものです。*
