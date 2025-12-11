# 過学習問題の原因分析レポート

> **注意**: このドキュメントは分析レポートです。コード修正は行いません。

## 調査対象
- **プロファイル**: profile_4 (tid-test-17obj-ver)
- **データセット**: dataset_20251211_koike_house_test (17クラス × 200サンプル = 3,400枚)
- **トレーニング**: training_20251211_040815

---

## 過学習の証拠

### トレーニング履歴からの分析

| Epoch | mAP50 | mAP50-95 | 備考 |
|-------|-------|----------|------|
| 20 | 0.995 | 0.728 | 早期にmAP50が飽和 |
| 37 | 0.995 | 0.804 | mAP50-95のみ緩やかに上昇 |
| 54 | 0.995 | 0.840 | 最高値 |
| 68 | 0.995 | 0.815 | 停滞・微減 |

**問題点**:
1. **mAP50がEpoch 20で0.995に到達し、以降変化なし** → Validationセットへの完璧な適合
2. **mAP50-95の不安定な推移** (0.72→0.84→0.82) → 一般化性能の不安定さ
3. **patience=15でもEarly Stoppingが発動せず** → 改善指標の選択問題

---

## 過学習の主要原因

### 1. データセット構成の問題

| 項目 | 現在の設定 | 問題点 |
|------|-----------|--------|
| Train/Val比率 | 85% / 15% | Valセットが少なすぎる（約510枚/17クラス ≈ 30枚/クラス） |
| サンプル数 | 200枚/クラス | 連続撮影による類似画像が多い可能性 |
| データ多様性 | 不明 | 同一シーン・同一照明条件の画像が多い可能性 |

**根本原因**: Validationセットが少なく、かつTrainセットと類似した画像が多いため、Trainデータを記憶するだけで高いVal精度が出る

### 2. トレーニングパラメータの問題

#### profile_4で使用した設定（High Accuracyプリセット）:
```json
{
  "epochs": 80,          // 多すぎる
  "batch": 8,            // 小さすぎる（GPUに対して）
  "patience": 15,        // 長すぎる
  "lr0": 0.0008,         // 適切
  "weight_decay": 0.0005, // 標準的だが不十分
  "mosaic": 1.0,         // OK
  "mixup": 0.2,          // OK
  "freeze": なし         // 全層学習 → 過学習しやすい
}
```

| パラメータ | 現在値 | 推奨値 | 理由 |
|-----------|--------|--------|------|
| epochs | 80 | 40-50 | 過剰学習防止 |
| patience | 15 | 8-10 | 早期停止を早める |
| freeze | なし | 10 | 最初の10層を凍結し、特徴抽出層を固定 |
| weight_decay | 0.0005 | 0.001-0.002 | 正則化を強化 |
| batch | 8 | 16 | より安定した勾配 |

### 3. データ分割の問題

`scripts/annotation/annotation_utils.py:273`の`split_dataset()`関数:
- `random.shuffle()`のみでランダム分割
- **連続フレームが同じsplit（train/val）に入る可能性がある**
- 同一オブジェクトの類似画像がtrain/val両方に存在 → リーク

---

## 対策案

### 即時対策（パラメータ調整）

#### A. トレーニングパラメータの変更

1. **epochs削減**: 80 → 40-50
2. **patience短縮**: 15 → 8-10
3. **freeze有効化**: freeze=10（最初の10バックボーン層を凍結）
4. **weight_decay増加**: 0.0005 → 0.001
5. **learning rate調整**: lr0=0.0005, lrf=0.01

#### B. 推奨設定（修正版High Accuracyプリセット）

```python
{
    "epochs": 50,
    "batch": 16,
    "patience": 10,
    "close_mosaic": 10,
    "freeze": 10,              # 追加: バックボーン凍結
    # Optimizer
    "optimizer": "AdamW",
    "lr0": 0.0005,             # 下げる
    "lrf": 0.01,
    "weight_decay": 0.001,     # 増やす
    # Augmentation（強化）
    "hsv_h": 0.02,
    "hsv_s": 0.8,
    "hsv_v": 0.5,
    "degrees": 15.0,
    "translate": 0.15,
    "scale": 0.6,
    "shear": 3.0,
    "mosaic": 1.0,
    "mixup": 0.2,
}
```

---

## 推奨パラメータ設定（GUIでの設定変更）

次回トレーニング時に、以下のパラメータをGUIのAdvanced Parametersで設定してください:

### 最重要（今すぐ変更すべき）

| パラメータ | 現在値 (profile_4) | 推奨値 | 設定場所 |
|-----------|-------------------|--------|----------|
| **epochs** | 80 | **50** | 基本設定 |
| **patience** | 15 | **10** | Advanced → Performance |
| **freeze** | なし | **10** | ※現在GUIに無い |
| **weight_decay** | 0.0005 | **0.001** | Advanced → Optimizer |
| **lr0** | 0.0008 | **0.0005** | Advanced → Optimizer |

### プリセット選択

**「High Accuracy」ではなく「Competition」プリセットを使用することを推奨**

理由:
- High Accuracyプリセットはepochs=80, patience=15と過学習しやすい設定
- Competitionプリセットはepochs=50, patience=10, freeze=10と過学習対策済み

### 追加推奨設定

| パラメータ | 推奨値 | 理由 |
|-----------|--------|------|
| batch | 16 | 8より安定した勾配（GPU VRAMに余裕があれば） |
| close_mosaic | 10 | 最終10エポックでmosaic無効化 |
| lrf | 0.01 | 最終学習率を適切に |

---

## freeze パラメータについて

### 現状の問題

`quick_finetune.py`のCOMPETITION_CONFIGには`freeze: 10`が定義されていますが、GUI経由（training_advanced_params.py）のプリセットには**freezeパラメータが含まれていません**。

これにより、GUI経由でトレーニングすると**全層が学習対象**となり、小規模データセットで過学習しやすくなります。

### 影響

- **freeze=10**: 最初の10バックボーン層を凍結 → 事前学習済みの特徴抽出能力を保持
- **freeze=0（現状）**: 全層が学習対象 → 小規模データに過適合しやすい

### 対処法

1. **コマンドライン経由でトレーニング**:
   ```bash
   python scripts/training/quick_finetune.py --dataset data.yaml
   # COMPETITION_CONFIGのfreeze=10が適用される
   ```

2. **GUIを使う場合**（将来的な改善点）:
   - Advanced Parametersにfreezeパラメータを追加する必要あり

---

## 検証方法

トレーニング後、以下を確認してください:

1. **mAP50が0.99を大きく超えていないか**
   - 0.99以上は過学習の疑い
   - 目標: 0.85-0.95程度

2. **Train loss vs Val lossの乖離**
   - Train lossが下がり続けているのにVal lossが上昇 → 過学習
   - TensorBoardで確認可能

3. **未知の画像でのテスト**
   - Validationに含まれない新しい画像で推論テスト
   - Visual Verificationタブで確認

---

## まとめ

### 過学習の主な原因

1. **freeze未使用** - 全層が学習対象で過適合しやすい
2. **epochs過多** (80) - 長すぎる学習
3. **patience過長** (15) - Early Stoppingが遅い
4. **weight_decay不足** (0.0005) - 正則化が弱い
5. **Validationセットが小さい** (15%) - 汎化性能の評価が不正確

### 推奨アクション

| 優先度 | アクション | 方法 |
|--------|-----------|------|
| 1 | **Competitionプリセットを使用** | GUIでプリセット選択 |
| 2 | **epochs=50に短縮** | 基本設定で変更 |
| 3 | **patience=10に短縮** | Advanced → Performance |
| 4 | **weight_decay=0.001** | Advanced → Optimizer |
| 5 | **lr0=0.0005** | Advanced → Optimizer |
| 6 | **CLI経由でトレーニング** | freeze=10を有効化するため |

---

## 参考: 関連ファイル

| ファイル | 内容 |
|----------|------|
| `scripts/training/quick_finetune.py:48-90` | COMPETITION_CONFIG定義（freeze=10含む） |
| `app/components/training_advanced_params.py:22-53` | Competitionプリセット（freezeなし） |
| `profiles/prof_4/app_data/tasks/training_20251211_040815.json` | 過学習したトレーニングの設定 |
