<div align="center">

[日本語](#日本語) | [English](#english)

</div>

---

# 日本語

## HSR Perception Pipeline for RoboCup@Home

RoboCup@Home 大会向けの HSR 用物体認識パイプライン。このブランチは
[`pybullet_hsr`](../pybullet_hsr) で生成される YOLO 形式データセットを読み込んで
学習と評価だけを行う、絞り込んだ構成になっています。データ収集・自動アノテーション・
ROS2 キャプチャノードは削除済みです。

**環境**: Ubuntu 22.04 / Python 3.10 / CUDA 12.1 (Docker)

### パイプライン（3 ステップ）

```
prepare-dataset → train → evaluate
```

| ステップ | スクリプト | 役割 |
|---------|-----------|------|
| prepare-dataset | `scripts/data/prepare_dataset.py` | pybullet_hsr の出力を train/val に分割し `data.yaml` を生成 |
| train | `scripts/training/quick_finetune.py` | YOLOv8 fine-tuning（GPU 自動スケーリング / LLRD / SWA / TensorBoard 対応） |
| evaluate | `scripts/evaluation/evaluate_model.py` | mAP・推論速度の検証 |

### 前提条件（Docker 実行）

| 項目 | 要件 |
|------|------|
| Docker | 24.0+ |
| Docker Compose | v2+ |
| NVIDIA Driver | 525+ |
| NVIDIA Container Toolkit | インストール済み |
| pybullet_hsr | `/home/roboworks/repos/pybullet_hsr` にクローン済み（または `PYBULLET_HSR_ROOT` で指定） |

### クイックスタート

```bash
# 依存関係のインストール（ローカル実行する場合）
pip install -r requirements.txt

# --- ワンショット（推奨）: 最新 dump を同期してそのまま学習 ----------------
# sync_latest.py は、ローカルの datasets/<name>/ が最新 dump と同期済みなら
# no-op で終わるので、学習のたびに走らせて問題ない。--force で強制再生成。
python scripts/data/sync_latest.py
./start.sh sync                                 # Docker 版（同じ no-op 挙動）
./start.sh train-latest -- --fast --epochs 1   # 同期 + 学習を 1 コマンドで

# --- 段階的に実行する場合 -------------------------------------------------

# 1. データセット準備（manifest.json を読んで train/val 分割 + data.yaml 生成）
#    --latest は $PYBULLET_HSR_ROOT/annotation_data/ 配下の最新 manifest 付き dump を自動選択
#    既存 datasets/<name>/ が同じ dump・設定で準備済みなら自動で no-op（--force で再生成）。
python scripts/data/prepare_dataset.py \
    --source /home/roboworks/repos/pybullet_hsr/annotation_data --latest --symlink

# 2. 学習（出力先は datasets/<dataset_name>/data.yaml — manifest の dataset_name がデフォルト名）
python scripts/training/quick_finetune.py --dataset datasets/<dataset_name>/data.yaml --fast --epochs 1

# 3. 評価
python scripts/evaluation/evaluate_model.py \
    --model models/finetuned/<run>/weights/best.pt \
    --dataset datasets/<dataset_name>/data.yaml
```

### データセット同期の仕組み

`prepare_dataset.py` は成功時に `datasets/<name>/.prepare_meta.json` を
書き出し、source dump のパス・`manifest.created_at`・val ratio・seed・
symlink 設定を記録します。次回以降の `prepare_dataset.py` / `sync_latest.py`
呼び出しは、このサイドカーと新しい dump を突き合わせて **変更がなければ
no-op** で終了します（表示は `status=up-to-date`）。これにより、学習の
たびに同期コマンドを叩いても無駄な再生成が走りません。

Streamlit UI（Dashboard・Training）は同じサイドカーを参照し、
「✅ 同期済み」「⚠️ 新しい dump があります」のバッジと、ワンクリックの
**Sync latest dump** / **Prepare & Train on latest dump** ボタンを表示します。

**manifest が未生成の dump**（古い生成物など）は、pybullet_hsr 側で 1 度遡及生成する必要があります:

```bash
cd /home/roboworks/repos/pybullet_hsr
python scripts/write_manifest.py --dump-dir annotation_data/<name>_<ts> \
    --classes-yaml configs/datasets/<name>.yaml
```

### Streamlit UI

```bash
./run_app.sh                  # ローカル実行（要: pip install -r requirements.txt）
./start.sh                    # Docker で起動し http://localhost:8501 へ
./start.sh --tensorboard      # TensorBoard（6006）も起動
docker compose run --rm app prepare-dataset --source /pybullet_hsr/annotation_data --latest --symlink
docker compose run --rm app train --dataset datasets/<dataset_name>/data.yaml --fast
```

UI は起動時に `annotation_data/*/manifest.json` を自動 glob し、Dashboard と Training ページに
検出済み dump を一覧表示します。

UI のページ構成：Dashboard（状態サマリ） / Training（準備・学習） / Evaluation（mAP 評価） /
Settings（環境変数とパス）。

### ディレクトリ構成

```
hsr-perception-robocup/
├── app/                    # Streamlit UI（Dashboard / Training / Evaluation / Settings）
├── scripts/
│   ├── data/               # pybullet_hsr dump → YOLO dataset 変換
│   ├── training/           # YOLOv8 fine-tuning
│   ├── evaluation/         # 評価ツール
│   └── common/             # 共通ユーティリティ
├── docker/                 # Dockerfile + entrypoint
├── config/                 # 設定ファイル（現在ほぼ空）
├── models/finetuned/       # 学習結果
├── datasets/               # 準備済み YOLO データセット
├── runs/                   # TensorBoard ログ
└── tests/                  # pytest（backend + frontend）
```

### 環境変数

| 変数 | 既定値 | 用途 |
|------|--------|------|
| `PYBULLET_HSR_ROOT` | `/home/roboworks/repos/pybullet_hsr` (ホスト) / `/pybullet_hsr` (Docker) | pybullet_hsr クローンへのパス |

### テスト

```bash
docker compose run --rm app test tests/backend/training tests/backend/evaluation -v
```

### 参考資料

- [pybullet_hsr](../pybullet_hsr) — データセット生成元
- [RoboCup@Home Rulebook](https://github.com/RoboCupAtHome/RuleBook)
- [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/)

---

# English

## HSR Perception Pipeline for RoboCup@Home

Training + evaluation pipeline that consumes YOLO-format datasets produced by the companion
`pybullet_hsr` repository. Data collection, auto-annotation, and the ROS2 capture node used to
live here but have been removed — they are no longer in scope for this branch.

**Environment**: Ubuntu 22.04 / Python 3.10 / CUDA 12.1 (Docker)

### Pipeline (3 steps)

```
prepare-dataset → train → evaluate
```

| Step | Script | Role |
|------|--------|------|
| prepare-dataset | `scripts/data/prepare_dataset.py` | Split the pybullet_hsr dump into train/val + emit `data.yaml` |
| train | `scripts/training/quick_finetune.py` | YOLOv8 fine-tuning (GPU auto-scaling, LLRD, SWA, TensorBoard) |
| evaluate | `scripts/evaluation/evaluate_model.py` | mAP + inference-time check |

### Quick start

```bash
pip install -r requirements.txt

# Fast path: sync the newest dump (no-op if already fresh) and train.
python scripts/data/sync_latest.py
./start.sh train-latest -- --fast --epochs 1   # Docker one-shot

# Or step by step:
python scripts/data/prepare_dataset.py \
    --source /home/roboworks/repos/pybullet_hsr/annotation_data --latest --symlink

python scripts/training/quick_finetune.py --dataset datasets/<dataset_name>/data.yaml --fast --epochs 1
python scripts/evaluation/evaluate_model.py \
    --model models/finetuned/<run>/weights/best.pt \
    --dataset datasets/<dataset_name>/data.yaml
```

The dump's `manifest.json` is the single source of truth. If a dump was generated before the
manifest emitter was added, run pybullet_hsr's `scripts/write_manifest.py --dump-dir <dir>
--classes-yaml configs/datasets/<name>.yaml` once to emit it retroactively.

### Docker

```bash
./start.sh                   # Build + run Streamlit at http://localhost:8501
./start.sh --tensorboard     # Also run TensorBoard (6006)
./start.sh sync              # Prepare the newest dump (no-op if already fresh)
./start.sh train-latest -- --fast --epochs 1   # Sync + train in one go
docker compose run --rm app prepare-dataset --source /pybullet_hsr/annotation_data --latest --symlink
docker compose run --rm app train --dataset datasets/<dataset_name>/data.yaml --fast
docker compose run --rm app test tests/backend/ -v
```

### Dataset freshness

`prepare_dataset.py` writes a `.prepare_meta.json` sidecar to the
destination. `sync_latest.py` / `./start.sh sync` check that sidecar
against the newest dump and exit early when the local dataset is still
in sync, so you can wire these into any training workflow without
re-doing the split every time. Pass `--force` to rebuild.

### Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `PYBULLET_HSR_ROOT` | `/home/roboworks/repos/pybullet_hsr` (host) / `/pybullet_hsr` (Docker) | Path to pybullet_hsr clone (mounted read-only into the container) |

### References

- [pybullet_hsr](../pybullet_hsr) — dataset source
- [RoboCup@Home Rulebook](https://github.com/RoboCupAtHome/RuleBook)
- [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/)
