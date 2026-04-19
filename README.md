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
| xtion-live (任意) | `scripts/evaluation/xtion_live_infer.py` | 学習済み `.pt` を ROS2 画像トピック（Xtion）にかけてライブで閲覧する PyQt6 ビューア |

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

### Xtion ライブ推論（学習後の実機検証）

学習済みモデル（`.pt`）を、ROS2 経由で Xtion PRO LIVE の画像トピックに
適用し、検出結果を重ねた映像を PyQt6 ウィンドウでライブ閲覧できます。
トピック選択 / 信頼度スライダ / FPS・検出リスト表示付き。

ROS2 Humble + OpenNI2 + PyQt6 は学習用イメージ `hsr-perception:latest`
に直接同梱しており、別イメージを建てる必要はありません。

#### ホスト初回セットアップ（1 回だけ）

```bash
# Xtion PRO LIVE を video グループで触れるようにする udev ルール
sudo cp docker/99-xtion.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
sudo usermod -aG video $USER          # 反映には再ログインが必要

# X11 フォワーディング用（xhost は start.sh が自動で叩く）
sudo apt-get install -y x11-xserver-utils
```

#### 起動手順

PC + Xtion だけで完結する場合はコマンド 1 発:

```bash
./start.sh xtion-live -- \
    --model models/finetuned/<run>/weights/best.pt --conf 0.25
```

コンテナ内で `openni2_camera` publisher がバックグラウンド起動し、
トピック (`/camera/rgb/image_raw`) が advertise されてから
PyQt6 ビューアが立ち上がります。GUI 終了と同時に publisher も停止します。

HSR や別マシンで既に publish している場合は `--no-camera` を付けて
publisher 起動をスキップ:

```bash
./start.sh xtion-live -- --no-camera \
    --model models/finetuned/<run>/weights/best.pt --conf 0.25
```

GUI が立ち上がったら、**Refresh** → トピック（例 `/camera/rgb/image_raw`）
を選択 → **Subscribe** で購読を開始。信頼度はスライダで調整できます。

`./start.sh xtion-live` の中身：

- 学習用と同じ `hsr-perception:latest` を再利用（追加ビルドなし）
- `xhost +local:docker` で X11 を開放
- `docker compose run --rm app xtion-live -- …` を実行
  （`network_mode: host`、USB passthrough、GPU、X11 ソケットマウント付き）

#### ホスト直接実行（Docker を使わない場合）

ホストに ROS2 Humble + PyQt6 + ultralytics が入っていれば、そのまま実行可能:

```bash
source /opt/ros/humble/setup.bash
python scripts/evaluation/xtion_live_infer.py \
    --model models/finetuned/<run>/weights/best.pt --conf 0.25
```

### ディレクトリ構成

```
hsr-perception-robocup/
├── app/                    # Streamlit UI（Dashboard / Training / Evaluation / Settings）
├── scripts/
│   ├── data/               # pybullet_hsr dump → YOLO dataset 変換
│   ├── training/           # YOLOv8 fine-tuning
│   ├── evaluation/         # 評価ツール（mAP 評価 + Xtion ライブ推論ビューア）
│   └── common/             # 共通ユーティリティ
├── docker/                 # Dockerfile（ROS2 同梱）+ entrypoint + 99-xtion.rules
├── config/                 # fastdds_profile.xml など
├── models/finetuned/       # 学習結果
├── datasets/               # 準備済み YOLO データセット
├── runs/                   # TensorBoard ログ
└── tests/                  # pytest（backend + frontend）
```

### 環境変数

| 変数 | 既定値 | 用途 |
|------|--------|------|
| `PYBULLET_HSR_ROOT` | `/home/roboworks/repos/pybullet_hsr` (ホスト) / `/pybullet_hsr` (Docker) | pybullet_hsr クローンへのパス |
| `ROS_DOMAIN_ID` | `0` | `xtion-live` サービスが Xtion パブリッシャを発見するための ROS2 DDS ドメイン |
| `DISPLAY` / `XAUTHORITY` | ホストから継承 | `xtion-live` サービスがホストの X サーバに PyQt6 ウィンドウを出すために必要 |

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
| xtion-live (optional) | `scripts/evaluation/xtion_live_infer.py` | PyQt6 viewer that runs the trained `.pt` on a ROS2 image topic (Xtion) live |

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

### Xtion live inference

Once a model is trained, visualise it on a real Xtion PRO LIVE stream.
`scripts/evaluation/xtion_live_infer.py` subscribes to a ROS2
`sensor_msgs/Image` topic, runs the `.pt` weights on every frame, and
renders the annotated video in a PyQt6 window (topic selector,
confidence slider, FPS counter, per-detection list).

ROS2 Humble + OpenNI2 + PyQt6 are bundled directly into
`hsr-perception:latest`, so no separate overlay image is needed.

```bash
# One-time host setup
sudo cp docker/99-xtion.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
sudo usermod -aG video $USER         # logout/login to take effect
sudo apt-get install -y x11-xserver-utils

# PC + Xtion (default): one command; publisher and viewer run in the
# same container, cleaned up together on exit.
./start.sh xtion-live -- \
    --model models/finetuned/<run>/weights/best.pt --conf 0.25

# HSR / external publisher: skip the in-container openni2_camera.
./start.sh xtion-live -- --no-camera \
    --model models/finetuned/<run>/weights/best.pt --conf 0.25
```

`start.sh xtion-live` runs `xhost +local:docker` and invokes
`docker compose run --rm app xtion-live -- …` against the already-built
`hsr-perception:latest` (`network_mode: host`, USB passthrough, GPU, X11
mounted).

Host path (no Docker) — if ROS2 Humble + PyQt6 are already installed:

```bash
source /opt/ros/humble/setup.bash
python scripts/evaluation/xtion_live_infer.py \
    --model models/finetuned/<run>/weights/best.pt --conf 0.25
```

### Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `PYBULLET_HSR_ROOT` | `/home/roboworks/repos/pybullet_hsr` (host) / `/pybullet_hsr` (Docker) | Path to pybullet_hsr clone (mounted read-only into the container) |
| `ROS_DOMAIN_ID` | `0` | ROS2 DDS domain used by `xtion-live` / `ros2-camera` to discover publishers |
| `DISPLAY` / `XAUTHORITY` | inherited | Needed by `xtion-live` to open the PyQt6 window on the host X server |

### References

- [pybullet_hsr](../pybullet_hsr) — dataset source
- [RoboCup@Home Rulebook](https://github.com/RoboCupAtHome/RuleBook)
- [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/)
