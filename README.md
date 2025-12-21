# HSR Perception Pipeline for RoboCup@Home

RoboCup@Home大会向けのHSR（Human Support Robot）用物体認識パイプライン。

**環境**: Ubuntu 22.04 / ROS2 Humble / Python

---

## パイプライン概要

```
Collection → Annotation → Training → Evaluation → Deploy
```

| ステップ | 説明 |
|---------|------|
| Collection | 画像収集（ファイルアップロード / ROS2カメラ / 動画抽出） |
| Annotation | SAM2による自動アノテーション |
| Training | YOLOv8 fine-tuning（GPU自動スケーリング対応） |
| Evaluation | mAP・推論速度の検証 |
| Deploy | HSRへのモデル配備 |

---

## Docker実行

すべての機能はDocker上で動作します。

### 前提条件

| 項目 | 要件 |
|------|------|
| Docker | 24.0以上 |
| Docker Compose | v2.0以上 |
| NVIDIA Driver | 525以上 |
| NVIDIA Container Toolkit | インストール済み |

### クイックスタート

```bash
# 起動（初回はイメージビルド・udevルール設定を自動実行）
./start.sh

# ブラウザで http://localhost:8501 を開く
```

**オプション:**
```bash
./start.sh --build        # イメージを強制再ビルド
./start.sh --tensorboard  # TensorBoard付きで起動（ポート6006）
./start.sh -d             # バックグラウンド起動
```

### 主要コマンド

| コマンド | 説明 |
|---------|------|
| `docker compose up` | Streamlit UI起動 |
| `docker compose down` | 停止 |
| `docker compose run --rm hsr-perception bash` | シェルアクセス |

### 含まれるコンポーネント

| コンポーネント | バージョン |
|---------------|-----------|
| Python | 3.10 |
| PyTorch | 2.x（CUDA 12.1対応） |
| Ultralytics | >=8.3.0（YOLOv8） |
| SAM2 | latest |
| ROS2 | Humble |
| Streamlit | >=1.28.0 |

---

## 使い方

起動後、ブラウザで http://localhost:8501 を開くとStreamlit GUIにアクセスできます。

**ページ構成:**
| ページ | 機能 |
|--------|------|
| Dashboard | パイプライン全体の進捗・状態可視化 |
| Registry | オブジェクト登録・参照画像管理 |
| Collection | データ収集（ROS2/ファイル/動画抽出） |
| Annotation | 自動アノテーション実行 |
| Training | YOLOv8 fine-tuning・進捗監視 |
| Evaluation | モデル評価・可視化テスト |

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
├── scripts/                # MLパイプラインスクリプト
│   ├── annotation/         # 自動アノテーション
│   ├── training/           # 学習パイプライン
│   └── evaluation/         # 評価ツール
│
├── src/hsr_perception/     # ROS2パッケージ
│
├── config/                 # 設定ファイル
├── profiles/               # プロファイルデータ
├── models/                 # 学習済みモデル
└── datasets/               # データセット
```

各ファイルの詳細は [docs/implementation.md](docs/implementation.md) を参照。

---

## 技術スタック

| カテゴリ | 技術 |
|---------|------|
| 物体検出 | YOLOv8 (Ultralytics) |
| セグメンテーション | SAM2 (Meta) |
| GUI | Streamlit |
| ロボティクス | ROS2 Humble |
| コンテナ | Docker + Docker Compose |

---

## テスト

```bash
# 全テスト実行
docker compose run --rm hsr-perception test

# Backendテストのみ
docker compose run --rm hsr-perception test tests/backend/ -v

# E2Eテスト（Playwright）
cd tests/e2e && npm test
```

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
