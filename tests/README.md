# HSR Perception Pipeline - テストガイド

## テスト実行環境

**重要**: 全てのテストはDocker コンテナ内で実行してください。
ホスト環境ではROS2プラグインとの競合が発生する可能性があります。

## テスト実行方法

### バックエンドテスト（推奨）

```bash
# バックエンド全体のテスト
docker compose run --rm hsr-perception test tests/backend/ -v

# 特定のテストファイルのみ
docker compose run --rm hsr-perception test tests/backend/common/test_validation.py -v

# 特定のテストクラスのみ
docker compose run --rm hsr-perception test tests/backend/common/test_validation.py::TestValidationResult -v

# キーワードでフィルタ
docker compose run --rm hsr-perception test tests/backend/ -k "config" -v
```

### フロントエンドテスト

```bash
# フロントエンドテスト
docker compose run --rm hsr-perception test tests/frontend/ -v
```

### E2Eテスト（Playwright）

```bash
# Playwright E2Eテスト
docker compose run --rm hsr-perception test tests/e2e/ -v
```

### 全テスト実行

```bash
# 全テスト
docker compose run --rm hsr-perception test tests/ -v
```

## テストの構造

```
tests/
├── conftest.py          # 共通フィクスチャ
├── pytest.ini           # pytest設定（ルートディレクトリ）
├── backend/             # バックエンドテスト
│   ├── conftest.py      # バックエンド用フィクスチャ（torch, cv2モック等）
│   ├── common/          # 共通ユーティリティテスト
│   ├── annotation/      # アノテーションモジュールテスト
│   ├── training/        # トレーニングモジュールテスト
│   ├── augmentation/    # データ拡張モジュールテスト
│   └── BACKEND_TEST_CASES.md  # テスト計画
├── frontend/            # フロントエンド（Streamlit）テスト
└── e2e/                 # E2E（Playwright）テスト
```

## カバレッジレポート

```bash
# カバレッジ付きで実行
docker compose run --rm hsr-perception test tests/backend/ --cov=scripts --cov-report=html -v

# HTMLレポートは htmlcov/ に生成されます
```

## トラブルシューティング

### ROS2プラグインエラー
ホスト環境で以下のエラーが発生した場合:
```
PluginValidationError: unknown hook 'pytest_launch_collect_makemodule'
```

**解決方法**: Docker内でテストを実行してください。

### GPUが必要なテスト
GPU関連のテストはモック化されており、GPUなしでも実行可能です。

### 一時ファイルエラー
テストは `tmp_path` フィクスチャを使用するため、一時ディレクトリの
パーミッション問題は発生しません。
