# Benchmark Quick Start Guide

すぐに始めるためのクイックガイドです。

## 最速スタート（30秒）

```bash
# テストを実行
docker compose run --rm hsr-perception pytest tests/benchmark/ -v -s
```

これだけで全てのベンチマークテストが実行されます。

## よく使うコマンド

### 1. 順次実行のベンチマーク（最も基本）

```bash
docker compose run --rm hsr-perception pytest \
    tests/benchmark/test_synthetic_performance.py::TestSyntheticPerformance::test_sequential_generation \
    -v -s
```

**期待される出力:**
```
Target images:    20
Generated:        20
Elapsed time:     2-5 seconds
Images/second:    4-10
```

### 2. メモリ効率テスト

```bash
docker compose run --rm hsr-perception pytest \
    tests/benchmark/test_synthetic_performance.py::TestSyntheticPerformance::test_memory_efficiency \
    -v -s
```

**期待される出力:**
```
Images generated:  10
Peak memory delta: 10-30 MB
Memory per image:  1-3 MB
```

### 3. スタンドアロンスクリプト（結果をJSON保存）

```bash
docker compose run --rm hsr-perception python scripts/benchmark/benchmark_synthetic.py \
    --num-images 50 \
    --save-results results.json
```

## 出力の見方

### テスト成功の例

```
tests/benchmark/test_synthetic_performance.py::TestSyntheticPerformance::test_sequential_generation PASSED

============================================================
SEQUENTIAL GENERATION BENCHMARK
============================================================
Target images:    20
Generated:        20          ✅ 全て生成成功
Failed:           0           ✅ 失敗なし
Elapsed time:     3.452 seconds
Images/second:    5.79        ✅ 合理的な速度
Avg objects/img:  2.35        ✅ 適切な密度
============================================================
```

### テスト失敗の例

```
tests/benchmark/test_synthetic_performance.py::TestSyntheticPerformance::test_sequential_generation FAILED

AssertionError: Should generate at least some images
```

**対処法**:
- テストデータが正しく生成されているか確認
- ログを確認: `pytest tests/benchmark/ -v -s --log-cli-level=DEBUG`

## 並列テストについて

現在、以下のテストは **スキップ** されます:

```
test_parallel_generation_2_workers - SKIPPED (並列未実装)
test_parallel_generation_4_workers - SKIPPED (並列未実装)
test_speedup_ratio - SKIPPED (並列未実装)
```

これは正常な動作です。並列実装が完了すると自動的に有効化されます。

## トラブルシューティング

### 問題: "ModuleNotFoundError: No module named 'augmentation'"

**解決法:**
```bash
# Dockerイメージを再ビルド
docker compose build
```

### 問題: テストが非常に遅い（30秒以上）

**確認事項:**
- HDDではなくSSDを使用していますか？
- Dockerのリソース制限を確認（メモリ、CPU）

**一時的な対処:**
```bash
# 画像数を減らす（conftest.pyを編集）
# または、特定のテストのみ実行
pytest tests/benchmark/test_synthetic_performance.py::TestGenerationQuality -v -s
```

### 問題: メモリ不足エラー

**解決法:**
```bash
# Dockerのメモリ制限を増やす
# または、テストデータのサイズを減らす（conftest.py編集）
```

### 問題: すべてのテストがスキップされる

**確認:**
```bash
# pytestのバージョン確認
docker compose run --rm hsr-perception pytest --version

# 明示的にテストを指定
pytest tests/benchmark/test_synthetic_performance.py -v
```

## 次のステップ

- 📖 詳細な使用例: `tests/benchmark/USAGE_EXAMPLES.md`
- 📖 完全なドキュメント: `tests/benchmark/README.md`
- 🔧 スクリプトの使い方: `scripts/benchmark/README.md`
- 📊 実装の詳細: `BENCHMARK_IMPLEMENTATION_SUMMARY.md`

## 質問とサポート

よくある質問:

**Q: ベンチマークはどのくらいの頻度で実行すべき？**
A: 大きな変更の前後、リリース前、週次で定期実行を推奨

**Q: 結果をどこに保存すべき？**
A: Gitには含めず、CI/CDのアーティファクトとして保存するか、別のストレージに保存

**Q: 並列実装はいつ追加される？**
A: 優先度に応じて実装予定。テストコードは準備済みなので、実装後すぐに有効化できます

## チートシート

```bash
# 全テスト実行
docker compose run --rm hsr-perception pytest tests/benchmark/ -v -s

# 順次実行のみ
docker compose run --rm hsr-perception pytest tests/benchmark/ -v -s -k "sequential"

# メモリテストのみ
docker compose run --rm hsr-perception pytest tests/benchmark/ -v -s -k "memory"

# 品質テストのみ
docker compose run --rm hsr-perception pytest tests/benchmark/ -v -s -k "Quality"

# スタンドアロンスクリプト
docker compose run --rm hsr-perception python scripts/benchmark/benchmark_synthetic.py

# カスタムデータで実行
docker compose run --rm hsr-perception python scripts/benchmark/benchmark_synthetic.py \
    --backgrounds-dir /workspace/data/bg \
    --annotated-dir /workspace/data/annotated \
    --num-images 100
```

Happy Benchmarking! 🚀
