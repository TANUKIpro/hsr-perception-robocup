# Benchmark Usage Examples

このファイルには、ベンチマークテストとスクリプトの実際の使用例を示します。

## クイックスタート

### 1. 最も簡単な方法（自動生成データ）

```bash
# Docker経由で全てのベンチマークテストを実行
docker compose run --rm hsr-perception pytest tests/benchmark/ -v -s

# または、スタンドアロンスクリプトを実行
docker compose run --rm hsr-perception python scripts/benchmark/benchmark_synthetic.py
```

これにより、テストデータが自動生成され、ベンチマークが実行されます。

### 2. 特定のテストのみ実行

```bash
# 順次実行のベンチマークのみ
docker compose run --rm hsr-perception pytest \
    tests/benchmark/test_synthetic_performance.py::TestSyntheticPerformance::test_sequential_generation \
    -v -s

# メモリ効率テストのみ
docker compose run --rm hsr-perception pytest \
    tests/benchmark/test_synthetic_performance.py::TestSyntheticPerformance::test_memory_efficiency \
    -v -s

# スケーラビリティテストのみ
docker compose run --rm hsr-perception pytest \
    tests/benchmark/test_synthetic_performance.py::TestSyntheticPerformance::test_scalability \
    -v -s
```

### 3. 品質テストのみ実行

```bash
docker compose run --rm hsr-perception pytest \
    tests/benchmark/test_synthetic_performance.py::TestGenerationQuality \
    -v -s
```

## 実際のデータを使用したベンチマーク

### データセットが既にある場合

```bash
# 既存のデータでベンチマーク実行
docker compose run --rm hsr-perception python scripts/benchmark/benchmark_synthetic.py \
    --backgrounds-dir /workspace/data/competition_2024/backgrounds \
    --annotated-dir /workspace/data/competition_2024/annotated \
    --num-images 100 \
    --output-dir /workspace/benchmarks/competition_2024
```

### 特定のクラスのみテスト

```bash
docker compose run --rm hsr-perception python scripts/benchmark/benchmark_synthetic.py \
    --annotated-dir /workspace/data/annotated \
    --class-names apple banana orange \
    --num-images 50
```

## 結果の保存と比較

### 結果をタイムスタンプ付きで保存

```bash
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

docker compose run --rm hsr-perception python scripts/benchmark/benchmark_synthetic.py \
    --num-images 50 \
    --output-dir benchmarks/$TIMESTAMP \
    --save-results benchmarks/$TIMESTAMP/results.json
```

### 複数回実行して平均を取る

```bash
#!/bin/bash
# run_multiple_benchmarks.sh

for i in {1..5}; do
    echo "Run $i/5"
    docker compose run --rm hsr-perception python scripts/benchmark/benchmark_synthetic.py \
        --num-images 20 \
        --output-dir benchmarks/run_$i \
        --save-results benchmarks/run_$i/results.json
done

# 結果を集計
python scripts/benchmark/aggregate_results.py benchmarks/run_*/results.json
```

## CI/CDでの使用

### GitHub Actions例

```yaml
name: Performance Benchmarks

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]
  schedule:
    # 毎週日曜日の深夜に実行
    - cron: '0 0 * * 0'

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Build Docker image
        run: docker compose build

      - name: Run benchmarks
        run: |
          docker compose run --rm hsr-perception \
            pytest tests/benchmark/ -v --json-report --json-report-file=benchmark_results.json

      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: benchmark-results
          path: benchmark_results.json

      - name: Compare with baseline
        run: |
          # 過去の結果と比較（カスタムスクリプト）
          python scripts/benchmark/compare_with_baseline.py \
            benchmark_results.json \
            baseline/benchmark_results.json
```

## 開発中のパフォーマンステスト

### 変更前後の比較

```bash
# 変更前のベンチマーク
git checkout main
docker compose build
docker compose run --rm hsr-perception python scripts/benchmark/benchmark_synthetic.py \
    --num-images 50 \
    --save-results benchmarks/before.json

# 変更後のベンチマーク
git checkout feature/my-optimization
docker compose build
docker compose run --rm hsr-perception python scripts/benchmark/benchmark_synthetic.py \
    --num-images 50 \
    --save-results benchmarks/after.json

# 結果を比較
python -c "
import json
with open('benchmarks/before.json') as f: before = json.load(f)
with open('benchmarks/after.json') as f: after = json.load(f)

before_time = before['sequential']['elapsed_time']
after_time = after['sequential']['elapsed_time']
speedup = before_time / after_time

print(f'Before: {before_time:.3f}s')
print(f'After:  {after_time:.3f}s')
print(f'Speedup: {speedup:.2f}x ({(speedup-1)*100:.1f}% improvement)')
"
```

## プロファイリングとの組み合わせ

### cProfileでボトルネックを特定

```bash
docker compose run --rm hsr-perception python -m cProfile -o profile.stats \
    scripts/benchmark/benchmark_synthetic.py --num-images 20

# プロファイル結果を表示
docker compose run --rm hsr-perception python -c "
import pstats
p = pstats.Stats('profile.stats')
p.sort_stats('cumulative')
p.print_stats(20)
"
```

### line_profilerで行単位の分析

```bash
# 事前にline_profilerをインストール
pip install line_profiler

# プロファイル対象の関数に@profileデコレータを追加してから
kernprof -l -v scripts/benchmark/benchmark_synthetic.py --num-images 10
```

## メモリプロファイリング

### memory_profilerを使用

```bash
# memory_profilerをインストール
pip install memory_profiler

# メモリ使用量を測定
docker compose run --rm hsr-perception python -m memory_profiler \
    scripts/benchmark/benchmark_synthetic.py --num-images 20
```

## 異なる設定での比較

### 画像サイズの影響を測定

```python
# custom_benchmark.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.benchmark.benchmark_synthetic import run_benchmark, create_test_data

# 異なる画像サイズでテスト
for size in [64, 128, 256, 512]:
    print(f"\n{'='*60}")
    print(f"Testing with {size}x{size} images")
    print(f"{'='*60}")

    # カスタムテストデータ作成（サイズ指定）
    # ...実装...
```

## トラブルシューティング

### テストが失敗する場合

```bash
# より詳細なログを表示
docker compose run --rm hsr-perception pytest tests/benchmark/ -v -s --log-cli-level=DEBUG

# 特定のテストのみデバッグ
docker compose run --rm hsr-perception pytest tests/benchmark/ -v -s -k "sequential"
```

### Dockerコンテナに入ってインタラクティブに実行

```bash
# コンテナに入る
docker compose run --rm hsr-perception bash

# コンテナ内で実行
cd /workspace
python scripts/benchmark/benchmark_synthetic.py --num-images 10

# テストも実行可能
pytest tests/benchmark/test_synthetic_performance.py::TestSyntheticPerformance::test_sequential_generation -v -s
```

## 期待される結果

### 小規模テスト（デフォルト設定）

```
Target images:    20
Generated:        20
Failed:           0
Elapsed time:     2-5 seconds (環境による)
Images/second:    4-10 images/sec
Avg objects/img:  2-3 objects
```

### 大規模テスト（100画像）

```
Target images:    100
Generated:        100
Failed:           0-5 (配置失敗による)
Elapsed time:     20-50 seconds
Images/second:    2-5 images/sec
```

### メモリ使用

- 小画像（100x100px）: < 5MB/image
- 中画像（640x480px）: < 20MB/image
- 大画像（1920x1080px）: < 50MB/image

## 注意事項

1. **初回実行は遅い**: Dockerイメージのビルドとダウンロードに時間がかかります
2. **ディスク容量**: ベンチマーク結果は数GB必要な場合があります
3. **並列テストはスキップ**: 現在は順次実行のみサポート
4. **環境依存**: 実行環境（CPU、メモリ、ディスク）により結果は大きく変わります

## さらなる最適化のヒント

1. **SSDの使用**: HDDの場合、I/Oがボトルネックになります
2. **tmpfsの活用**: 一時ファイルをメモリ上に置く
3. **画像圧縮品質**: JPEGの品質を下げると高速化（品質とのトレードオフ）
4. **バッチサイズ**: 小さいバッチで試してから大きくする
5. **並列実行**: 実装後は並列化で2-3倍の高速化が期待できます
