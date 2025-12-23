# Benchmark Tests

パフォーマンス重視のコンポーネントのベンチマークテストです。

## 概要

このディレクトリには、合成画像生成などの重要な処理のパフォーマンスを測定するためのベンチマークテストが含まれています。

## 実行方法

### Docker経由で実行（推奨）

```bash
# すべてのベンチマークテストを実行
docker compose run --rm hsr-perception pytest tests/benchmark/ -v

# 特定のテストのみ実行
docker compose run --rm hsr-perception pytest tests/benchmark/test_synthetic_performance.py::TestSyntheticPerformance::test_sequential_generation -v

# 詳細な出力を表示
docker compose run --rm hsr-perception pytest tests/benchmark/ -v -s
```

### ローカル環境で実行

```bash
# 仮想環境を有効化
source venv/perception/bin/activate

# テスト実行
pytest tests/benchmark/ -v
```

## テストの種類

### `test_synthetic_performance.py`

合成画像生成のパフォーマンステスト:

- **test_sequential_generation**: 順次実行のベースライン測定
- **test_parallel_generation_2_workers**: 2ワーカーでの並列実行（未実装のためスキップ）
- **test_parallel_generation_4_workers**: 4ワーカーでの並列実行（未実装のためスキップ）
- **test_speedup_ratio**: 順次vs並列のスピードアップ比較（未実装のためスキップ）
- **test_memory_efficiency**: メモリ使用効率のテスト
- **test_scalability**: バッチサイズによるスケーラビリティテスト

### `TestGenerationQuality`

生成画像の品質テスト:

- **test_generated_images_exist**: 画像とラベルが正しく生成されるかテスト
- **test_label_format_valid**: YOLOフォーマットのバリデーション

## 並列実行について

現在、合成画像生成は順次実行のみサポートしています。並列実行の実装が完了すると、以下のテストが有効になります:

- `test_parallel_generation_2_workers`
- `test_parallel_generation_4_workers`
- `test_speedup_ratio`

期待されるスピードアップ:
- 2ワーカー: 1.5-1.8倍
- 4ワーカー: 2.5-3.0倍

## 結果の解釈

### パフォーマンスメトリクス

- **Elapsed time**: 処理全体にかかった時間
- **Images/second**: 1秒あたりの生成画像数
- **Speedup**: ベースライン（順次実行）に対する倍率
- **Memory per image**: 1画像あたりのメモリ使用量

### ベンチマークの目安

小さなテスト画像（100x100px）の場合:
- 順次実行: 0.1-0.2秒/画像
- メモリ使用: < 5MB/画像
- スケーラビリティ: バッチサイズに対してほぼ線形

## トラブルシューティング

### メモリ不足エラー

テストデータのサイズを減らすか、バッチサイズを調整してください:

```python
# conftest.pyの画像サイズを調整
img = np.zeros((50, 50, 3), dtype=np.uint8)  # 100x100 -> 50x50
```

### テストが遅すぎる

ダミー画像のサイズと数を減らすことを検討してください:

```python
# conftest.py
for i in range(3):  # 5 -> 3に減らす
    ...
```

## CI/CDでの使用

ベンチマークテストは時間がかかるため、通常のCIパイプラインからは分離することを推奨します:

```yaml
# .github/workflows/benchmark.yml
- name: Run benchmarks
  run: |
    docker compose run --rm hsr-perception pytest tests/benchmark/ -v --benchmark-only
```

## 今後の拡張

- [ ] 並列実行の実装と対応するテストの有効化
- [ ] GPUアクセラレーションのベンチマーク
- [ ] 異なる画像サイズでのパフォーマンステスト
- [ ] メモリプロファイリングの詳細化
- [ ] 継続的なパフォーマンス追跡（ベースラインとの比較）
