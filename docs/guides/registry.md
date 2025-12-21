# Registry - オブジェクト登録

オブジェクトの登録・管理、収集進捗のトラッキング、参照画像の管理を行うシステム。

---

## 関連ファイル

| ファイル | 説明 |
|---------|------|
| `app/object_registry.py` | ObjectRegistryクラス（コア実装） |
| `app/pages/2_Registry.py` | Streamlit UIページ |
| `config/object_classes.json` | クラス定義テンプレート |

---

## 使用技術

- **Python dataclasses** - データ構造定義
- **JSON** - 永続化フォーマット
- **Streamlit** - Web UI

---

## データ構造

### RegisteredObject

```python
@dataclass
class RegisteredObject:
    id: int                      # 一意のID（1始まり）
    name: str                    # 内部名（スペースなし）
    display_name: str            # 表示名
    category: str                # カテゴリ
    versions: List[ObjectVersion]  # 参照画像バージョン
    properties: ObjectProperties   # 物理属性
    remarks: str                 # 備考
    target_samples: int          # 目標サンプル数
    collected_samples: int       # 収集済みサンプル数
    last_updated: Optional[str]  # 最終更新日時
    thumbnail_path: Optional[str] # サムネイルパス
```

### ObjectProperties

把持戦略に関する物理属性:

```python
@dataclass
class ObjectProperties:
    is_heavy: bool           # 重量物（500g以上）
    is_tiny: bool            # 小型物体（2cm未満）
    has_liquid: bool         # 液体容器
    size_cm: Optional[str]   # サイズ（例: "10x5x3"）
    grasp_strategy: Optional[str]  # 把持戦略メモ
```

---

## 主要機能

### オブジェクト管理

| メソッド | 説明 |
|---------|------|
| `add_object()` | 新規オブジェクト登録（ディレクトリ自動作成） |
| `remove_object()` | オブジェクト削除 |
| `update_object()` | プロパティ更新（ディレクトリ名変更対応） |
| `get_object_by_name()` | 名前でオブジェクト取得 |
| `get_objects_by_category()` | カテゴリでフィルタ |

### 収集進捗トラッキング

| メソッド | 説明 |
|---------|------|
| `update_collection_count()` | ディスク上の画像をカウント |
| `update_all_collection_counts()` | 全オブジェクトを一括更新 |
| `get_collection_stats()` | 統計情報取得 |

### 統計情報の形式

```python
{
    "total_objects": int,       # 登録オブジェクト数
    "total_target": int,        # 目標サンプル合計
    "total_collected": int,     # 収集済み合計
    "progress_percent": float,  # 進捗率
    "by_category": {
        "カテゴリ名": {
            "target": int,
            "collected": int,
            "objects": int
        }
    },
    "ready_objects": int  # 50%以上収集済みのオブジェクト数
}
```

---

## ディレクトリ構造

```
$DATA_DIR/
├── registry.json          # オブジェクト定義
├── thumbnails/            # サムネイル画像
│   └── <object_name>.jpg
├── reference_images/      # 参照画像
│   └── <object_name>/
│       ├── v1.jpg
│       └── v2.jpg
└── raw_captures/          # 収集データ
    └── <object_name>/
        └── <object_name>_YYYYMMDD_HHMMSS.jpg
```

---

## YOLO形式エクスポート

`export_to_yolo_config()` でYOLO互換形式に変換:

```python
{
    "classes": [
        {"id": 0, "name": "bottle", "category": "container"},
        {"id": 1, "name": "cup", "category": "container"}
    ],
    "nc": 2,  # クラス数
    "names": ["bottle", "cup"],
    "settings": {
        "train_val_split": 0.85,
        "min_samples_for_training": 50
    }
}
```

---

## 使用方法

### GUIアプリ（推奨）

```bash
# Docker起動
./start.sh
# または
docker compose up
```

ブラウザで http://localhost:8501 を開き、Registryページにアクセス

1. 「Add Object」でオブジェクト追加
2. 参照画像をアップロード
3. 収集進捗を確認

### プログラムから

```python
from app.object_registry import ObjectRegistry

registry = ObjectRegistry(data_dir="path/to/data")

# オブジェクト追加
registry.add_object(
    name="bottle01",
    display_name="ペットボトル",
    category="container",
    target_samples=200
)

# 収集状況を更新
registry.update_all_collection_counts()

# 統計取得
stats = registry.get_collection_stats()
print(f"進捗: {stats['progress_percent']:.1f}%")
```
