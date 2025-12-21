# Registry - Object Registration

System for object registration and management, collection progress tracking, and reference image management.

---

## Related Files

| File | Description |
|------|-------------|
| `app/object_registry.py` | ObjectRegistry class (core implementation) |
| `app/pages/2_Registry.py` | Streamlit UI page |
| `config/object_classes.json` | Class definition template |

---

## Technologies Used

- **Python dataclasses** - Data structure definition
- **JSON** - Persistence format
- **Streamlit** - Web UI

---

## Data Structures

### RegisteredObject

```python
@dataclass
class RegisteredObject:
    id: int                      # Unique ID (starting from 1)
    name: str                    # Internal name (no spaces)
    display_name: str            # Display name
    category: str                # Category
    versions: List[ObjectVersion]  # Reference image versions
    properties: ObjectProperties   # Physical attributes
    remarks: str                 # Notes
    target_samples: int          # Target sample count
    collected_samples: int       # Collected sample count
    last_updated: Optional[str]  # Last update timestamp
    thumbnail_path: Optional[str] # Thumbnail path
```

### ObjectProperties

Physical attributes for grasping strategy:

```python
@dataclass
class ObjectProperties:
    is_heavy: bool           # Heavy object (500g or more)
    is_tiny: bool            # Tiny object (less than 2cm)
    has_liquid: bool         # Liquid container
    size_cm: Optional[str]   # Size (e.g., "10x5x3")
    grasp_strategy: Optional[str]  # Grasping strategy notes
```

---

## Main Features

### Object Management

| Method | Description |
|--------|-------------|
| `add_object()` | Register new object (auto-creates directory) |
| `remove_object()` | Delete object |
| `update_object()` | Update properties (handles directory rename) |
| `get_object_by_name()` | Get object by name |
| `get_objects_by_category()` | Filter by category |

### Collection Progress Tracking

| Method | Description |
|--------|-------------|
| `update_collection_count()` | Count images on disk |
| `update_all_collection_counts()` | Batch update all objects |
| `get_collection_stats()` | Get statistics |

### Statistics Format

```python
{
    "total_objects": int,       # Registered object count
    "total_target": int,        # Total target samples
    "total_collected": int,     # Total collected
    "progress_percent": float,  # Progress percentage
    "by_category": {
        "category_name": {
            "target": int,
            "collected": int,
            "objects": int
        }
    },
    "ready_objects": int  # Objects with 50%+ collected
}
```

---

## Directory Structure

```
$DATA_DIR/
├── registry.json          # Object definitions
├── thumbnails/            # Thumbnail images
│   └── <object_name>.jpg
├── reference_images/      # Reference images
│   └── <object_name>/
│       ├── v1.jpg
│       └── v2.jpg
└── raw_captures/          # Collected data
    └── <object_name>/
        └── <object_name>_YYYYMMDD_HHMMSS.jpg
```

---

## YOLO Format Export

Convert to YOLO-compatible format with `export_to_yolo_config()`:

```python
{
    "classes": [
        {"id": 0, "name": "bottle", "category": "container"},
        {"id": 1, "name": "cup", "category": "container"}
    ],
    "nc": 2,  # Number of classes
    "names": ["bottle", "cup"],
    "settings": {
        "train_val_split": 0.85,
        "min_samples_for_training": 50
    }
}
```

---

## Usage

### GUI Application (Recommended)

```bash
# Start Docker
./start.sh
# or
docker compose up
```

Open http://localhost:8501 in your browser and access the Registry page

1. Click "Add Object" to add an object
2. Upload reference images
3. Monitor collection progress

### Programmatic Usage

```python
from app.object_registry import ObjectRegistry

registry = ObjectRegistry(data_dir="path/to/data")

# Add object
registry.add_object(
    name="bottle01",
    display_name="Plastic Bottle",
    category="container",
    target_samples=200
)

# Update collection status
registry.update_all_collection_counts()

# Get statistics
stats = registry.get_collection_stats()
print(f"Progress: {stats['progress_percent']:.1f}%")
```
