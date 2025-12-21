# Annotation - Auto-Annotation

Automatically generate bounding boxes from collected images and create YOLO-format datasets.

---

## Related Files

| File | Description |
|------|-------------|
| `scripts/annotation/auto_annotate.py` | Main pipeline |
| `scripts/annotation/background_subtraction.py` | Background subtraction annotator |
| `scripts/annotation/sam2_annotator.py` | SAM2 annotator |
| `scripts/annotation/sam2_interactive_app.py` | SAM2 interactive GUI |
| `scripts/annotation/annotation_utils.py` | YOLO format conversion utilities |
| `scripts/annotation/prepare_dataset.py` | Dataset preparation |
| `scripts/annotation/video_tracking_predictor.py` | Video tracking |
| `app/pages/4_Annotation.py` | Streamlit UI page |

---

## Technologies Used

- **OpenCV** - Image processing (background subtraction, contour detection)
- **SAM2 (Segment Anything 2)** - Segmentation model
- **PyTorch** - Deep learning framework
- **Tkinter** - Interactive GUI

---

## Annotation Methods

### 1. Background Subtraction Method (Recommended)

**Features:**
- Fast processing
- Requires uniform background (white sheet recommended)
- No GPU required

**Algorithm:**
```
Load background image
    ↓
Grayscale conversion + Gaussian blur
    ↓
Calculate difference from background
    ↓
Thresholding (Otsu/Adaptive/Fixed)
    ↓
Morphological operations (erosion → dilation)
    ↓
Detect largest contour
    ↓
Generate bounding box
    ↓
Save in YOLO format
```

**Configuration Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_contour_area` | 500 | Minimum contour area (pixels) |
| `blur_kernel_size` | 5 | Blur kernel size |
| `threshold_method` | "otsu" | Threshold method (otsu/adaptive/fixed) |
| `morph_kernel_size` | 5 | Morphology kernel size |
| `bbox_margin_ratio` | 0.02 | BBox margin (2%) |

### 2. SAM2 Method (Fallback)

**Features:**
- High-precision segmentation
- Handles complex backgrounds
- GPU required (CUDA)

**Configuration Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_path` | "sam2_b.pt" | SAM2 model path |
| `points_per_side` | 32 | Grid point count |
| `pred_iou_thresh` | 0.88 | IoU threshold |
| `stability_score_thresh` | 0.92 | Stability score threshold |
| `min_mask_region_area` | 100 | Minimum mask area |

### 3. SAM2 Interactive (GUI)

**Features:**
- Interactive segmentation with point clicking
- Improve accuracy with foreground/background points
- Also supports video tracking

**Keyboard Controls:**

| Action | Function |
|--------|----------|
| Left Click | Add foreground point |
| Right Click | Add background point |
| Enter | Confirm and save |
| Escape | Reset points |
| Ctrl+Z | Undo |
| ←/→ | Image navigation |
| Space/M | Toggle mask overlay |
| S | Skip |

---

## Docker Execution

### Command Help

```bash
docker compose run --rm hsr-perception annotate --help
```

### Background Subtraction Method

```bash
docker compose run --rm hsr-perception annotate \
    --method background \
    --background /workspace/datasets/backgrounds/white_sheet.jpg \
    --input-dir /workspace/datasets/raw_captures \
    --output-dir /workspace/datasets/competition_day \
    --class-config /workspace/config/object_classes.json
```

### SAM2 Method

```bash
docker compose run --rm hsr-perception annotate \
    --method sam2 \
    --input-dir /workspace/datasets/raw_captures \
    --output-dir /workspace/datasets/competition_day \
    --class-config /workspace/config/object_classes.json
```

### Main Options

| Option | Description | Default |
|--------|-------------|---------|
| `--method, -m` | Annotation method (background/sam2) | background |
| `--background, -b` | Background image path (required for background method) | - |
| `--input-dir, -i` | Input directory (required) | - |
| `--output-dir, -o` | Output directory (required) | - |
| `--class-config, -c` | Class configuration file | config/object_classes.json |
| `--split` | train/val split ratio | 0.80 |
| `--min-area` | Minimum contour area | 500 |
| `--group-frames` | Frame grouping (prevents data leakage) | Enabled |
| `--no-verify` | Skip annotation verification | - |

---

## Output Format

### YOLO Format Labels (.txt)

```
<class_id> <x_center> <y_center> <width> <height>
```

- All coordinates are normalized to 0-1
- One object per line
- Filename matches image (e.g., image.jpg → image.txt)

### Directory Structure

```
output_dir/
├── images/
│   ├── train/     # Training images (80%)
│   └── val/       # Validation images (20%)
├── labels/
│   ├── train/     # Training labels
│   └── val/       # Validation labels
├── data.yaml      # YOLO dataset configuration
└── annotation_report.json  # Processing report
```

### data.yaml Format

```yaml
path: /absolute/path/to/dataset
train: images/train
val: images/val
nc: 5                    # Number of classes
names:
  0: bottle
  1: cup
  2: box
  3: snack
  4: fruit
```

---

## Annotation Report

```json
{
  "timestamp": "2024-12-08 10:30:45",
  "method": "background",
  "total_classes": 5,
  "total_images": 250,
  "successful": 242,
  "failed": 8,
  "success_rate": 96.8,
  "train_count": 205,
  "val_count": 37,
  "class_results": {
    "bottle": {
      "total": 50,
      "successful": 48,
      "failed": 2,
      "success_rate": 96.0
    }
  }
}
```

---

## Coordinate Conversion

### YOLO → Pixel Coordinates

```python
x_min = (x_center - width/2) * img_width
y_min = (y_center - height/2) * img_height
x_max = (x_center + width/2) * img_width
y_max = (y_center + height/2) * img_height
```

### Pixel Coordinates → YOLO

```python
x_center = (x_min + x_max) / 2 / img_width
y_center = (y_min + y_max) / 2 / img_height
width = (x_max - x_min) / img_width
height = (y_max - y_min) / img_height
```
