# Evaluation

Model accuracy (mAP) and inference speed evaluation. Supports visual verification, robustness testing, and real-time testing.

---

## Related Files

| File | Description |
|------|-------------|
| `scripts/evaluation/evaluate_model.py` | CLI evaluation tool |
| `scripts/evaluation/visual_verification.py` | Visual verification tool |
| `scripts/evaluation/xtion_test_app.py` | Xtion real-time test |
| `app/pages/6_Evaluation.py` | Streamlit UI page |
| `app/components/robustness_test.py` | Robustness test |
| `app/components/robustness_augmentation.py` | Robustness augmentation |

---

## Technologies Used

- **YOLOv8 (Ultralytics)** - Inference and evaluation
- **OpenCV** - Image processing and visualization
- **Streamlit** - Web UI
- **Tkinter** - Real-time test GUI
- **Plotly** - Graph display

---

## Competition Requirements

| Metric | Target | Description |
|--------|--------|-------------|
| mAP@50 | >= 85% | Mean average precision at IoU=0.50 |
| Inference Time | <= 100ms | Ensure real-time capability |

---

## Evaluation Metrics

### Overall Metrics

| Metric | Description |
|--------|-------------|
| mAP@50 | Mean average precision at IoU=0.50 |
| mAP@50-95 | COCO standard (IoU=0.50-0.95) |
| Precision | TP/(TP+FP) |
| Recall | TP/(TP+FN) |

### Per-Class Metrics

| Metric | Description |
|--------|-------------|
| AP@50 | Per-class average precision |
| F1 Score | Harmonic mean of precision and recall |
| Sample Count | Number of test images |

### Inference Time

- Measured 100 times on 640x480 synthetic image
- 10 warm-up runs (excluded)
- Mean +/- standard deviation

---

## Docker Execution

### Model Evaluation

```bash
# Show help
docker compose run --rm hsr-perception evaluate --help

# Evaluate on dataset
docker compose run --rm hsr-perception evaluate \
    --model /workspace/models/finetuned/competition_*/weights/best.pt \
    --dataset /workspace/datasets/competition_day/data.yaml

# Measure inference time only
docker compose run --rm hsr-perception evaluate \
    --model /workspace/models/finetuned/competition_*/weights/best.pt \
    --time-only

# Save report as JSON
docker compose run --rm hsr-perception evaluate \
    --model /workspace/models/finetuned/competition_*/weights/best.pt \
    --dataset /workspace/datasets/competition_day/data.yaml \
    --save-report /workspace/evaluation_report.json
```

### evaluate Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model, -m` | Model path (required) | - |
| `--dataset, -d` | Dataset YAML | - |
| `--image, -i` | Single image path | - |
| `--output, -o` | Output path | - |
| `--conf` | Confidence threshold | 0.25 |
| `--time-only` | Measure inference time only | - |
| `--save-report` | Save report as JSON | - |

### Visual Verification

```bash
# Show help
docker compose run --rm hsr-perception verify --help

# Batch mode (interactively review images)
docker compose run --rm hsr-perception verify \
    --model /workspace/models/finetuned/competition_*/weights/best.pt \
    --batch-dir /workspace/datasets/competition_day/images/val

# Single image
docker compose run --rm hsr-perception verify \
    --model /workspace/models/finetuned/competition_*/weights/best.pt \
    --image /workspace/path/to/image.jpg

# Grid display
docker compose run --rm hsr-perception verify \
    --model /workspace/models/finetuned/competition_*/weights/best.pt \
    --batch-dir /workspace/datasets/competition_day/images/val \
    --grid --grid-cols 3
```

### verify Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model, -m` | Model path (required) | - |
| `--image, -i` | Single image path | - |
| `--batch-dir, -b` | Batch verification directory | - |
| `--output, -o` | Output path/directory | - |
| `--conf` | Confidence threshold | 0.25 |
| `--grid` | Grid display mode | - |
| `--grid-cols` | Grid column count | 3 |
| `--report-samples` | Generate report samples | - |
| `--class-config` | Class configuration JSON | - |

### Keyboard Controls (Batch Mode)

| Key | Function |
|-----|----------|
| n / → | Next image |
| p / ← | Previous image |
| s | Save result |
| q | Quit |

---

## Evaluation Report

### Output Format

```json
{
  "model_path": "models/finetuned/competition_*/weights/best.pt",
  "dataset_path": "datasets/competition_day/data.yaml",
  "overall_map50": 0.87,
  "overall_map50_95": 0.72,
  "overall_precision": 0.89,
  "overall_recall": 0.83,
  "per_class_metrics": {
    "bottle": {
      "ap50": 0.92,
      "ap50_95": 0.78,
      "precision": 0.94,
      "recall": 0.88,
      "f1_score": 0.91,
      "num_samples": 50
    }
  },
  "inference_time_ms": 45.2,
  "inference_time_std": 3.1,
  "num_test_images": 150,
  "meets_requirements": true
}
```

### Console Output Example

```
=== Model Evaluation Report ===

Overall Metrics:
  mAP@50:     87.2% [PASS] (target: >=85%)
  mAP@50-95:  72.1%
  Precision:  89.3%
  Recall:     83.1%

Inference Time:
  Mean:       45.2ms [PASS] (target: <=100ms)
  Std Dev:    3.1ms

Per-Class Results:
+----------+-------+----------+-----------+--------+---------+
| Class    | AP@50 | AP@50-95 | Precision | Recall | Samples |
+----------+-------+----------+-----------+--------+---------+
| bottle   | 92.1% | 78.3%    | 94.2%     | 88.1%  | 50      |
| cup      | 85.3% | 68.2%    | 87.1%     | 82.3%  | 48      |
+----------+-------+----------+-----------+--------+---------+

All requirements met!
```

---

## Robustness Testing

### Test Modes

1. **Real-time Preview** - Adjust transformation parameters with sliders
2. **Batch Test** - Apply effects to multiple images
3. **Similar Object Test** - Check confusion between similar classes

### Supported Transformations

| Transformation | Parameter | Description |
|----------------|-----------|-------------|
| Brightness | -100 to +100 | Brightness change |
| Shadow | 0 to 100% | Shadow addition |
| Occlusion | 0 to 50% | Occlusion simulation |
| Hue Rotation | 0 to 180 | Hue rotation |

---

## Streamlit UI Features

### Tab Layout

| Tab | Function |
|-----|----------|
| Run Evaluation | Execute evaluation on dataset |
| Results History | Past evaluation history |
| Visual Test | Single test with image upload |
| Robustness Test | Robustness testing |
| Xtion Live Test | Launch real-time test |

---

## Troubleshooting

### If mAP Doesn't Meet Target

1. **Check Data** - Annotation accuracy, sample count
2. **Additional Epochs** - Increase patience and retrain
3. **Adjust Augmentation** - Reduce excessive augmentation
4. **Change Model Size** - Try a larger model

### If Inference Time is Too Long

1. **Reduce Model Size** - yolov8m → yolov8s
2. **Reduce Image Size** - 640 → 480
3. **Adjust Batch Size** - Optimize GPU utilization
