# Training

YOLOv8 model fine-tuning. Supports GPU auto-optimization, OOM recovery, and TensorBoard monitoring.

---

## Related Files

| File | Description |
|------|-------------|
| `scripts/training/quick_finetune.py` | Main training script |
| `scripts/training/gpu_scaler.py` | GPU auto-scaling |
| `scripts/training/training_config.py` | Training configuration management |
| `scripts/training/tensorboard_monitor.py` | TensorBoard monitoring |
| `app/pages/5_Training.py` | Streamlit UI page |
| `app/components/training_charts.py` | Training graph display |
| `app/services/task_runners/run_training.py` | Task execution wrapper |

---

## Technologies Used

- **YOLOv8 (Ultralytics)** - Object detection model
- **PyTorch** - Deep learning framework
- **TensorBoard** - Training monitoring
- **CUDA** - GPU computation
- **AMP (Automatic Mixed Precision)** - Mixed precision training

---

## Model Selection

### YOLOv8 Variants

| Model | Parameters | Inference Speed | Accuracy | Recommended Use |
|-------|------------|-----------------|----------|-----------------|
| yolov8n.pt | 3.2M | Fastest | Low | Testing |
| yolov8s.pt | 11.2M | Fast | Medium | Fast mode |
| yolov8m.pt | 25.9M | Medium | High | **Recommended (Default)** |
| yolov8l.pt | 43.7M | Slow | Higher | Accuracy-focused |
| yolov8x.pt | 68.2M | Slowest | Highest | Workstation use |

### Recommended Settings by GPU Tier

| GPU Tier | VRAM | Model | Batch Size | Image Size |
|----------|------|-------|------------|------------|
| Low | <6GB | yolov8s | 8 | 480 |
| Medium | 6-12GB | yolov8m | 16 | 640 |
| High | 12-24GB | yolov8l | 32 | 640 |
| Workstation | >24GB | yolov8x | 64 | 800 |
| CPU-Only | - | yolov8n | 4 | 320 |

---

## Docker Execution

### Command Help

```bash
docker compose run --rm hsr-perception train --help
```

### Basic Training

```bash
docker compose run --rm hsr-perception train \
    --dataset /workspace/datasets/competition_day/data.yaml \
    --model yolov8m.pt \
    --output /workspace/models/finetuned
```

### Fast Mode (For Testing)

```bash
docker compose run --rm hsr-perception train \
    --dataset /workspace/datasets/competition_day/data.yaml \
    --fast
```

### With GPU Auto-Scaling

```bash
docker compose run --rm hsr-perception train \
    --dataset /workspace/datasets/competition_day/data.yaml
```
*GPU auto-scaling is enabled by default

### Main Options

| Option | Description | Default |
|--------|-------------|---------|
| `--dataset, -d` | Dataset YAML path (required) | - |
| `--model, -m` | Base model | Auto-detect |
| `--output, -o` | Output directory | models/finetuned |
| `--fast` | Fast training settings (smaller model, fewer epochs) | - |
| `--epochs` | Override epoch count | - |
| `--batch` | Override batch size | - |
| `--resume` | Resume from checkpoint | - |
| `--validate-only` | Run validation only | - |

### Advanced Options

| Option | Description | Default |
|--------|-------------|---------|
| `--no-auto-scale` | Disable GPU auto-scaling | Enabled |
| `--gpu-tier` | Specify GPU tier (low/medium/high/workstation) | Auto |
| `--llrd` | Enable Layer-wise Learning Rate Decay | - |
| `--llrd-decay-rate` | LLRD decay rate | 0.9 |
| `--no-tensorboard` | Disable TensorBoard monitoring | Enabled |
| `--tensorboard-port` | TensorBoard port | 6006 |
| `--no-oom-recovery` | Disable OOM recovery | Enabled |

### Dynamic Copy-Paste Augmentation

| Option | Description | Default |
|--------|-------------|---------|
| `--dynamic-synthetic` | Dynamic Copy-Paste synthetic data generation | Enabled |
| `--no-dynamic-synthetic` | Disable Copy-Paste | - |
| `--backgrounds-dir` | Background images directory | - |
| `--annotated-dir` | Annotated images directory | - |
| `--synthetic-ratio` | Synthetic/real image ratio | 2.0 |

### Model Export

```bash
docker compose run --rm hsr-perception train \
    --dataset /workspace/datasets/competition_day/data.yaml \
    --export onnx
```

---

## TensorBoard Monitoring

### Start TensorBoard

```bash
docker compose run --rm -p 6006:6006 hsr-perception tensorboard /workspace/runs
```

Open http://localhost:6006 in your browser

### Displayed Metrics

- **Loss**: box_loss, cls_loss, dfl_loss
- **Metrics**: mAP50, mAP50-95, precision, recall
- **Learning Rate**: Learning rate progression

---

## Configuration Parameters

### Basic Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | yolov8m.pt | Base model |
| `epochs` | 50 | Number of epochs |
| `batch` | 16 | Batch size |
| `imgsz` | 640 | Input image size |
| `patience` | 10 | Early stopping patience |

### Optimizer Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `optimizer` | AdamW | Optimization algorithm |
| `lr0` | 0.001 | Initial learning rate |
| `lrf` | 0.01 | Final learning rate factor |
| `momentum` | 0.937 | Momentum |
| `weight_decay` | 0.0005 | Weight decay |

### Data Augmentation

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hsv_h` | 0.015 | Hue variation (+/-1.5%) |
| `hsv_s` | 0.7 | Saturation variation (+/-70%) |
| `hsv_v` | 0.4 | Value variation (+/-40%) |
| `degrees` | 10.0 | Rotation (+/-10 degrees) |
| `translate` | 0.1 | Translation (+/-10%) |
| `scale` | 0.5 | Scale (0.5x - 1.5x) |
| `fliplr` | 0.5 | Horizontal flip probability |
| `mosaic` | 1.0 | Mosaic augmentation probability |
| `mixup` | 0.1 | MixUp probability |

### Performance Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `workers` | 8 | Data loader worker count |
| `cache` | True | Cache images in RAM |
| `amp` | True | Mixed precision training |
| `close_mosaic` | 10 | Disable mosaic for last N epochs |

---

## Output Files

### Directory Structure

```
models/finetuned/
└── competition_YYYYMMDD_HHMMSS/
    ├── weights/
    │   ├── best.pt          # Best mAP model
    │   ├── last.pt          # Final epoch model
    │   └── epoch*.pt        # Intermediate checkpoints
    ├── results.csv          # Per-epoch metrics
    ├── training_result.json # Training result summary
    └── tensorboard/         # TensorBoard logs
        └── events.out.tfevents.*
```

### results.csv Contents

```csv
epoch,box_loss,cls_loss,dfl_loss,mAP50,mAP50-95,precision,recall
1,1.234,0.567,0.890,0.45,0.32,0.65,0.55
2,1.012,0.456,0.789,0.52,0.38,0.72,0.62
...
```

### training_result.json

```json
{
  "best_model_path": "weights/best.pt",
  "last_model_path": "weights/last.pt",
  "metrics": {
    "mAP50": 0.87,
    "mAP50-95": 0.72,
    "precision": 0.89,
    "recall": 0.83
  },
  "training_time_minutes": 42.5,
  "epochs_completed": 50,
  "tensorboard_url": "http://localhost:6006",
  "config": { ... }
}
```

---

## OOM Recovery

Automatic countermeasures for GPU memory exhaustion:

```
1. Halve batch size
2. Reduce image size
3. Disable caching
4. Reduce worker count
```

---

## Competition Preset

```python
COMPETITION_CONFIG = {
    "model": "yolov8m.pt",
    "imgsz": 640,
    "epochs": 50,
    "batch": 16,
    "patience": 10,
    "optimizer": "AdamW",
    "lr0": 0.001,
    "amp": True,
    "cache": True,
    "close_mosaic": 10,
}
```
