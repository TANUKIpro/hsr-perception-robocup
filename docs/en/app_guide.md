# HSR Object Manager App Guide

GUI application usage guide for the RoboCup@Home object recognition pipeline.

## Overview

HSR Object Manager is a Streamlit-based web application that provides a GUI for the entire object recognition model creation process, from data collection to evaluation.

**Key Features:**
- Dashboard: Visualize collection progress and pipeline status
- Registry: Object registration and reference image management
- Collection: Data collection from ROS2/camera/files
- Annotation: Run auto-annotation pipeline
- Training: Execute YOLOv8 fine-tuning
- Evaluation: Model evaluation and competition requirement check
- Settings: System settings and status check

---

## Starting the App

```bash
# Docker launch (recommended)
./start.sh

# Or
docker compose up
```

Open http://localhost:8501 in your browser.

---

## Page Guide

### Dashboard

Get an overview of the entire pipeline status at a glance.

**Displayed Information:**
- Number of registered objects
- Collected images count and target achievement rate
- Progress by category
- Number of annotated datasets
- Number of trained models
- Number of running tasks

**Actions:**
- "Export to YOLO Config": Update `config/object_classes.json`

### Registry

Register and manage objects.

**View Objects Tab:**
- Display list of registered objects
- Filter by category
- Add reference images
- "Collect" to navigate to Collection page
- Delete objects

**Add New Object Tab:**
- ID (auto-numbered)
- Name (lowercase English, no spaces)
- Display name
- Category selection
- Target sample count
- Properties (heavy/tiny/has liquid/size)

### Collection

Collect image data. Four methods are available.

#### ROS2 Camera Tab

Collect images directly from HSR or ROS2-compatible cameras.

**Requirements:**
- ROS2-compatible camera connected
- For Xtion camera, `start.sh` auto-configures

```bash
# Start ROS2 camera node (separate terminal)
docker compose run --rm hsr-perception ros2-camera
```

**Operation Steps:**
1. Select target object
2. Select image topic (auto-detected or preset)
3. "Set Capture Class" to set the class
4. Specify number of images and interval
5. "Start Burst Capture" to start continuous capture

**Topic Presets:**
- HSR: `/hsrb/head_rgbd_sensor/rgb/image_rect_color`
- Generic USB: `/usb_cam/image_raw`
- RealSense: `/camera/color/image_raw`

#### Local Camera Tab

Capture with webcam or device built-in camera.

1. Select object
2. Click "Take a photo"
3. Auto-saved

#### File Upload Tab

Upload image files.

1. Select object
2. Select and upload multiple images
3. Auto-saved with count update

#### Folder Import Tab

Bulk import by specifying a folder path.

1. Select object
2. Enter folder path
3. Click "Import from Folder"

**Data Sync:**
After collection, click "Sync to Datasets Directory" to sync data to `datasets/raw_captures/`.

### Annotation

Run the auto-annotation pipeline.

**Prerequisites:**
- Image data exists in `datasets/raw_captures/`
- Classes defined in `config/object_classes.json`

**Settings:**

| Item | Description | Recommended |
|------|-------------|-------------|
| Annotation Method | background / sam2 | background (fast) |
| Train/Val Split | Train/validation split ratio | 0.85 |
| Background Image | Background image (background method only) | White background |
| Minimum Contour Area | Min contour area (background method) | 500 |

**Execution Steps:**
1. Select method
2. Select/upload background image (for background method)
3. Set split ratio
4. Click "Start Annotation"

**Output:**
- `datasets/annotated/{session_name}/`
  - `images/train/`, `images/val/`
  - `labels/train/`, `labels/val/`
  - `data.yaml`

### Training

Execute YOLOv8 fine-tuning.

**Settings:**

| Item | Description | Recommended |
|------|-------------|-------------|
| Dataset | Annotated dataset | - |
| Base Model | Base model | yolov8m.pt |
| Epochs | Number of epochs | 50 |
| Batch Size | Batch size | 16 |
| Fast Mode | Fast mode (for testing) | OFF |

**Model Selection Guide:**
- `yolov8m.pt`: Accuracy-focused (competition recommended)
- `yolov8s.pt`: Balanced
- `yolov8n.pt`: Fast

**GPU Check:**
GPU availability is automatically checked on the page.

**Execution Steps:**
1. Select dataset
2. Set base model and epochs
3. Click "Start Training"

**Output:**
- `models/finetuned/{run_name}/weights/best.pt`
- `models/finetuned/{run_name}/weights/last.pt`
- `models/finetuned/{run_name}/training_result.json`

### Evaluation

Evaluate trained models.

**Run Evaluation Tab:**

1. Select model
2. Select dataset
3. Set confidence threshold (default: 0.25)
4. Click "Start Evaluation"

**Metrics:**
- mAP@50: Target >= 85%
- mAP@50-95
- Precision / Recall
- Inference time: Target <= 100ms
- Competition requirement check (automatic)

**Results Tab:**
List of past evaluation results with details.

**Visual Test Tab:**
Test predictions on a single image.

1. Select model
2. Upload image or select from dataset
3. Click "Run Prediction"
4. Visualize detection results

### Settings

System settings and status check.

**Data Management:**
- Export to YOLO Config: Export class settings
- Update All Collection Counts: Recount collection numbers
- Sync All to Datasets: Sync all data

**Category Management:**
- Add categories

**System Status:**
- ROS2 connection status
- GPU availability
- Topic/node counts

**Data Paths:**
- View various paths

---

## Troubleshooting

### Cannot Connect to ROS2

1. Start camera node:
   ```bash
   docker compose run --rm hsr-perception ros2-camera
   ```
2. For Xtion camera, use `start.sh` for automatic configuration

### Annotation Fails

1. Check if image data exists in `datasets/raw_captures/`
2. Check if background image is correctly selected
3. Check if classes are defined in `config/object_classes.json`

### Training is Slow

1. Check GPU availability (Settings page)
2. Reduce batch size
3. Enable Fast Mode (for testing)

### Low mAP in Evaluation

1. Check if data volume is sufficient (50+ images per class recommended)
2. Check annotation quality
3. Increase epochs and retrain

---

## Configuration Files

See [configuration.md](./configuration.md) for details.
