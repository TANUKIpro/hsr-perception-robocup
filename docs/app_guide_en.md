# HSR Object Manager App Guide

GUI application usage guide for the RoboCup@Home object recognition pipeline.

## Overview

HSR Object Manager is a Streamlit-based web application that provides a GUI for the entire object recognition model creation process, from data collection to evaluation.

**Key Features:**
- 📊 Dashboard: Visualize collection progress and pipeline status
- 📋 Registry: Object registration and reference image management
- 📸 Collection: Data collection from ROS2/camera/files
- 🏷️ Annotation: Run auto-annotation pipeline
- 🎓 Training: Execute YOLOv8 fine-tuning
- 📈 Evaluation: Model evaluation and competition requirement check
- ⚙️ Settings: System settings and status check

---

## Starting the App

```bash
# Method 1: Shell script
./run_app.sh

# Method 2: Direct launch
streamlit run app/main.py

# Method 3: Custom port
streamlit run app/main.py --server.port 8502
```

Open http://localhost:8501 in your browser.

---

## Page Guide

### 📊 Dashboard

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

### 📋 Registry

Register and manage objects.

**View Objects Tab:**
- Display list of registered objects
- Filter by category
- Add reference images
- "📸 Collect" to navigate to Collection page
- Delete objects

**Add New Object Tab:**
- ID (auto-numbered)
- Name (lowercase English, no spaces)
- Display name
- Category selection
- Target sample count
- Properties (heavy/tiny/has liquid/size)

**Quick Import:**
- "Import iHR Standard Objects" to bulk register iHR standard objects

### 📸 Collection

Collect image data. Four methods are available.

#### 🤖 ROS2 Camera Tab

Collect images directly from HSR or ROS2-compatible cameras.

**Requirements:**
- ROS2 Humble installed and sourced
- Continuous capture node running

```bash
# Node launch command
ros2 launch hsr_perception capture.launch.py
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

#### 📷 Local Camera Tab

Capture with webcam or device built-in camera.

1. Select object
2. Click "Take a photo"
3. Auto-saved

#### 📁 File Upload Tab

Upload image files.

1. Select object
2. Select and upload multiple images
3. Auto-saved with count update

#### 📂 Folder Import Tab

Bulk import by specifying a folder path.

1. Select object
2. Enter folder path
3. Click "Import from Folder"

**Data Sync:**
After collection, click "Sync to Datasets Directory" to sync data to `datasets/raw_captures/`.

### 🏷️ Annotation

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

**Estimated Time:**
- Approximately 5 minutes per 100 images

**Output:**
- `datasets/annotated/{session_name}/`
  - `images/train/`, `images/val/`
  - `labels/train/`, `labels/val/`
  - `data.yaml`

### 🎓 Training

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
- `yolov8m.pt`: Accuracy-focused (competition recommended), training time: ~45 min
- `yolov8s.pt`: Balanced, training time: ~30 min
- `yolov8n.pt`: Fast, training time: ~20 min

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

### 📈 Evaluation

Evaluate trained models.

**Run Evaluation Tab:**

1. Select model
2. Select dataset
3. Set confidence threshold (default: 0.25)
4. Click "Start Evaluation"

**Metrics:**
- mAP@50: Target ≥ 85%
- mAP@50-95
- Precision / Recall
- Inference time: Target ≤ 100ms
- Competition requirement check (automatic)

**Results Tab:**
List of past evaluation results with details.

**Visual Test Tab:**
Test predictions on a single image.

1. Select model
2. Upload image or select from dataset
3. Click "Run Prediction"
4. Visualize detection results

### ⚙️ Settings

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

## Competition Day Workflow

### Preparation (Before Competition)

1. Start the app: `./run_app.sh`
2. Registry → Quick Import to register standard objects
3. Prepare background image (white sheet, etc.)

### Phase 1: Data Collection (40 min)

1. Go to Collection → ROS2 Camera tab
2. Start continuous capture node (separate terminal)
3. For each object:
   - Select object
   - Set class
   - Burst capture (50-100 images)
   - Rotate object and capture multiple times

### Phase 2: Annotation (25 min)

1. Go to Annotation page
2. Select Background method
3. Select background image
4. Click "Start Annotation"
5. Wait for completion

### Phase 3: Training (45 min)

1. Go to Training page
2. Select created dataset
3. Set `yolov8m.pt` / 50 epochs
4. Click "Start Training"
5. Monitor progress bar

### Phase 4: Evaluation (15 min)

1. Go to Evaluation page
2. Select trained model
3. Click "Start Evaluation"
4. Competition requirement check:
   - mAP@50 ≥ 85% ✓
   - Inference time ≤ 100ms ✓
5. Final confirmation with Visual Test

### Deployment

Copy the trained model path:
```
models/finetuned/{run_name}/weights/best.pt
```

---

## Troubleshooting

### Cannot Connect to ROS2

1. Source ROS2 environment:
   ```bash
   source /opt/ros/humble/setup.bash
   ```
2. Start capture node:
   ```bash
   ros2 launch hsr_perception capture.launch.py
   ```

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

## Related Commands

### CLI Operations (Alternative to GUI)

```bash
# Annotation
python scripts/annotation/auto_annotate.py \
    --method background \
    --background datasets/backgrounds/bg.jpg \
    --input-dir datasets/raw_captures \
    --output-dir datasets/annotated/session1

# Training
python scripts/training/quick_finetune.py \
    --dataset datasets/annotated/session1/data.yaml

# Evaluation
python scripts/evaluation/evaluate_model.py \
    --model models/finetuned/run1/weights/best.pt \
    --dataset datasets/annotated/session1/data.yaml
```

---

## Configuration Files

See [configuration.md](./configuration.md) for details.
