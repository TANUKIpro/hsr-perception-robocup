# Collection - Data Collection

Image collection from ROS2 camera topics. Supports single capture, burst capture, and video recording.

---

## Related Files

| File | Description |
|------|-------------|
| `scripts/capture/capture_app.py` | Tkinter GUI (burst/single capture) |
| `scripts/capture/record_app.py` | Video recording app |
| `scripts/capture/burst_capture.py` | CLI burst capture |
| `scripts/capture/capture_frame.py` | Single frame capture |
| `scripts/capture/preview_window.py` | Preview window |
| `src/hsr_perception/hsr_perception/continuous_capture_node.py` | ROS2 node |
| `src/hsr_perception/srv/SetClass.srv` | Class setting service |
| `src/hsr_perception/srv/StartBurst.srv` | Burst start service |
| `src/hsr_perception/srv/GetStatus.srv` | Status retrieval service |

---

## Technologies Used

- **ROS2 Humble** - Robot middleware
- **OpenCV** - Image processing
- **Tkinter** - GUI
- **NumPy** - Array operations

---

## Capture Modes

### 1. Single Capture
Save the current frame as a single image

### 2. Burst Capture
Continuous capture at specified intervals (default: 0.2 second intervals, 50 images)

### 3. Video Recording
Record MP4 video and extract frames uniformly afterwards

---

## Docker Execution

### Start Xtion Camera Node

```bash
# Start OpenNI2 camera node (for Xtion)
docker compose run --rm hsr-perception ros2-camera
```

### Start HSR Capture Node

```bash
# Start capture node for HSR
docker compose run --rm hsr-perception ros2-capture
```

### Using start.sh (Recommended)

```bash
# Automatically starts camera node when Xtion camera is connected
./start.sh
```

`start.sh` automatically handles:
- Xtion camera detection
- udev rule configuration
- Background camera node startup

---

## GUI Application (Streamlit UI Recommended)

For data collection via Streamlit UI, use the Collection page.

### Collection Methods

| Method | Description |
|--------|-------------|
| ROS2 Camera | Real-time collection from ROS2 topics |
| Local Camera | Capture with webcam |
| File Upload | Upload image files |
| Folder Import | Batch import from folder |

---

## ROS2 Node

### Services (Execute from within Docker)

```bash
# Access Docker container
docker compose run --rm hsr-perception bash

# Execute ROS2 commands inside container
ros2 service call /continuous_capture/set_class \
    hsr_perception/srv/SetClass "{class_id: 0}"

ros2 service call /continuous_capture/start_burst \
    hsr_perception/srv/StartBurst "{num_images: 100, interval_seconds: 0.2}"

ros2 service call /continuous_capture/stop_burst std_srvs/srv/Empty

ros2 service call /continuous_capture/get_status \
    hsr_perception/srv/GetStatus
```

---

## Configuration Parameters

### Burst Capture

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_images` | 50 | Number of images to capture |
| `interval_seconds` | 0.2 | Capture interval (seconds) |

### Image Saving

| Parameter | Default | Description |
|-----------|---------|-------------|
| `jpeg_quality` | 95 | JPEG compression quality |

---

## Output Format

### Filename Format

```
<class_name>_YYYYMMDD_HHMMSS_ffffff.jpg
```

- `class_name`: Object name
- `YYYYMMDD_HHMMSS`: Date and time
- `ffffff`: Microseconds (for uniqueness)

### Directory Structure

```
raw_captures/
└── <class_name>/
    ├── <class_name>_20241208_103045_123456.jpg
    ├── <class_name>_20241208_103045_323456.jpg
    └── ...

videos/
└── <class_name>_20241208-10-30.mp4
```

---

## Supported Image Encodings

ROS2 Image message encodings supported:

- `rgb8` - RGB 8-bit
- `bgr8` - BGR 8-bit (OpenCV native)
- `mono8` - Grayscale 8-bit
- `16UC1` - Depth image 16-bit
- `32FC1` - Depth image 32-bit float

---

## Topic Presets

| Preset | Topic |
|--------|-------|
| HSR | `/hsrb/head_rgbd_sensor/rgb/image_rect_color` |
| Generic USB | `/usb_cam/image_raw` |
| RealSense | `/camera/color/image_raw` |
| Xtion | `/camera/rgb/image_raw` |
