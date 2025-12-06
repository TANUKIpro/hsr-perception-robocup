# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
Although the text is written in English, please respond to any questions or clarifications from users in Japanese.

## Project Overview

HSR (Human Support Robot) perception pipeline for RoboCup@Home competitions. The system handles object detection under time constraints, where models must be fine-tuned within 2-3 hours on competition day.

**Environment**: Ubuntu 22.04 / ROS2 Humble / Python

**Python Virtual Environment**: `/home/roboworks/Documents/hsr-perception-robocup/venv/perception/bin/python`
- Use this venv when running Python scripts locally for testing

## Architecture

### Competition Day Workflow
```
Data Collection → Auto-Annotation → Fine-tuning → Evaluation → Deploy
   (ROS2 node)    (背景差分/SAM2)    (YOLOv8m)    (mAP check)   (HSR)
     ~40min         ~25min           ~45min        ~15min
```

### Detection Strategy
- **Primary**: YOLOv8m fine-tuning (target: mAP ≥85%, inference ≤100ms)
- **Auto-annotation**: Background subtraction (primary) + SAM2 (fallback)

## Implemented Components

### Configuration
- `config/object_classes.json` - Class/category definitions with sample tracking

### Auto-Annotation (`scripts/annotation/`)
- `annotation_utils.py` - YOLO format conversion, dataset split utilities
- `background_subtraction.py` - Background subtraction annotator (fast, simple)
- `sam2_annotator.py` - SAM2 annotator (accurate, GPU required)
- `auto_annotate.py` - Main orchestration pipeline

### Training (`scripts/training/`)
- `quick_finetune.py` - Competition-optimized YOLOv8 fine-tuning

### Evaluation (`scripts/evaluation/`)
- `evaluate_model.py` - mAP, inference time, requirements verification
- `visual_verification.py` - Interactive prediction visualization

### ROS2 Package (`src/hsr_perception/`)
- `continuous_capture_node.py` - Data collection node with burst capture
- `srv/SetClass.srv`, `srv/StartBurst.srv`, `srv/GetStatus.srv` - Service definitions
- `launch/capture.launch.py` - Launch file for capture node

## Build & Run Commands

```bash
# Install Python dependencies
pip install -r requirements.txt

# ROS2 package build (from src directory)
colcon build --packages-select hsr_perception
source install/setup.bash

# Launch capture node
ros2 launch hsr_perception capture.launch.py

# Run annotation
python scripts/annotation/auto_annotate.py --method background \
    --background path/to/bg.jpg --input-dir raw/ --output-dir dataset/

# Run training
python scripts/training/quick_finetune.py --dataset dataset/data.yaml

# Evaluate model
python scripts/evaluation/evaluate_model.py --model best.pt --dataset data.yaml
```

## Key Files

| File | Purpose |
|------|---------|
| `config/object_classes.json` | Class definitions (edit for each competition) |
| `scripts/annotation/auto_annotate.py` | Main annotation pipeline |
| `scripts/training/quick_finetune.py` | Competition day training |
| `scripts/evaluation/evaluate_model.py` | Model verification |

## Tech Stack

- **Object Detection**: YOLOv8 (Ultralytics)
- **Segmentation**: SAM2 (Meta)
- **ROS2**: cv_bridge, sensor_msgs, custom services

## Branch Strategy

- `main` - Stable
- `develop` - Development
- `feature/*` - Feature development
- `competition/*` - Competition-specific adjustments

## Commit Message Convention

```
feat: New feature
fix: Bug fix
docs: Documentation
refactor: Refactoring
test: Test additions/modifications
```
