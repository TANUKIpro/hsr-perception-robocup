# =============================================================================
# HSR Perception Pipeline - Dockerfile
# Ubuntu 22.04 + CUDA 12.1 + YOLOv8 + Streamlit
# Consumer of pybullet_hsr datasets for training and evaluation only.
# =============================================================================

FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo

# -----------------------------------------------------------------------------
# System tools
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y \
    locales \
    curl \
    wget \
    git \
    vim \
    && locale-gen en_US.UTF-8 ja_JP.UTF-8 \
    && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

# -----------------------------------------------------------------------------
# Python
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip setuptools wheel

# -----------------------------------------------------------------------------
# PyTorch + CUDA 12.1
# Pinned: torch 2.5.x is the last series with cu121 wheels. Later PyTorch
# releases (2.6 → cu124, 2.7+ → cu126+) require newer NVIDIA drivers than
# the host 535-series. Do not float this without also bumping the driver.
# -----------------------------------------------------------------------------
RUN pip3 install --no-cache-dir --ignore-installed \
    "torch==2.5.1" \
    "torchvision==0.20.1" \
    --index-url https://download.pytorch.org/whl/cu121

# -----------------------------------------------------------------------------
# Ultralytics (YOLOv8)
# -----------------------------------------------------------------------------
RUN pip3 install --no-cache-dir "ultralytics>=8.3.0"

# -----------------------------------------------------------------------------
# OpenCV runtime deps
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# Python app dependencies
# -----------------------------------------------------------------------------
COPY requirements.txt /tmp/requirements.txt
# Note: no --ignore-installed here — otherwise pip would reinstall torch
# from the default PyPI index and overwrite the pinned cu121 build above.
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# -----------------------------------------------------------------------------
# Workspace layout
# -----------------------------------------------------------------------------
WORKDIR /workspace

RUN mkdir -p /workspace/datasets \
             /workspace/models/pretrained \
             /workspace/models/finetuned \
             /workspace/runs \
             /workspace/app_data/tasks

# -----------------------------------------------------------------------------
# Application code
# -----------------------------------------------------------------------------
COPY app/ /workspace/app/
COPY scripts/ /workspace/scripts/
COPY config/ /workspace/config/
COPY tests/ /workspace/tests/
COPY pytest.ini /workspace/pytest.ini

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# -----------------------------------------------------------------------------
# Pre-cache YOLOv8 models for offline use
# -----------------------------------------------------------------------------
RUN mkdir -p /opt/model-cache && cd /opt/model-cache && \
    python3 -c "from ultralytics import YOLO; \
                YOLO('yolov8n.pt'); YOLO('yolov8s.pt'); \
                YOLO('yolov8m.pt'); YOLO('yolov8l.pt')" && \
    ls -la /opt/model-cache/

# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------
ENV PYTHONPATH=/workspace:/workspace/scripts:$PYTHONPATH
ENV PYBULLET_HSR_ROOT=/pybullet_hsr

EXPOSE 8501
EXPOSE 6006

ENTRYPOINT ["/entrypoint.sh"]
CMD ["streamlit"]
