# =============================================================================
# HSR Perception Pipeline - Dockerfile
# Ubuntu 22.04 + CUDA 12.1 + ROS2 Humble + Xtion PRO LIVE Support
# =============================================================================

FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo

# =============================================================================
# Layer 1: System locale and basic tools
# =============================================================================
RUN apt-get update && apt-get install -y \
    locales \
    curl \
    gnupg2 \
    lsb-release \
    software-properties-common \
    wget \
    git \
    vim \
    && locale-gen en_US.UTF-8 ja_JP.UTF-8 \
    && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

# =============================================================================
# Layer 2: ROS2 Humble Installation
# =============================================================================
# Add ROS2 GPG key and repository
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | \
    gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
    http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
    > /etc/apt/sources.list.d/ros2.list

# Install ROS2 Humble and required packages
RUN apt-get update && apt-get install -y \
    ros-humble-desktop \
    ros-humble-cv-bridge \
    ros-humble-sensor-msgs \
    ros-humble-std-msgs \
    ros-humble-std-srvs \
    ros-humble-image-transport \
    ros-humble-vision-msgs \
    ros-humble-openni2-camera \
    ros-dev-tools \
    python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*

# =============================================================================
# Layer 3: OpenNI2 for Xtion PRO LIVE
# =============================================================================
RUN apt-get update && apt-get install -y \
    libopenni2-dev \
    libopenni2-0 \
    openni2-utils \
    libusb-1.0-0-dev \
    udev \
    usbutils \
    && rm -rf /var/lib/apt/lists/*

# =============================================================================
# Layer 4: Python 3.10 and pip
# =============================================================================
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    python3-tk \
    && rm -rf /var/lib/apt/lists/*

# Update pip
RUN pip3 install --upgrade pip setuptools wheel

# =============================================================================
# Layer 5: PyTorch with CUDA 12.1
# =============================================================================
# Note: --ignore-installed is needed to handle pre-installed sympy in CUDA base image
RUN pip3 install --no-cache-dir --ignore-installed \
    torch>=2.1.0 \
    torchvision>=0.16.0 \
    --index-url https://download.pytorch.org/whl/cu121

# =============================================================================
# Layer 6: ML Dependencies (Ultralytics, SAM2)
# =============================================================================
# Install Ultralytics (YOLOv8)
RUN pip3 install --no-cache-dir ultralytics>=8.3.0

# Install SAM2 (Segment Anything 2)
RUN pip3 install --no-cache-dir \
    git+https://github.com/facebookresearch/segment-anything-2.git

# =============================================================================
# Layer 7: Application Dependencies
# =============================================================================
# Install GUI/OpenCV system dependencies + PyQt6 dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libfontconfig1 \
    libice6 \
    # PyQt6/Qt6 dependencies
    libxcb-cursor0 \
    libxcb-xinerama0 \
    libxkbcommon-x11-0 \
    libegl1 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python requirements
# Note: --ignore-installed handles distutils-installed packages in base image
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir --ignore-installed -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# =============================================================================
# Layer 8: Workspace Setup
# =============================================================================
WORKDIR /workspace

# Create required directories
RUN mkdir -p /workspace/datasets \
             /workspace/models/pretrained \
             /workspace/models/finetuned \
             /workspace/profiles \
             /workspace/runs \
             /workspace/src

# =============================================================================
# Layer 9: ROS2 Package Build
# =============================================================================
# Copy ROS2 package source
COPY src/hsr_perception /workspace/src/hsr_perception

# Build ROS2 package
# Note: Not using --symlink-install to avoid CMake target name conflicts
RUN /bin/bash -c "source /opt/ros/humble/setup.bash && \
    cd /workspace && \
    colcon build --packages-select hsr_perception"

# =============================================================================
# Layer 10: Application Code
# =============================================================================
# Copy application code
COPY app/ /workspace/app/
COPY scripts/ /workspace/scripts/
COPY config/ /workspace/config/
COPY run_app.sh /workspace/run_app.sh
COPY tests/ /workspace/tests/
COPY pytest.ini /workspace/pytest.ini

# Copy entrypoint script
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# =============================================================================
# Layer 11: Pre-download ML Models
# =============================================================================
# Download models to /opt/model-cache/ (not affected by volume mounts)
# These will be copied to /workspace/models/pretrained/ at container startup

RUN mkdir -p /opt/model-cache

# YOLOv8 models - download to cache for offline use
# All models used by GPU auto-scaling are pre-downloaded:
# - yolov8n.pt: Nano (minimal GPU, <4GB)
# - yolov8s.pt: Small (budget GPU, 4-6GB)
# - yolov8m.pt: Medium (standard GPU, 6-8GB) - default
# - yolov8l.pt: Large (performance GPU, 8-12GB)
RUN cd /opt/model-cache && \
    python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt'); YOLO('yolov8s.pt'); YOLO('yolov8m.pt'); YOLO('yolov8l.pt')" && \
    ls -la /opt/model-cache/

# SAM2 model - download for offline annotation
# sam2.1_hiera_base_plus.pt: Base+ model (~324MB, good balance of speed/accuracy)
RUN wget -q --show-progress -O /opt/model-cache/sam2.1_hiera_base_plus.pt \
    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt" && \
    ls -la /opt/model-cache/

# =============================================================================
# Environment Variables
# =============================================================================
ENV ROS_DOMAIN_ID=0
ENV FASTRTPS_DEFAULT_PROFILES_FILE=/workspace/config/fastdds_profile.xml
ENV PYTHONPATH=/workspace:/workspace/scripts:$PYTHONPATH

# =============================================================================
# Ports
# =============================================================================
# Streamlit
EXPOSE 8501
# TensorBoard
EXPOSE 6006

# =============================================================================
# Entrypoint
# =============================================================================
ENTRYPOINT ["/entrypoint.sh"]
CMD ["streamlit"]
