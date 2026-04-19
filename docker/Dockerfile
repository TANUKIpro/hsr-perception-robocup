# =============================================================================
# HSR Perception Pipeline - Dockerfile
# Ubuntu 22.04 + CUDA 12.1 + ROS2 Humble + OpenNI2 (Xtion) + YOLOv8 + Streamlit
# Consumer of pybullet_hsr datasets for training and evaluation, plus live
# inference from ROS2 image topics (Xtion PRO LIVE).
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
    gnupg2 \
    lsb-release \
    software-properties-common \
    && locale-gen en_US.UTF-8 ja_JP.UTF-8 \
    && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

# -----------------------------------------------------------------------------
# ROS2 Humble (ros-base keeps the image slim vs ros-desktop)
# -----------------------------------------------------------------------------
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | \
    gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
    http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
    > /etc/apt/sources.list.d/ros2.list

RUN apt-get update && apt-get install -y \
    ros-humble-ros-base \
    ros-humble-sensor-msgs \
    ros-humble-std-msgs \
    ros-humble-openni2-camera \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# OpenNI2 + USB runtime for Xtion PRO LIVE
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y \
    libopenni2-dev \
    libopenni2-0 \
    openni2-utils \
    libusb-1.0-0 \
    libusb-1.0-0-dev \
    udev \
    usbutils \
    && rm -rf /var/lib/apt/lists/*

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
# releases (2.6 -> cu124, 2.7+ -> cu126+) require newer NVIDIA drivers than
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
# OpenCV + Qt6/X11 runtime deps
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libfontconfig1 \
    libice6 \
    libdbus-1-3 \
    libegl1 \
    libxkbcommon-x11-0 \
    libxcb-cursor0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libxcb-sync1 \
    libxcb-xfixes0 \
    libxcb-xinerama0 \
    libxcb-xkb1 \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# Python app dependencies (PyQt6 is pulled in via requirements.txt)
# -----------------------------------------------------------------------------
# python3-blinker is installed by ROS2's apt dependency chain as a distutils
# package, which pip cannot uninstall. Streamlit pulls in a newer blinker,
# so we overwrite the distutils copy explicitly before the main install.
RUN pip3 install --no-cache-dir --ignore-installed blinker

COPY requirements.txt /tmp/requirements.txt
# Note: no --ignore-installed on the requirements install - otherwise pip
# would reinstall torch from the default PyPI index and overwrite the
# pinned cu121 build above.
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
# Source ROS2 automatically for interactive shells
# -----------------------------------------------------------------------------
RUN echo "source /opt/ros/humble/setup.bash" >> /etc/bash.bashrc

# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------
ENV PYTHONPATH=/workspace:/workspace/scripts:$PYTHONPATH
ENV PYBULLET_HSR_ROOT=/pybullet_hsr
ENV ROS_DISTRO=humble
ENV ROS_DOMAIN_ID=0
ENV FASTRTPS_DEFAULT_PROFILES_FILE=/workspace/config/fastdds_profile.xml
ENV QT_X11_NO_MITSHM=1
# Make rclpy importable without sourcing (entrypoint still sources for CLI).
ENV AMENT_PREFIX_PATH=/opt/ros/humble
ENV LD_LIBRARY_PATH=/opt/ros/humble/lib:${LD_LIBRARY_PATH}
ENV PYTHONPATH=/opt/ros/humble/lib/python3.10/site-packages:${PYTHONPATH}

EXPOSE 8501
EXPOSE 6006

ENTRYPOINT ["/entrypoint.sh"]
CMD ["streamlit"]
