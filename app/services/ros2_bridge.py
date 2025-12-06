"""
ROS2 Bridge

Provides bridge between Streamlit app and ROS2 system.
Uses subprocess to call ros2 CLI commands, avoiding direct rclpy imports.
"""

import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import re


@dataclass
class ROS2TopicInfo:
    """Information about a ROS2 topic."""
    name: str
    type: str
    publishers: int = 0
    subscribers: int = 0


@dataclass
class ServiceCallResult:
    """Result of a ROS2 service call."""
    success: bool
    message: str = ""
    data: Optional[Dict[str, Any]] = None


class ROS2Bridge:
    """
    Bridge between Streamlit app and ROS2 system.

    Uses subprocess to call ros2 CLI commands and parse output.
    This approach avoids importing rclpy directly, which can cause issues
    with Streamlit's event loop.

    For production with heavy ROS2 interaction, consider using:
    - rosbridge_suite with WebSocket
    - roslibpy for Python WebSocket client

    Usage:
        bridge = ROS2Bridge()

        if bridge.is_available():
            topics = bridge.list_image_topics()
            for topic in topics:
                print(f"{topic.name}: {topic.type}")

            # Call capture service
            result = bridge.start_burst_capture(
                class_id=0,
                num_images=50,
                interval=0.2
            )
    """

    # Default ROS2 settings
    DEFAULT_SOURCE_SCRIPT = "/opt/ros/humble/setup.bash"

    # Common HSR image topics
    HSR_IMAGE_TOPICS = [
        "/hsrb/head_rgbd_sensor/rgb/image_rect_color",
        "/hsrb/head_rgbd_sensor/rgb/image_raw",
        "/hsrb/head_rgbd_sensor/depth_registered/image_rect_raw",
        "/hsrb/hand_camera/image_raw",
        "/hsrb/head_l_stereo_camera/image_rect_color",
        "/hsrb/head_r_stereo_camera/image_rect_color",
    ]

    # Common generic camera topics
    GENERIC_IMAGE_TOPICS = [
        "/camera/color/image_raw",
        "/camera/rgb/image_raw",
        "/camera/image_raw",
        "/image_raw",
        "/usb_cam/image_raw",
    ]

    # Capture node service names
    CAPTURE_SERVICES = {
        "set_class": "/continuous_capture/set_class",
        "start_burst": "/continuous_capture/start_burst",
        "get_status": "/continuous_capture/get_status",
        "single_capture": "/continuous_capture/single_capture",
    }

    def __init__(
        self,
        source_script: Optional[str] = None,
        timeout: float = 5.0,
    ):
        """
        Initialize ROS2 bridge.

        Args:
            source_script: Path to ROS2 setup script. Defaults to Humble.
            timeout: Default timeout for CLI commands in seconds.
        """
        self.source_script = source_script or self.DEFAULT_SOURCE_SCRIPT
        self.timeout = timeout
        self._available: Optional[bool] = None

    def _run_ros2_cmd(
        self,
        cmd: List[str],
        timeout: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """
        Run ros2 command and return output.

        Args:
            cmd: Command arguments (without 'ros2' prefix)
            timeout: Command timeout in seconds

        Returns:
            Tuple of (success, output_text)
        """
        timeout = timeout or self.timeout

        # Build full command with source (escape arguments for shell)
        escaped_args = " ".join(shlex.quote(arg) for arg in cmd)
        full_cmd = f"source {self.source_script} 2>/dev/null && ros2 {escaped_args}"

        try:
            result = subprocess.run(
                ["bash", "-c", full_cmd],
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            return (result.returncode == 0, result.stdout + result.stderr)

        except subprocess.TimeoutExpired:
            return (False, f"Command timed out after {timeout}s")
        except Exception as e:
            return (False, str(e))

    def is_available(self) -> bool:
        """
        Check if ROS2 is available and working.

        Returns:
            True if ROS2 commands can be executed
        """
        if self._available is not None:
            return self._available

        success, _ = self._run_ros2_cmd(["topic", "list"], timeout=3.0)
        self._available = success
        return self._available

    def refresh_availability(self) -> bool:
        """
        Re-check ROS2 availability.

        Returns:
            True if ROS2 is available
        """
        self._available = None
        return self.is_available()

    # ========== Topic Operations ==========

    def list_topics(self) -> List[ROS2TopicInfo]:
        """
        List all available ROS2 topics.

        Returns:
            List of topic information
        """
        if not self.is_available():
            return []

        success, output = self._run_ros2_cmd(["topic", "list", "-t"])

        if not success:
            return []

        topics = []
        for line in output.strip().split("\n"):
            if not line.strip():
                continue

            # Format: "/topic/name [type]"
            match = re.match(r"(/\S+)\s+\[(.+)\]", line.strip())
            if match:
                topics.append(ROS2TopicInfo(
                    name=match.group(1),
                    type=match.group(2),
                ))

        return topics

    def list_image_topics(self) -> List[ROS2TopicInfo]:
        """
        List available image topics.

        Returns:
            List of image topic information
        """
        topics = self.list_topics()

        image_types = [
            "sensor_msgs/msg/Image",
            "sensor_msgs/Image",
        ]

        return [t for t in topics if any(img_type in t.type for img_type in image_types)]

    def get_common_topics(self) -> List[str]:
        """
        Get list of common camera topics to check.

        Returns:
            Combined list of HSR and generic topics
        """
        return self.HSR_IMAGE_TOPICS + self.GENERIC_IMAGE_TOPICS

    def check_topic_exists(self, topic_name: str) -> bool:
        """
        Check if a specific topic exists.

        Args:
            topic_name: Topic name to check

        Returns:
            True if topic exists
        """
        topics = self.list_topics()
        return any(t.name == topic_name for t in topics)

    # ========== Capture Node Services ==========

    def set_capture_class(
        self,
        class_id: int,
        service_name: Optional[str] = None,
    ) -> ServiceCallResult:
        """
        Set the current capture class via service.

        Args:
            class_id: Class ID to set (0-indexed)
            service_name: Override default service name

        Returns:
            ServiceCallResult with success status
        """
        service = service_name or self.CAPTURE_SERVICES["set_class"]

        success, output = self._run_ros2_cmd([
            "service", "call",
            service,
            "hsr_perception/srv/SetClass",
            f"{{class_id: {class_id}}}"
        ], timeout=10.0)

        return ServiceCallResult(
            success=success and "success=True" in output.lower(),
            message=output if not success else f"Class set to {class_id}",
            data={"class_id": class_id}
        )

    def start_burst_capture(
        self,
        class_id: int,
        num_images: int = 50,
        interval: float = 0.2,
        service_name: Optional[str] = None,
    ) -> ServiceCallResult:
        """
        Start burst capture via service.

        Args:
            class_id: Class ID for captured images
            num_images: Number of images to capture
            interval: Interval between captures in seconds
            service_name: Override default service name

        Returns:
            ServiceCallResult with success status
        """
        service = service_name or self.CAPTURE_SERVICES["start_burst"]

        success, output = self._run_ros2_cmd([
            "service", "call",
            service,
            "hsr_perception/srv/StartBurst",
            f"{{class_id: {class_id}, num_images: {num_images}, interval_seconds: {interval}}}"
        ], timeout=10.0)

        return ServiceCallResult(
            success=success and "success=True" in output.lower(),
            message=output if not success else f"Burst started: {num_images} images",
            data={
                "class_id": class_id,
                "num_images": num_images,
                "interval": interval,
            }
        )

    def get_capture_status(
        self,
        service_name: Optional[str] = None,
    ) -> ServiceCallResult:
        """
        Get current capture node status.

        Args:
            service_name: Override default service name

        Returns:
            ServiceCallResult with status information
        """
        service = service_name or self.CAPTURE_SERVICES["get_status"]

        success, output = self._run_ros2_cmd([
            "service", "call",
            service,
            "hsr_perception/srv/GetStatus",
            "{}"
        ], timeout=10.0)

        # Parse output for status info
        data = {}
        if success:
            # Try to extract status fields from response
            if "is_capturing" in output.lower():
                data["is_capturing"] = "true" in output.lower()
            if "current_class" in output.lower():
                match = re.search(r"current_class[=:]\s*(\d+)", output.lower())
                if match:
                    data["current_class"] = int(match.group(1))
            if "images_captured" in output.lower():
                match = re.search(r"images_captured[=:]\s*(\d+)", output.lower())
                if match:
                    data["images_captured"] = int(match.group(1))

        return ServiceCallResult(
            success=success,
            message=output,
            data=data,
        )

    def call_single_capture(
        self,
        service_name: Optional[str] = None,
    ) -> ServiceCallResult:
        """
        Trigger single image capture.

        Args:
            service_name: Override default service name

        Returns:
            ServiceCallResult with success status
        """
        service = service_name or self.CAPTURE_SERVICES.get("single_capture")
        if not service:
            return ServiceCallResult(
                success=False,
                message="Single capture service not configured"
            )

        success, output = self._run_ros2_cmd([
            "service", "call",
            service,
            "std_srvs/srv/Trigger",
            "{}"
        ], timeout=10.0)

        return ServiceCallResult(
            success=success and "success=true" in output.lower(),
            message=output if not success else "Image captured",
        )

    # ========== Node Operations ==========

    def list_nodes(self) -> List[str]:
        """
        List running ROS2 nodes.

        Returns:
            List of node names
        """
        if not self.is_available():
            return []

        success, output = self._run_ros2_cmd(["node", "list"])

        if not success:
            return []

        return [line.strip() for line in output.strip().split("\n") if line.strip()]

    def check_capture_node_running(self) -> bool:
        """
        Check if the capture node is running.

        Returns:
            True if capture node is found
        """
        nodes = self.list_nodes()
        capture_node_names = ["continuous_capture", "capture_node"]
        return any(
            any(name in node for name in capture_node_names)
            for node in nodes
        )

    # ========== Service Operations ==========

    def list_services(self) -> List[str]:
        """
        List available ROS2 services.

        Returns:
            List of service names
        """
        if not self.is_available():
            return []

        success, output = self._run_ros2_cmd(["service", "list"])

        if not success:
            return []

        return [line.strip() for line in output.strip().split("\n") if line.strip()]

    def check_capture_services_available(self) -> Dict[str, bool]:
        """
        Check which capture services are available.

        Returns:
            Dictionary mapping service name to availability
        """
        services = self.list_services()
        return {
            key: any(svc in services for svc in [value, value.lstrip("/")])
            for key, value in self.CAPTURE_SERVICES.items()
        }

    # ========== Utility Methods ==========

    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Get diagnostic information about ROS2 status.

        Returns:
            Dictionary with diagnostic information
        """
        available = self.is_available()

        diagnostics = {
            "ros2_available": available,
            "source_script": self.source_script,
        }

        if available:
            diagnostics["topics_count"] = len(self.list_topics())
            diagnostics["image_topics_count"] = len(self.list_image_topics())
            diagnostics["nodes_count"] = len(self.list_nodes())
            diagnostics["capture_node_running"] = self.check_capture_node_running()
            diagnostics["capture_services"] = self.check_capture_services_available()

        return diagnostics

    def capture_preview_image(
        self,
        topic: str,
        output_path: str = "/tmp/preview_frame.jpg",
        timeout: float = 2.0
    ) -> Optional[str]:
        """
        Capture a single frame from an image topic.

        Uses a separate Python script to avoid rclpy import issues with Streamlit.

        Args:
            topic: Image topic to capture from
            output_path: Path to save the captured image
            timeout: Timeout in seconds

        Returns:
            Path to saved image if successful, None otherwise
        """
        script_path = Path(__file__).parent.parent.parent / "scripts/utils/capture_frame.py"

        if not script_path.exists():
            return None

        # Build command with ROS2 environment sourced
        # Note: Cannot use _run_ros2_cmd as it adds 'ros2' prefix
        full_cmd = (
            f"source {self.source_script} 2>/dev/null && "
            f"python3 {script_path} --topic {topic} --output {output_path} --timeout {timeout}"
        )

        try:
            result = subprocess.run(
                ["bash", "-c", full_cmd],
                capture_output=True,
                text=True,
                timeout=timeout + 2.0,  # Extra time for setup
            )

            if result.returncode == 0 and Path(output_path).exists():
                return output_path
            return None

        except subprocess.TimeoutExpired:
            return None
        except Exception:
            return None

    def open_preview_window(self, topic: str) -> bool:
        """
        Open a real-time preview window for the specified topic.

        Launches a separate OpenCV window process that displays the camera
        feed with a center reticle overlay.

        Args:
            topic: Image topic to display

        Returns:
            True if successfully launched, False otherwise
        """
        script_path = Path(__file__).parent.parent.parent / "scripts/utils/preview_window.py"

        if not script_path.exists():
            return False

        # Build command with ROS2 environment sourced
        full_cmd = (
            f"source {self.source_script} 2>/dev/null && "
            f"python3 {script_path} --topic {topic}"
        )

        try:
            # Run in background (non-blocking)
            subprocess.Popen(
                ["bash", "-c", full_cmd],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True
        except Exception:
            return False

    def open_capture_app(self, output_dir: Optional[str] = None) -> bool:
        """
        Open the capture application with full GUI.

        Launches the Tkinter-based capture application with:
        - Topic selection from available ROS2 topics
        - Real-time preview with center reticle
        - Single and burst capture controls
        - Configurable parameters

        Args:
            output_dir: Default output directory for captures

        Returns:
            True if successfully launched, False otherwise
        """
        script_path = Path(__file__).parent.parent.parent / "scripts/utils/capture_app.py"

        if not script_path.exists():
            return False

        # Build command with ROS2 environment sourced
        cmd_parts = [
            f"source {self.source_script} 2>/dev/null",
            f"python3 {script_path}"
        ]

        if output_dir:
            cmd_parts[1] += f" --output-dir {shlex.quote(output_dir)}"

        full_cmd = " && ".join(cmd_parts)

        try:
            # Run in background (non-blocking)
            subprocess.Popen(
                ["bash", "-c", full_cmd],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True
        except Exception:
            return False

    def start_direct_burst_capture(
        self,
        topic: str,
        output_dir: str,
        class_name: str,
        count: int = 50,
        interval: float = 0.2,
    ) -> bool:
        """
        Start burst capture directly from a topic.

        Captures images from the specified topic without needing
        the continuous_capture_node.

        Args:
            topic: Image topic to capture from
            output_dir: Directory to save captured images
            class_name: Class name for filename prefix
            count: Number of images to capture
            interval: Interval between captures in seconds

        Returns:
            True if capture started successfully
        """
        script_path = Path(__file__).parent.parent.parent / "scripts/utils/burst_capture.py"

        if not script_path.exists():
            return False

        # Build command with ROS2 environment sourced
        full_cmd = (
            f"source {self.source_script} 2>/dev/null && "
            f"python3 {script_path} "
            f"--topic {shlex.quote(topic)} "
            f"--output-dir {shlex.quote(output_dir)} "
            f"--class-name {shlex.quote(class_name)} "
            f"--count {count} "
            f"--interval {interval}"
        )

        try:
            # Run in background (non-blocking)
            subprocess.Popen(
                ["bash", "-c", full_cmd],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True
        except Exception:
            return False
