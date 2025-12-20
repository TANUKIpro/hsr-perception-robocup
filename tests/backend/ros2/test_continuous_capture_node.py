"""
Tests for continuous capture ROS2 node.

Tests the ContinuousCaptureNode class including image subscription,
service handlers, and burst capture functionality.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch, PropertyMock

import numpy as np
import pytest

# Add src directory to path
src_dir = Path(__file__).parent.parent.parent.parent / "src" / "hsr_perception" / "hsr_perception"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))


# Mock ROS2 modules before importing the node
@pytest.fixture
def mock_ros2():
    """Create mock ROS2 modules."""
    # Create comprehensive ROS2 mocks
    mock_rclpy = MagicMock()
    mock_node = MagicMock()

    # Mock QoS
    mock_qos = MagicMock()
    mock_qos_profile = MagicMock()
    mock_reliability = MagicMock()
    mock_history = MagicMock()

    # Mock services and messages
    mock_image = MagicMock()
    mock_string = MagicMock()
    mock_int32 = MagicMock()
    mock_trigger = MagicMock()

    # Custom services
    mock_set_class = MagicMock()
    mock_start_burst = MagicMock()
    mock_get_status = MagicMock()

    # CvBridge
    mock_cv_bridge = MagicMock()

    return {
        "rclpy": mock_rclpy,
        "rclpy.node": mock_node,
        "rclpy.qos": mock_qos,
        "sensor_msgs.msg": mock_image,
        "std_msgs.msg": MagicMock(),
        "std_srvs.srv": mock_trigger,
        "cv_bridge": mock_cv_bridge,
        "hsr_perception.srv": MagicMock(),
    }


class TestContinuousCaptureNodeInit:
    """Test ContinuousCaptureNode initialization concepts."""

    def test_node_parameter_names(self):
        """Test that the expected parameters are defined in the node source."""
        # This test verifies the expected parameters exist by checking the source
        # without actually instantiating the ROS2 node (which requires a running ROS system)
        expected_params = [
            "class_config_path",
            "output_base_dir",
            "image_topic",
            "burst_interval",
            "burst_default_count",
            "image_format",
            "jpeg_quality",
            "preview_enabled",
        ]

        # Read the source file
        src_path = Path(__file__).parent.parent.parent.parent / "src" / "hsr_perception" / "hsr_perception" / "continuous_capture_node.py"
        source_content = src_path.read_text()

        # Verify all expected parameters are declared in source
        # Note: Some parameters may span multiple lines, so check for just the parameter name pattern
        for param in expected_params:
            # Check for both single-line and multi-line declare_parameter patterns
            single_line = f'declare_parameter("{param}"' in source_content
            multi_line = f'"{param}"' in source_content and "declare_parameter" in source_content
            assert single_line or multi_line, f"Parameter {param} not found"

    def test_node_service_names(self):
        """Test that the expected services are defined in the node source."""
        expected_services = [
            "set_class",
            "start_burst",
            "stop_burst",
            "single_capture",
            "get_status",
            "reload_config",
        ]

        src_path = Path(__file__).parent.parent.parent.parent / "src" / "hsr_perception" / "hsr_perception" / "continuous_capture_node.py"
        source_content = src_path.read_text()

        for service in expected_services:
            assert f"~/{service}" in source_content, f"Service {service} not found"


class TestImageCallback:
    """Test image callback functionality concepts."""

    def test_image_callback_method_exists(self):
        """Test that the image callback method is defined."""
        src_path = Path(__file__).parent.parent.parent.parent / "src" / "hsr_perception" / "hsr_perception" / "continuous_capture_node.py"
        source_content = src_path.read_text()

        assert "def _image_callback" in source_content
        assert "imgmsg_to_cv2" in source_content
        assert "image_lock" in source_content


class TestSetClassService:
    """Test set_class service handler."""

    def test_set_class_valid_id(self):
        """Test setting a valid class ID."""
        from continuous_capture_node import ContinuousCaptureNode

        # Create mock node
        node = MagicMock(spec=ContinuousCaptureNode)
        node.class_config = {
            "objects": [
                {"class_id": 0, "class_name": "cup", "target_samples": 100},
                {"class_id": 1, "class_name": "bottle", "target_samples": 50},
            ]
        }
        node.class_counts = {"cup": 10, "bottle": 5}
        node.current_class_id = -1
        node.current_class_name = ""
        node.status_pub = MagicMock()

        # Create a proper _get_class_info method
        def get_class_info(class_id):
            for obj in node.class_config["objects"]:
                if obj["class_id"] == class_id:
                    return obj
            return None

        def get_target_samples(class_name):
            for obj in node.class_config["objects"]:
                if obj["class_name"] == class_name:
                    return obj.get("target_samples", 100)
            return 100

        node._get_class_info = get_class_info
        node._get_target_samples = get_target_samples

        # Create request
        request = MagicMock()
        request.class_id = 0

        # Create response
        response = MagicMock()

        # Call handler
        result = ContinuousCaptureNode._handle_set_class(node, request, response)

        assert result.success is True
        assert node.current_class_id == 0
        assert node.current_class_name == "cup"

    def test_set_class_invalid_id(self):
        """Test setting an invalid class ID."""
        from continuous_capture_node import ContinuousCaptureNode

        node = MagicMock(spec=ContinuousCaptureNode)
        node.class_config = {
            "objects": [{"class_id": 0, "class_name": "cup"}]
        }
        node._get_class_info = lambda x: None

        request = MagicMock()
        request.class_id = 99  # Non-existent ID

        response = MagicMock()

        result = ContinuousCaptureNode._handle_set_class(node, request, response)

        assert result.success is False
        assert "not found" in result.message


class TestStartBurstService:
    """Test start_burst service handler."""

    def test_start_burst_success(self):
        """Test starting burst capture successfully."""
        from continuous_capture_node import ContinuousCaptureNode

        node = MagicMock(spec=ContinuousCaptureNode)
        node.class_config = {
            "objects": [{"class_id": 0, "class_name": "cup"}]
        }
        node.current_class_id = 0
        node.current_class_name = "cup"
        node.is_capturing = False
        node.burst_default_count = 50
        node.burst_interval = 0.2
        node.create_timer = MagicMock()

        def get_class_info(class_id):
            for obj in node.class_config["objects"]:
                if obj["class_id"] == class_id:
                    return obj
            return None

        node._get_class_info = get_class_info

        request = MagicMock()
        request.class_id = -1  # Use current class
        request.num_images = 10
        request.interval_seconds = 0.1

        response = MagicMock()

        result = ContinuousCaptureNode._handle_start_burst(node, request, response)

        assert result.success is True
        assert node.is_capturing is True
        assert node.burst_target == 10
        node.create_timer.assert_called_once()

    def test_start_burst_no_class_selected(self):
        """Test starting burst without selecting a class."""
        from continuous_capture_node import ContinuousCaptureNode

        node = MagicMock(spec=ContinuousCaptureNode)
        node.current_class_name = ""
        node.is_capturing = False
        node._get_class_info = lambda x: None

        request = MagicMock()
        request.class_id = -1

        response = MagicMock()

        result = ContinuousCaptureNode._handle_start_burst(node, request, response)

        assert result.success is False
        assert "No class selected" in result.message

    def test_start_burst_already_capturing(self):
        """Test starting burst when already capturing."""
        from continuous_capture_node import ContinuousCaptureNode

        node = MagicMock(spec=ContinuousCaptureNode)
        node.current_class_name = "cup"
        node.is_capturing = True  # Already capturing
        node._get_class_info = lambda x: None

        request = MagicMock()
        request.class_id = -1

        response = MagicMock()

        result = ContinuousCaptureNode._handle_start_burst(node, request, response)

        assert result.success is False
        assert "Already capturing" in result.message


class TestStopBurstService:
    """Test stop_burst service handler."""

    def test_stop_burst_success(self):
        """Test stopping burst capture."""
        from continuous_capture_node import ContinuousCaptureNode

        node = MagicMock(spec=ContinuousCaptureNode)
        node.is_capturing = True
        node.burst_count = 25
        node.burst_timer = MagicMock()

        def stop_burst():
            node.is_capturing = False
            if node.burst_timer:
                node.burst_timer.cancel()
                node.burst_timer = None

        node._stop_burst = stop_burst

        request = MagicMock()
        response = MagicMock()

        result = ContinuousCaptureNode._handle_stop_burst(node, request, response)

        assert result.success is True
        assert "25" in result.message

    def test_stop_burst_not_capturing(self):
        """Test stopping burst when not capturing."""
        from continuous_capture_node import ContinuousCaptureNode

        node = MagicMock(spec=ContinuousCaptureNode)
        node.is_capturing = False

        request = MagicMock()
        response = MagicMock()

        result = ContinuousCaptureNode._handle_stop_burst(node, request, response)

        assert result.success is False
        assert "Not currently capturing" in result.message


class TestSingleCaptureService:
    """Test single_capture service handler."""

    def test_single_capture_success(self, tmp_path):
        """Test successful single image capture."""
        from continuous_capture_node import ContinuousCaptureNode

        node = MagicMock(spec=ContinuousCaptureNode)
        node.current_class_name = "cup"
        node.latest_image = np.zeros((480, 640, 3), dtype=np.uint8)
        node.image_lock = MagicMock()
        node.class_counts = {"cup": 10}
        node.count_pub = MagicMock()

        def save_image(image, class_name):
            return str(tmp_path / f"{class_name}_test.jpg")

        node._save_image = save_image

        request = MagicMock()
        response = MagicMock()

        result = ContinuousCaptureNode._handle_single_capture(node, request, response)

        assert result.success is True
        assert "Captured" in result.message

    def test_single_capture_no_class(self):
        """Test single capture without class selected."""
        from continuous_capture_node import ContinuousCaptureNode

        node = MagicMock(spec=ContinuousCaptureNode)
        node.current_class_name = ""

        request = MagicMock()
        response = MagicMock()

        result = ContinuousCaptureNode._handle_single_capture(node, request, response)

        assert result.success is False
        assert "No class selected" in result.message

    def test_single_capture_no_image(self):
        """Test single capture without available image."""
        from continuous_capture_node import ContinuousCaptureNode

        node = MagicMock(spec=ContinuousCaptureNode)
        node.current_class_name = "cup"
        node.latest_image = None
        node.image_lock = MagicMock()

        request = MagicMock()
        response = MagicMock()

        result = ContinuousCaptureNode._handle_single_capture(node, request, response)

        assert result.success is False
        assert "No image available" in result.message


class TestGetStatusService:
    """Test get_status service handler."""

    def test_get_status(self):
        """Test getting capture status."""
        from continuous_capture_node import ContinuousCaptureNode

        node = MagicMock(spec=ContinuousCaptureNode)
        node.current_class_name = "cup"
        node.current_class_id = 0
        node.class_counts = {"cup": 10, "bottle": 5}
        node.is_capturing = True
        node.output_base_dir = "/output"
        node.class_config = {
            "objects": [
                {"class_id": 0, "class_name": "cup", "target_samples": 100},
                {"class_id": 1, "class_name": "bottle", "target_samples": 50},
            ]
        }

        def get_target_samples(class_name):
            for obj in node.class_config["objects"]:
                if obj["class_name"] == class_name:
                    return obj.get("target_samples", 100)
            return 100

        node._get_target_samples = get_target_samples

        request = MagicMock()
        response = MagicMock()
        response.all_class_names = []
        response.all_class_counts = []
        response.all_target_counts = []

        result = ContinuousCaptureNode._handle_get_status(node, request, response)

        assert result.current_class == "cup"
        assert result.current_class_id == 0
        assert result.current_count == 10
        assert result.is_capturing is True
        assert len(result.all_class_names) == 2


class TestReloadConfigService:
    """Test reload_config service handler."""

    def test_reload_config_success(self):
        """Test successful config reload."""
        from continuous_capture_node import ContinuousCaptureNode

        node = MagicMock(spec=ContinuousCaptureNode)
        node.class_config = {"objects": []}

        def load_config():
            node.class_config = {
                "objects": [{"class_id": 0, "class_name": "new_class"}]
            }

        node._load_class_config = load_config

        request = MagicMock()
        response = MagicMock()

        result = ContinuousCaptureNode._handle_reload_config(node, request, response)

        assert result.success is True
        assert "Reloaded config" in result.message


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
