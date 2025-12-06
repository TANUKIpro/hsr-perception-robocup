"""
Launch file for continuous capture node.

Usage:
  ros2 launch hsr_perception capture.launch.py
  ros2 launch hsr_perception capture.launch.py class_config:=/path/to/config.json
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for capture node."""

    # Declare arguments
    class_config_arg = DeclareLaunchArgument(
        "class_config",
        default_value="config/object_classes.json",
        description="Path to class configuration JSON file",
    )

    output_dir_arg = DeclareLaunchArgument(
        "output_dir",
        default_value="datasets/raw_captures",
        description="Base directory for captured images",
    )

    image_topic_arg = DeclareLaunchArgument(
        "image_topic",
        default_value="/hsrb/head_rgbd_sensor/rgb/image_rect_color",
        description="Camera image topic to subscribe",
    )

    burst_interval_arg = DeclareLaunchArgument(
        "burst_interval",
        default_value="0.2",
        description="Interval between burst captures in seconds",
    )

    # Create node
    capture_node = Node(
        package="hsr_perception",
        executable="continuous_capture_node.py",
        name="continuous_capture",
        output="screen",
        parameters=[
            {
                "class_config_path": LaunchConfiguration("class_config"),
                "output_base_dir": LaunchConfiguration("output_dir"),
                "image_topic": LaunchConfiguration("image_topic"),
                "burst_interval": LaunchConfiguration("burst_interval"),
                "burst_default_count": 50,
                "image_format": "jpg",
                "jpeg_quality": 95,
                "preview_enabled": True,
            }
        ],
        remappings=[
            # Add any topic remappings here if needed
        ],
    )

    return LaunchDescription(
        [
            class_config_arg,
            output_dir_arg,
            image_topic_arg,
            burst_interval_arg,
            capture_node,
        ]
    )
