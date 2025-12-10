"""
Custom Exception Classes for HSR Perception App

Provides a hierarchy of exceptions for different error scenarios,
enabling more precise error handling and better error messages.

Usage:
    from utils.exceptions import TaskError, PathError

    try:
        # ... operation
    except TaskError as e:
        logger.error(f"Task failed: {e}")
"""

from typing import Optional, Any


class HSRPerceptionError(Exception):
    """
    Base exception class for HSR Perception application.

    All custom exceptions inherit from this class, allowing
    broad exception catching when needed.
    """

    def __init__(self, message: str, details: Optional[dict[str, Any]] = None):
        """
        Initialize the exception.

        Args:
            message: Human-readable error message
            details: Optional dictionary with additional error context
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class TaskError(HSRPerceptionError):
    """
    Exception for task execution errors.

    Raised when a background task (training, annotation, evaluation)
    fails to start, execute, or complete.
    """

    def __init__(
        self,
        message: str,
        task_id: Optional[str] = None,
        task_type: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        details = details or {}
        if task_id:
            details["task_id"] = task_id
        if task_type:
            details["task_type"] = task_type
        super().__init__(message, details)
        self.task_id = task_id
        self.task_type = task_type


class PathError(HSRPerceptionError):
    """
    Exception for path-related errors.

    Raised when path operations fail, such as missing directories,
    invalid paths, or symlink failures.
    """

    def __init__(
        self,
        message: str,
        path: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        details = details or {}
        if path:
            details["path"] = path
        super().__init__(message, details)
        self.path = path


class ROS2Error(HSRPerceptionError):
    """
    Exception for ROS2 communication errors.

    Raised when ROS2 operations fail, such as service calls,
    topic subscriptions, or node communication issues.
    """

    def __init__(
        self,
        message: str,
        service_name: Optional[str] = None,
        topic_name: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        details = details or {}
        if service_name:
            details["service_name"] = service_name
        if topic_name:
            details["topic_name"] = topic_name
        super().__init__(message, details)
        self.service_name = service_name
        self.topic_name = topic_name


class RegistryError(HSRPerceptionError):
    """
    Exception for object registry errors.

    Raised when registry operations fail, such as object
    creation, update, or deletion errors.
    """

    def __init__(
        self,
        message: str,
        object_id: Optional[str] = None,
        operation: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        details = details or {}
        if object_id:
            details["object_id"] = object_id
        if operation:
            details["operation"] = operation
        super().__init__(message, details)
        self.object_id = object_id
        self.operation = operation


class ValidationError(HSRPerceptionError):
    """
    Exception for validation errors.

    Raised when input validation fails, such as invalid
    configuration, missing required fields, or constraint violations.
    """

    def __init__(
        self,
        message: str,
        field_name: Optional[str] = None,
        invalid_value: Optional[Any] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        details = details or {}
        if field_name:
            details["field_name"] = field_name
        if invalid_value is not None:
            details["invalid_value"] = str(invalid_value)
        super().__init__(message, details)
        self.field_name = field_name
        self.invalid_value = invalid_value
