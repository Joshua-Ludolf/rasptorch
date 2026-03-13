"""Utility functions for rasptorch CLI."""

import json
from typing import Any, Dict


def parse_shape(shape_str: str) -> tuple:
    """Parse shape string like '2,3,4' into tuple (2, 3, 4)."""
    try:
        return tuple(int(x.strip()) for x in shape_str.split(","))
    except ValueError:
        raise ValueError(f"Invalid shape: {shape_str}")


def parse_device(device: str) -> str:
    """Validate and normalize device specification."""
    device = device.lower()
    if device not in ["cpu", "gpu"]:
        raise ValueError(f"Device must be 'cpu' or 'gpu', got '{device}'")
    return device


def format_success(message: str, data: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Format successful command result."""
    result = {
        "status": "success",
        "message": message,
    }
    if data:
        result["data"] = data
    return result


def format_error(message: str) -> Dict[str, Any]:
    """Format error result."""
    return {
        "status": "error",
        "message": message,
    }


def format_json_output(data: Dict[str, Any]) -> str:
    """Format output as JSON for agent consumption."""
    return json.dumps(data, indent=2, default=str)
