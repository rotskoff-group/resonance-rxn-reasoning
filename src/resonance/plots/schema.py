"""Loading and validating the per-figure JSON inputs."""
import json
from pathlib import Path


class DataError(ValueError):
    """Input JSON does not match the documented format."""


def load(path):
    path = Path(path)
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise DataError(f"{path}: invalid JSON ({exc})") from exc


def require(obj, keys, where):
    """Raise DataError naming every missing key."""
    if not isinstance(obj, dict):
        raise DataError(f"{where}: expected an object, got {type(obj).__name__}")
    missing = [k for k in keys if k not in obj]
    if missing:
        raise DataError(f"{where}: missing {', '.join(missing)}")


def require_list(obj, key, where):
    value = obj.get(key)
    if not isinstance(value, list) or not value:
        raise DataError(f"{where}.{key}: expected a non-empty list")
    return value
