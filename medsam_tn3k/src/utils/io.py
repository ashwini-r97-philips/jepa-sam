"""I/O helpers for configs, JSON, and checkpoints."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path: str | Path) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def load_split(path: str | Path) -> List[Dict[str, str]]:
    """Load a split JSON file (list of {image, mask} dicts)."""
    return load_json(path)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
