"""Utilities for recording and aggregating SSL checkpoint sweep results."""
from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def save_sweep_row(row: Dict[str, Any], jsonl_path: str | Path) -> None:
    """Append a single result row to a JSONL file."""
    jsonl_path = Path(jsonl_path)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, "a") as f:
        f.write(json.dumps(row) + "\n")


def load_sweep_rows(jsonl_path: str | Path) -> List[Dict[str, Any]]:
    """Load all rows from a JSONL file."""
    rows = []
    path = Path(jsonl_path)
    if path.exists():
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def compile_sweep_summary(jsonl_path: str | Path, output_dir: str | Path) -> None:
    """Read JSONL results and produce CSV + Markdown summary table."""
    import pandas as pd

    rows = load_sweep_rows(jsonl_path)
    if not rows:
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)

    # CSV
    df.to_csv(output_dir / "sweep_summary.csv", index=False)

    # Markdown table
    cols = [
        "method", "ssl_epoch", "ssl_train_loss",
        "zero_shot_dice_mean", "zero_shot_iou_mean",
        "finetune_best_epoch", "finetune_best_val_loss",
        "dice_mean", "dice_std", "iou_mean", "iou_std", "n_cases",
    ]
    available_cols = [c for c in cols if c in df.columns]
    md_df = df[available_cols].copy()

    lines = []
    lines.append("# SSL Checkpoint Sweep Results\n")
    lines.append(f"Generated: {datetime.now().isoformat()}\n")
    lines.append(md_df.to_markdown(index=False))
    lines.append("")

    with open(output_dir / "sweep_summary.md", "w") as f:
        f.write("\n".join(lines))


def save_config_snapshot(
    configs: Dict[str, Any],
    output_dir: str | Path,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Save a full config snapshot including git hash and timestamp."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshot = {
        "timestamp": datetime.now().isoformat(),
        "git_hash": _get_git_hash(),
        "configs": configs,
    }
    if extra:
        snapshot.update(extra)

    with open(output_dir / "config_snapshot.json", "w") as f:
        json.dump(snapshot, f, indent=2, default=str)


def _get_git_hash() -> Optional[str]:
    """Get current git commit hash, or None if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None
