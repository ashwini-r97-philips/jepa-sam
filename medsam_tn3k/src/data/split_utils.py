"""Dataset split utilities."""
from __future__ import annotations

import random
from typing import Dict, List


def stratified_split(
    items: List[Dict[str, str]],
    ratios: Dict[str, float],
    seed: int = 0,
) -> Dict[str, List[Dict[str, str]]]:
    """Split items into named subsets according to ratios.

    Args:
        items: List of dicts (e.g. {"image": ..., "mask": ...}).
        ratios: Mapping of split name to fraction, must sum to ~1.0.
        seed: Random seed for reproducibility.

    Returns:
        Dict mapping split name to list of items.
    """
    assert abs(sum(ratios.values()) - 1.0) < 1e-6, f"Ratios must sum to 1.0, got {sum(ratios.values())}"
    rng = random.Random(seed)
    shuffled = list(items)
    rng.shuffle(shuffled)

    splits: Dict[str, List[Dict[str, str]]] = {}
    offset = 0
    names = list(ratios.keys())
    for i, name in enumerate(names):
        if i == len(names) - 1:
            # Last split gets the remainder to avoid rounding issues
            splits[name] = shuffled[offset:]
        else:
            count = int(round(len(shuffled) * ratios[name]))
            splits[name] = shuffled[offset:offset + count]
            offset += count
    return splits


def labeled_unlabeled_split(
    items: List[Dict[str, str]],
    labeled_fraction: float = 0.1,
    seed: int = 0,
) -> tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Split items into labeled and unlabeled subsets.

    Returns:
        (labeled, unlabeled) tuple.
    """
    rng = random.Random(seed)
    shuffled = list(items)
    rng.shuffle(shuffled)
    n_labeled = max(1, int(round(len(shuffled) * labeled_fraction)))
    return shuffled[:n_labeled], shuffled[n_labeled:]
