"""Shared session / compare-pool helpers."""
from __future__ import annotations

from pathlib import Path


def compare_provider_names(
    provider_dirs: dict[str, Path],
    *,
    ground_truth_name: str,
    compare_providers: str,
    include_ground_truth: bool,
) -> list[str]:
    """Names used for A/B pair generation. Empty compare_providers → all ASRs (+ GT if enabled)."""
    asr_names = sorted(provider_dirs.keys())
    valid = set(asr_names) | {ground_truth_name}
    if compare_providers.strip():
        names = [name.strip() for name in compare_providers.split(",") if name.strip()]
    elif include_ground_truth:
        names = [ground_truth_name] + asr_names
    else:
        names = list(asr_names)
    missing = [name for name in names if name not in valid]
    if missing:
        raise ValueError(f"Unknown compare_providers names: {missing}. Available: {sorted(valid)}")
    if len(names) < 2:
        raise ValueError("Need at least two providers in compare_providers")
    return names
