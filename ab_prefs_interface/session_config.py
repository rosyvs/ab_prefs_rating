"""Shared session / compare-pool helpers."""
from __future__ import annotations

from pathlib import Path


def session_recording_pool_size(session: dict) -> int | None:
    """How many recordings to load when building comparison units."""
    if session.get("unique_recordings") is not None:
        n = int(session["unique_recordings"] or 0)
        return n if n > 0 else None
    n = int(session.get("demo_recordings") or 0)
    return n if n > 0 else None


def session_unique_recordings_target(session: dict) -> int | None:
    """Min distinct recording IDs required in the manifest queue (unique_recordings field only)."""
    if session.get("unique_recordings") is None:
        return None
    n = int(session["unique_recordings"] or 0)
    return n if n > 0 else None


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
