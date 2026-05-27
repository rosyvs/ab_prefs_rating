"""Shared session / compare-pool helpers."""
from __future__ import annotations

from pathlib import Path


def session_recording_pool_size(session: dict) -> int | None:
    """How many recordings to load when building comparison units."""
    if session.get("unique_recordings") is None:
        return None
    n = int(session["unique_recordings"] or 0)
    return n if n > 0 else None


def session_unique_recordings_target(session: dict) -> int | None:
    """Min distinct recording IDs required in the manifest queue."""
    return session_recording_pool_size(session)


def parse_compare_providers(raw: str | list | None) -> list[str]:
    """Normalize compare_providers from session JSON (comma string) or manifest (list)."""
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(p).strip() for p in raw if str(p).strip()]
    return [p.strip() for p in str(raw).split(",") if p.strip()]


def compare_provider_names(
    provider_dirs: dict[str, Path],
    *,
    ground_truth_name: str,
    compare_providers: str | list | None,
    include_ground_truth: bool,
) -> list[str]:
    """Names used for A/B pair generation. Empty compare_providers → all ASRs (+ GT if enabled)."""
    asr_names = sorted(provider_dirs.keys())
    valid = set(asr_names) | {ground_truth_name}
    names = parse_compare_providers(compare_providers)
    if not names:
        if include_ground_truth:
            names = [ground_truth_name] + asr_names
        else:
            names = list(asr_names)
    missing = [name for name in names if name not in valid]
    if missing:
        raise ValueError(f"Unknown compare_providers names: {missing}. Available: {sorted(valid)}")
    if len(names) < 2:
        raise ValueError("Need at least two providers in compare_providers")
    return names
