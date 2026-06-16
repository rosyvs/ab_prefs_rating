from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

from ab_prefs_interface.data_model import ComparisonUnit


def dir_fingerprint(directory: Path, pattern: str) -> str:
    root = directory.expanduser().resolve()
    files = sorted(root.glob(pattern))
    if not files:
        return f"{root}:empty"
    mtimes = [path.stat().st_mtime for path in files]
    nbytes = sum(path.stat().st_size for path in files)
    return f"{root}|n={len(files)}|max_mtime={max(mtimes):.6f}|bytes={nbytes}"


def build_cache_key(
    gt_dir: Path,
    provider_dirs: dict[str, Path],
    audio_dir: Path,
    ground_truth_name: str,
    *,
    recording_pool_size: int | None = None,
    recording_seed: int | None = None,
    recording_ids: list[str] | None = None,
    min_gt_words: int = 0,
    min_audio_seconds: float = 0.0,
    exclude_gt_markers: bool = True,
) -> str:
    parts = [
        "v3",  # bump when ProviderCandidate schema changes (forces cache rebuild)
        dir_fingerprint(gt_dir, "*.jsonl"),
        dir_fingerprint(audio_dir, "*.mp3"),
        ground_truth_name,
        f"min_gt_words={min_gt_words}",
        f"min_audio_seconds={min_audio_seconds}",
        f"exclude_gt_markers={exclude_gt_markers}",
    ]
    for name in sorted(provider_dirs):
        parts.append(f"{name}:{dir_fingerprint(provider_dirs[name], '*.json')}")
    if recording_ids is not None:
        parts.append("manifest_rec=" + ",".join(sorted(recording_ids)))
    elif recording_pool_size is not None:
        parts.append(f"pool_n={recording_pool_size}|recording_seed={recording_seed}")
    digest = hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()
    return digest[:16]


def subset_cache_root(cache_dir: Path) -> Path:
    """Separate tree from full unit cache so subset builds never overwrite full-cache pickles."""
    return cache_dir.expanduser().resolve() / "subset"


def cache_bucket_dir(cache_dir: Path, cache_key: str) -> Path:
    return cache_dir.expanduser().resolve() / cache_key


def recording_cache_path(cache_dir: Path, cache_key: str, recording_id: str) -> Path:
    return cache_bucket_dir(cache_dir, cache_key) / f"{recording_id}.pkl"


def load_recording_units(cache_path: Path) -> list[ComparisonUnit]:
    with cache_path.open("rb") as handle:
        return pickle.load(handle)


def save_recording_units(cache_path: Path, units: list[ComparisonUnit]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as handle:
        pickle.dump(units, handle, protocol=pickle.HIGHEST_PROTOCOL)
