"""Extract GT-span audio clips and embed them in the rating widget."""
from __future__ import annotations

import base64
import subprocess
from pathlib import Path

from tqdm import tqdm

from ab_prefs_interface.data_model import ComparisonUnit


def clip_path(clip_dir: Path, unit: ComparisonUnit) -> Path:
    end_ix = unit.segment_index if unit.segment_index_end is None else unit.segment_index_end
    name = (
        f"{unit.recording_id}_{unit.segment_index}-{end_ix}_"
        f"{unit.start_seconds:.3f}_{unit.end_seconds:.3f}.mp3"
    )
    return clip_dir.expanduser().resolve() / name


def extract_audio_clip(source: Path, start_seconds: float, end_seconds: float, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        str(start_seconds),
        "-to",
        str(end_seconds),
        "-i",
        str(source),
        "-acodec",
        "libmp3lame",
        "-q:a",
        "4",
        str(dest),
    ]
    subprocess.run(cmd, check=True)


def ensure_unit_clip(unit: ComparisonUnit, clip_dir: Path) -> Path:
    dest = clip_path(clip_dir, unit)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    extract_audio_clip(unit.audio_path, unit.start_seconds, unit.end_seconds, dest)
    return dest


def clip_data_uri(clip_file: Path) -> str:
    """Base64 embed — works in Cursor/VS Code notebooks (files/... URLs often don't)."""
    payload = base64.b64encode(clip_file.read_bytes()).decode("ascii")
    return f"data:audio/mpeg;base64,{payload}"


def served_file_url(file_path: Path, notebook_root: Path) -> str:
    """URL path for Jupyter/Colab file server (repo-relative under notebook_root)."""
    try:
        rel = file_path.resolve().relative_to(notebook_root.resolve())
    except ValueError as exc:
        raise ValueError(
            f"Clip {file_path} must live under notebook_root {notebook_root} for files/... URLs. "
            "Set clip_dir inside the repo (default: results/ab_prefs/audio_clips)."
        ) from exc
    return f"files/{rel.as_posix()}"


def unique_units(queue: list[tuple[ComparisonUnit, str, str]]) -> list[ComparisonUnit]:
    seen: set[str] = set()
    units: list[ComparisonUnit] = []
    for unit, _, _ in queue:
        if unit.span_key in seen:
            continue
        seen.add(unit.span_key)
        units.append(unit)
    return units


def ensure_queue_clips(
    queue: list[tuple[ComparisonUnit, str, str]],
    clip_dir: Path,
    notebook_root: Path | None = None,
    *,
    verbose: bool = False,
) -> dict[str, str]:
    """Build missing clips on disk; return span_key -> audio src for HTML <audio>."""
    clip_dir = clip_dir.expanduser().resolve()
    units = unique_units(queue)
    sources: dict[str, str] = {}
    unit_iter = tqdm(units, desc="audio clips", disable=not verbose)
    for unit in unit_iter:
        unit_iter.set_postfix(rec=unit.recording_id, span=f"{unit.start_seconds:.1f}s")
        clip_file = ensure_unit_clip(unit, clip_dir)
        sources[unit.span_key] = clip_data_uri(clip_file)
    return sources
