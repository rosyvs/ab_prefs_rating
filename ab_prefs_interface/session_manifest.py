"""Save/load a fixed A/B session queue so multiple raters review the same items."""
from __future__ import annotations

import json
from pathlib import Path

from ab_prefs_interface.data_model import ComparisonUnit
from ab_prefs_interface.matching import GT_SCRUB_TAG_RX, GT_SCRUB_WORD_RX, load_ground_truth_transcripts, merge_gt_segments


MANIFEST_VERSION = 1


def queue_item_dict(unit: ComparisonUnit, provider_a: str, provider_b: str) -> dict:
    return {
        "span_key": unit.span_key,
        "recording_id": unit.recording_id,
        "segment_index": unit.segment_index,
        "segment_index_end": unit.segment_index_end,
        "start_seconds": unit.start_seconds,
        "end_seconds": unit.end_seconds,
        "provider_a": provider_a,
        "provider_b": provider_b,
    }


def build_manifest_payload(
    queue: list[tuple[ComparisonUnit, str, str]],
    *,
    strategy: str,
    seed: int,
    session_items: int,
    ground_truth_name: str,
    compare_providers: list[str],
    include_ground_truth: bool,
    min_gt_words: int,
    min_audio_seconds: float,
    exclude_gt_markers: bool = True,
    unique_recordings: int | None = None,
    recording_seed: int | None = None,
) -> dict:
    return {
        "version": MANIFEST_VERSION,
        "strategy": strategy,
        "seed": seed,
        "session_items": session_items,
        "ground_truth_name": ground_truth_name,
        "include_ground_truth": include_ground_truth,
        "compare_providers": compare_providers,
        "min_gt_words": min_gt_words,
        "min_audio_seconds": min_audio_seconds,
        "exclude_gt_markers": exclude_gt_markers,
        "unique_recordings": unique_recordings,
        "recording_seed": recording_seed,
        "items": [queue_item_dict(unit, provider_a, provider_b) for unit, provider_a, provider_b in queue],
    }


def save_session_manifest(path: Path, payload: dict) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_session_manifest(path: Path) -> dict:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if payload.get("version") != MANIFEST_VERSION:
        raise ValueError(f"Unsupported manifest version: {payload.get('version')}")
    return payload


def index_units(units: list[ComparisonUnit]) -> dict[str, ComparisonUnit]:
    by_key: dict[str, ComparisonUnit] = {}
    for unit in units:
        by_key[unit.span_key] = unit
    return by_key


def manifest_recording_ids(manifest: dict) -> list[str]:
    return sorted({str(item["recording_id"]) for item in manifest["items"]})


def queue_from_manifest(
    manifest: dict,
    units: list[ComparisonUnit],
) -> list[tuple[ComparisonUnit, str, str]]:
    by_key = index_units(units)
    queue: list[tuple[ComparisonUnit, str, str]] = []
    for item in manifest["items"]:
        unit = by_key.get(item["span_key"])
        if unit is None:
            raise KeyError(
                f"Manifest item not found in built units: {item['span_key']}. "
                "Rebuild units with the same GT/provider/session settings as the manifest."
            )
        queue.append((unit, item["provider_a"], item["provider_b"]))
    return queue


def manifest_item_gt_span(
    item: dict,
    raw_segments: list[dict],
    *,
    min_gt_words: int,
    min_audio_seconds: float,
) -> tuple[str, list[dict]]:
    """Merged GT text + raw segment rows for a manifest item's segment_index range."""
    merged = merge_gt_segments(
        raw_segments, min_gt_words=min_gt_words, min_audio_seconds=min_audio_seconds
    )
    seg_start = int(item["segment_index"])
    seg_end = int(item["segment_index_end"]) if item.get("segment_index_end") is not None else seg_start
    for row in merged:
        if row["segment_index_start"] == seg_start and row["segment_index_end"] == seg_end:
            return str(row.get("orthographic_text") or ""), raw_segments[seg_start : seg_end + 1]
    return "", []


def manifest_item_is_scrub(
    item: dict,
    raw_segments: list[dict],
    *,
    min_gt_words: int,
    min_audio_seconds: float,
) -> bool:
    """True if GT span is SCRUB / should_scrub (same rules as matching.gt_segment_excluded_for_rating for scrub)."""
    text, segs = manifest_item_gt_span(
        item, raw_segments, min_gt_words=min_gt_words, min_audio_seconds=min_audio_seconds
    )
    if any(s.get("should_scrub") for s in segs):
        return True
    t = text.strip()
    if t in ("SCRUB", "[SCRUB]"):
        return True
    if GT_SCRUB_TAG_RX.search(text) or GT_SCRUB_WORD_RX.search(text):
        return True
    for seg in segs:
        seg_text = str(seg.get("orthographic_text") or "")
        if seg_text.strip() in ("SCRUB", "[SCRUB]"):
            return True
        if GT_SCRUB_TAG_RX.search(seg_text) or GT_SCRUB_WORD_RX.search(seg_text):
            return True
    return False


def filter_manifest_exclude_scrub(
    manifest: dict,
    gt_dir: Path,
    *,
    source_manifest: str | None = None,
) -> dict:
    """Copy manifest, keeping items whose GT span is not SCRUB/should_scrub."""
    gt_dir = gt_dir.expanduser().resolve()
    transcripts = load_ground_truth_transcripts(gt_dir)
    min_gt_words = int(manifest.get("min_gt_words", 0) or 0)
    min_audio_seconds = float(manifest.get("min_audio_seconds", 0.0) or 0.0)
    kept: list[dict] = []
    dropped: list[str] = []
    for item in manifest["items"]:
        raw = transcripts.get(str(item["recording_id"]))
        if raw is None:
            raise KeyError(f"No GT transcript for recording {item['recording_id']}")
        if manifest_item_is_scrub(item, raw, min_gt_words=min_gt_words, min_audio_seconds=min_audio_seconds):
            dropped.append(item["span_key"])
        else:
            kept.append(item)
    out = dict(manifest)
    out["items"] = kept
    out["session_items"] = len(kept)
    if source_manifest:
        out["source_manifest"] = source_manifest
    out["source_manifest_filter"] = {
        "exclude_scrub": True,
        "source_item_count": len(manifest["items"]),
        "dropped_span_keys": dropped,
    }
    return out


def copy_manifest_from_source(
    source_manifest_path: Path,
    *,
    gt_dir: Path,
    exclude_scrub: bool = False,
) -> dict:
    source_manifest_path = source_manifest_path.expanduser().resolve()
    manifest = load_session_manifest(source_manifest_path)
    source_label = str(source_manifest_path)
    if not exclude_scrub:
        out = dict(manifest)
        out["source_manifest"] = source_label
        return out
    return filter_manifest_exclude_scrub(
        manifest, gt_dir, source_manifest=source_label
    )
