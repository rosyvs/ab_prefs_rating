"""Save/load a fixed A/B session queue so multiple raters review the same items."""
from __future__ import annotations

import json
from pathlib import Path

from ab_prefs_demo.data_model import ComparisonUnit


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
    demo_recordings: int | None = None,
    demo_seed: int | None = None,
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
        "demo_recordings": demo_recordings,
        "demo_seed": demo_seed,
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
                "Rebuild units with the same GT/provider/demo settings as the manifest."
            )
        queue.append((unit, item["provider_a"], item["provider_b"]))
    return queue
