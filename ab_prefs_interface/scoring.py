from __future__ import annotations

import statistics
from pathlib import Path

from tqdm import tqdm

from ab_prefs_interface.asr_wer import RecordingWerCache
from ab_prefs_interface.data_model import ComparisonUnit
from ab_prefs_interface.matching import load_ground_truth_transcripts


def score_unit_against_ground_truth(
    unit: ComparisonUnit,
    provider_names: list[str],
    wer_cache: RecordingWerCache,
    ground_truth_name: str = "ground_truth",
) -> None:
    wers: list[float] = []
    deletions: list[int] = []
    for provider_name in provider_names:
        candidate = unit.provider_candidates.get(provider_name)
        if candidate is None:
            continue
        stats = wer_cache.clip_wer(
            unit.recording_id,
            provider_name,
            unit.start_seconds,
            unit.end_seconds,
        )
        rw = int(stats["ref_words"])
        wer_value = float(stats["wer"]) if rw > 0 else 0.0
        candidate.wer_vs_gt = wer_value
        candidate.deletions_vs_gt = int(stats["deletions"])
        candidate.substitutions_vs_gt = int(stats["substitutions"])
        candidate.insertions_vs_gt = int(stats["insertions"])
        wers.append(wer_value)
        deletions.append(int(stats["deletions"]))
    if not wers:
        unit.features["avg_wer"] = 0.0
        unit.features["wer_spread"] = 0.0
    else:
        unit.features["avg_wer"] = float(statistics.mean(wers))
        unit.features["wer_spread"] = float(max(wers) - min(wers))
    unit.features["avg_deletions"] = float(statistics.mean(deletions)) if deletions else 0.0


def score_units(
    units: list[ComparisonUnit],
    provider_names: list[str],
    *,
    gt_dir: Path,
    provider_dirs: dict[str, Path],
    asr_eval_root: Path | str,
    ground_truth_name: str = "ground_truth",
    verbose: bool = False,
) -> None:
    gt_dir = gt_dir.expanduser().resolve()
    recording_ids = sorted({u.recording_id for u in units})
    all_gt = load_ground_truth_transcripts(gt_dir)
    gt_by_rec = {rid: all_gt[rid] for rid in recording_ids}
    wer_cache = RecordingWerCache(asr_eval_root, gt_by_rec, provider_dirs)
    unit_iter = tqdm(units, desc="score vs GT (DD210 WER)", disable=not verbose)
    for unit in unit_iter:
        score_unit_against_ground_truth(
            unit=unit,
            provider_names=provider_names,
            wer_cache=wer_cache,
            ground_truth_name=ground_truth_name,
        )
    if verbose:
        print(
            f"Computed DD210 WER features for {len(units)} units × {len(provider_names)} providers "
            f"(asr_eval_root={Path(asr_eval_root).resolve()})"
        )
