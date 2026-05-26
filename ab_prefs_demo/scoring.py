from __future__ import annotations

import statistics

import jiwer
from tqdm import tqdm

from ab_prefs_demo.data_model import ComparisonUnit


normalize_for_scoring = jiwer.Compose(
    [
        jiwer.ToLowerCase(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
    ]
)


def safe_wer(reference: str, hypothesis: str) -> float:
    ref_clean = normalize_for_scoring(reference)
    hyp_clean = normalize_for_scoring(hypothesis)
    if not ref_clean and not hyp_clean:
        return 0.0
    if not ref_clean:
        return 1.0
    return jiwer.wer(ref_clean, hyp_clean)


def alignment_measures(reference: str, hypothesis: str) -> jiwer.WordOutput:
    ref_clean = normalize_for_scoring(reference)
    hyp_clean = normalize_for_scoring(hypothesis)
    return jiwer.process_words(ref_clean, hyp_clean)


def score_unit_against_ground_truth(
    unit: ComparisonUnit,
    provider_names: list[str],
    ground_truth_name: str = "ground_truth",
) -> None:
    gt_candidate = unit.provider_candidates[ground_truth_name]
    ref_text = gt_candidate.text
    wers: list[float] = []
    deletions: list[int] = []
    for provider_name in provider_names:
        candidate = unit.provider_candidates.get(provider_name)
        if candidate is None:
            continue
        wer_value = safe_wer(ref_text, candidate.text)
        measures = alignment_measures(ref_text, candidate.text)
        candidate.wer_vs_gt = wer_value
        candidate.deletions_vs_gt = int(measures.deletions)
        candidate.substitutions_vs_gt = int(measures.substitutions)
        candidate.insertions_vs_gt = int(measures.insertions)
        wers.append(wer_value)
        deletions.append(int(measures.deletions))
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
    ground_truth_name: str = "ground_truth",
    verbose: bool = False,
) -> None:
    unit_iter = tqdm(units, desc="score vs GT", disable=not verbose)
    for unit in unit_iter:
        score_unit_against_ground_truth(
            unit=unit,
            provider_names=provider_names,
            ground_truth_name=ground_truth_name,
        )
    if verbose:
        print(f"Computed WER features for {len(units)} units × {len(provider_names)} providers")
