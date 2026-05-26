from __future__ import annotations

import itertools
import random
from collections import Counter
from fractions import Fraction

from tqdm import tqdm

from ab_prefs_demo.data_model import ComparisonUnit

SAMPLING_STRATEGIES = ("random", "max_discrepancy", "max_wer", "most_deletions")


def format_exposure_share(count: int, total: int) -> str:
    if total <= 0:
        return "n/a"
    return f"{Fraction(count, total)} ({count}/{total})"


def pair_combinations(provider_names: list[str]) -> list[tuple[str, str]]:
    return [(left, right) for left, right in itertools.combinations(provider_names, 2)]


def asr_provider_names(provider_names: list[str], ground_truth_name: str) -> list[str]:
    return sorted(name for name in provider_names if name != ground_truth_name)


def provider_appearance_counts(
    queue: list[tuple[ComparisonUnit, str, str]],
    provider_names: list[str],
) -> Counter[str]:
    counts: Counter[str] = Counter({name: 0 for name in provider_names})
    for _, provider_a, provider_b in queue:
        if provider_a in counts:
            counts[provider_a] += 1
        if provider_b in counts:
            counts[provider_b] += 1
    return counts


def unit_has_pair_transcripts(unit: ComparisonUnit, provider_a: str, provider_b: str) -> bool:
    candidate_a = unit.provider_candidates.get(provider_a)
    candidate_b = unit.provider_candidates.get(provider_b)
    if candidate_a is None or candidate_b is None:
        return False
    return candidate_a.has_transcript() and candidate_b.has_transcript()


def filter_queue_with_transcripts(
    queue: list[tuple[ComparisonUnit, str, str]],
) -> list[tuple[ComparisonUnit, str, str]]:
    return [item for item in queue if unit_has_pair_transcripts(item[0], item[1], item[2])]


def units_with_pair(units: list[ComparisonUnit], provider_a: str, provider_b: str) -> list[ComparisonUnit]:
    return [unit for unit in units if unit_has_pair_transcripts(unit, provider_a, provider_b)]


def eligible_queue_items(
    units: list[ComparisonUnit],
    pairs: list[tuple[str, str]],
) -> list[tuple[ComparisonUnit, str, str]]:
    items: list[tuple[ComparisonUnit, str, str]] = []
    for provider_a, provider_b in pairs:
        for unit in units_with_pair(units, provider_a, provider_b):
            items.append((unit, provider_a, provider_b))
    return items


def metric_sort_key(strategy: str) -> str:
    if strategy == "max_discrepancy":
        return "wer_spread"
    if strategy == "max_wer":
        return "avg_wer"
    if strategy == "most_deletions":
        return "avg_deletions"
    raise ValueError(f"Unknown strategy for metric sort key: {strategy}")


def asr_balance_target(session_items: int, n_asr: int) -> int:
    if n_asr <= 0:
        return 0
    return (2 * session_items) // n_asr


def pick_session_items(
    ordered: list[tuple[ComparisonUnit, str, str]],
    session_items: int,
    asr_names: list[str],
    seed: int,
) -> list[tuple[ComparisonUnit, str, str]]:
    """Pick exactly session_items from pre-ordered eligible pool; skip ineligible empties."""
    if len(ordered) < session_items:
        raise ValueError(
            f"Only {len(ordered)} eligible comparisons (both providers have overlapping transcript); "
            f"need {session_items}. Add demo_recordings, relax min_gt_words/min_audio_seconds, "
            f"or lower session_items."
        )
    balance_cap = asr_balance_target(session_items, len(asr_names)) if asr_names else None
    rng = random.Random(seed)

    def try_pick(pool: list[tuple[ComparisonUnit, str, str]], cap: int | None) -> list[tuple[ComparisonUnit, str, str]]:
        counts: Counter[str] = Counter({name: 0 for name in asr_names})
        chosen: list[tuple[ComparisonUnit, str, str]] = []
        used: set[tuple[str, str, str]] = set()
        for item in pool:
            if len(chosen) >= session_items:
                break
            unit, provider_a, provider_b = item
            key = (unit.span_key, provider_a, provider_b)
            if key in used:
                continue
            asr_in = [name for name in (provider_a, provider_b) if name in counts]
            if cap is not None and asr_in and any(counts[name] >= cap for name in asr_in):
                continue
            chosen.append(item)
            used.add(key)
            for name in asr_in:
                counts[name] += 1
        return chosen

    if balance_cap and asr_names:
        for attempt in range(100):
            pool = ordered[:]
            if attempt > 0:
                rng.shuffle(pool)
            chosen = try_pick(pool, balance_cap)
            if len(chosen) == session_items:
                counts = provider_appearance_counts(chosen, asr_names)
                if all(counts[name] == balance_cap for name in asr_names):
                    return chosen
    chosen = try_pick(ordered[:], balance_cap)
    if len(chosen) < session_items:
        pool = ordered[:]
        rng.shuffle(pool)
        chosen = try_pick(pool, None)
    if len(chosen) < session_items:
        raise ValueError(
            f"Could only pick {len(chosen)} unique eligible comparisons; need {session_items}."
        )
    return chosen[:session_items]


def build_session_queue(
    units: list[ComparisonUnit],
    provider_names: list[str],
    strategy: str,
    seed: int,
    session_items: int = 30,
    per_pair_sample_size: int | None = None,
    ground_truth_name: str = "ground_truth",
    verbose: bool = False,
) -> list[tuple[ComparisonUnit, str, str]]:
    if strategy not in SAMPLING_STRATEGIES:
        raise ValueError(f"Unsupported strategy {strategy}. Must be one of {SAMPLING_STRATEGIES}")
    pairs = pair_combinations(provider_names)
    if not pairs:
        raise ValueError("Need at least two provider names to build a session queue")
    asr_names = asr_provider_names(provider_names, ground_truth_name)
    target_items = per_pair_sample_size * len(pairs) if per_pair_sample_size is not None else session_items
    eligible = eligible_queue_items(units, pairs)
    if verbose:
        print(f"Eligible comparisons (both providers have transcript): {len(eligible)}")
    if strategy == "random":
        rng = random.Random(seed)
        ordered = eligible[:]
        rng.shuffle(ordered)
    else:
        metric = metric_sort_key(strategy)
        ordered = sorted(eligible, key=lambda item: item[0].features.get(metric, 0.0), reverse=True)
    queue = pick_session_items(ordered, target_items, asr_names, seed)
    if verbose:
        print(f"Session queue: {len(queue)} items ({strategy}, target={target_items})")
        if asr_names:
            counts = provider_appearance_counts(queue, asr_names)
            total = sum(counts.values())
            exposure = {name: format_exposure_share(counts[name], total) for name in asr_names}
            print("ASR exposure:", exposure)
    return queue
