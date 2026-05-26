"""Aggregate human A/B preference JSON into win rates and pairwise tables."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from ab_prefs_demo.sampling import format_exposure_share


def summarize_cli_command(output_json: Path | str, ground_truth_name: str = "ground_truth") -> str:
    path = Path(output_json).expanduser().resolve()
    return (
        f"python -m ab_prefs_demo.summarize_preferences "
        f"--input-json {path} --ground-truth-name {ground_truth_name}"
    )


def load_preferences(path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return pd.DataFrame(payload["responses"])


def provider_names(df: pd.DataFrame) -> set[str]:
    names: set[str] = set()
    names.update(df["provider_a"].tolist())
    names.update(df["provider_b"].tolist())
    return names


def asr_exposure(df: pd.DataFrame, asr_names: list[str]) -> dict[str, int]:
    rated = df[df["choice"].isin(["A", "B", "tie"])]
    counts: dict[str, int] = {name: 0 for name in asr_names}
    for _, row in rated.iterrows():
        if row["provider_a"] in counts:
            counts[row["provider_a"]] += 1
        if row["provider_b"] in counts:
            counts[row["provider_b"]] += 1
    return counts


def win_loss_counts(df: pd.DataFrame, provider: str) -> tuple[int, int, int]:
    """Return wins, losses (excluding ties), ties for trials where provider is on ballot."""
    wins = losses = ties = 0
    subset = df[df["choice"].isin(["A", "B", "tie"])]
    for _, row in subset.iterrows():
        if provider not in (row["provider_a"], row["provider_b"]):
            continue
        if row["choice"] == "tie":
            ties += 1
        elif row["choice"] == "A":
            if row["provider_a"] == provider:
                wins += 1
            else:
                losses += 1
        elif row["choice"] == "B":
            if row["provider_b"] == provider:
                wins += 1
            else:
                losses += 1
    return wins, losses, ties


def summarize(
    df: pd.DataFrame,
    *,
    ground_truth_name: str = "ground_truth",
    exclude_ground_truth: bool = False,
) -> None:
    n_rows = len(df)
    print(f"Total rows: {n_rows}")
    print("\nChoices (share of trials):")
    choice_counts = df["choice"].value_counts()
    for choice, count in choice_counts.items():
        share = count / n_rows if n_rows else float("nan")
        print(f"  {choice}: {share:.3f}")
    names = sorted(provider_names(df))
    asr_names = [name for name in names if name != ground_truth_name]
    exposure = asr_exposure(df, asr_names)
    exposure_total = sum(exposure.values())
    print("\nASR exposure (share of rated ballot slots, ties excluded):")
    for name in asr_names:
        print(f"  {name}: {format_exposure_share(exposure[name], exposure_total)}")
    print("\nWin rate (wins / (wins + losses), ties excluded):")
    for name in names:
        if exclude_ground_truth and name == ground_truth_name:
            continue
        wins, losses, ties = win_loss_counts(df, name)
        denom = wins + losses
        rate_str = f"{wins / denom:.3f}" if denom else "n/a"
        print(f"  {name}: {rate_str}  n={denom}  ties={ties}")
    matrix: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for _, row in df[df["choice"].isin(["A", "B"])].iterrows():
        provider_a, provider_b = row["provider_a"], row["provider_b"]
        if exclude_ground_truth and ground_truth_name in (provider_a, provider_b):
            continue
        if provider_a == ground_truth_name or provider_b == ground_truth_name:
            continue
        pair = tuple(sorted([provider_a, provider_b]))
        winner = provider_a if row["choice"] == "A" else provider_b
        matrix[pair][winner] += 1
        matrix[pair]["total"] += 1
    print("\nPairwise ASR vs ASR (win rate):")
    for pair in sorted(matrix.keys()):
        counts = matrix[pair]
        total = counts["total"]
        left, right = pair
        left_wins = counts[left]
        right_wins = counts[right]
        left_rate = left_wins / total if total else float("nan")
        right_rate = right_wins / total if total else float("nan")
        print(
            f"  {left} vs {right}  n={total}  "
            f"{left}: {left_rate:.3f}  {right}: {right_rate:.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-json",
        type=Path,
        default=Path("results/ab_prefs/preferences_simulated.json"),
    )
    parser.add_argument("--ground-truth-name", type=str, default="ground_truth")
    parser.add_argument("--exclude-ground-truth", action="store_true")
    args = parser.parse_args()
    summarize(
        load_preferences(args.input_json),
        ground_truth_name=args.ground_truth_name,
        exclude_ground_truth=args.exclude_ground_truth,
    )


if __name__ == "__main__":
    main()
