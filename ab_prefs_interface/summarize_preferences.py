"""Aggregate human A/B preference JSON into win rates and pairwise tables."""
from __future__ import annotations

import argparse
import html
import json
from collections import defaultdict
from io import StringIO
from pathlib import Path

import pandas as pd

from ab_prefs_interface.dimension_ui import DIMENSION_CHOICE_COLS, DIMENSION_LABELS, DIMENSIONS, RATING_MODES
from ab_prefs_interface.sampling import format_exposure_share


def summarize_cli_command(output_json: Path | str, ground_truth_name: str = "ground_truth") -> str:
    path = Path(output_json).expanduser().resolve()
    return (
        f"python -m ab_prefs_interface.summarize_preferences "
        f"--input-json {path} --ground-truth-name {ground_truth_name}"
    )


def load_preferences(path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text(encoding="utf-8"))
    df = pd.DataFrame(payload["responses"])
    for col in ("rating_mode", "choice_text", "choice_timing", "choice_diarization"):
        if col not in df.columns:
            df[col] = ""
    return df


def detect_rating_mode(df: pd.DataFrame) -> str:
    if df.empty:
        return "overall"
    modes = df["rating_mode"].replace("", "overall").unique()
    if len(modes) == 1:
        return str(modes[0])
    if "multi_dimension" in modes:
        return "multi_dimension"
    return "overall"


def provider_names(df: pd.DataFrame) -> set[str]:
    names: set[str] = set()
    names.update(df["provider_a"].tolist())
    names.update(df["provider_b"].tolist())
    return names


def asr_exposure(df: pd.DataFrame, asr_names: list[str], *, choice_col: str) -> dict[str, int]:
    rated = df[df[choice_col].isin(["A", "B", "tie"])]
    counts: dict[str, int] = {name: 0 for name in asr_names}
    for _, row in rated.iterrows():
        if row["provider_a"] in counts:
            counts[row["provider_a"]] += 1
        if row["provider_b"] in counts:
            counts[row["provider_b"]] += 1
    return counts


def win_loss_counts(df: pd.DataFrame, provider: str, *, choice_col: str) -> tuple[int, int, int]:
    """Return wins, losses (excluding ties), ties for trials where provider is on ballot."""
    wins = losses = ties = 0
    subset = df[df[choice_col].isin(["A", "B", "tie"])]
    for _, row in subset.iterrows():
        if provider not in (row["provider_a"], row["provider_b"]):
            continue
        if row[choice_col] == "tie":
            ties += 1
        elif row[choice_col] == "A":
            if row["provider_a"] == provider:
                wins += 1
            else:
                losses += 1
        elif row[choice_col] == "B":
            if row["provider_b"] == provider:
                wins += 1
            else:
                losses += 1
    return wins, losses, ties


def summarize_to_text(
    df: pd.DataFrame,
    *,
    choice_col: str = "choice",
    ground_truth_name: str = "ground_truth",
    exclude_ground_truth: bool = False,
    title: str | None = None,
) -> str:
    buf = StringIO()
    if title:
        buf.write(f"=== {title} ===\n")
    n_rows = len(df)
    buf.write(f"Total rows: {n_rows}\n")
    buf.write("\nChoices (share of trials):\n")
    choice_counts = df[choice_col].value_counts()
    for choice, count in choice_counts.items():
        if not str(choice).strip():
            continue
        share = count / n_rows if n_rows else float("nan")
        buf.write(f"  {choice}: {share:.3f}\n")
    names = sorted(provider_names(df))
    asr_names = [name for name in names if name != ground_truth_name]
    exposure = asr_exposure(df, asr_names, choice_col=choice_col)
    exposure_total = sum(exposure.values())
    buf.write("\nASR exposure (share of rated ballot slots, ties excluded):\n")
    for name in asr_names:
        buf.write(f"  {name}: {format_exposure_share(exposure[name], exposure_total)}\n")
    buf.write("\nWin rate (wins / (wins + losses), ties excluded):\n")
    for name in names:
        if exclude_ground_truth and name == ground_truth_name:
            continue
        wins, losses, ties = win_loss_counts(df, name, choice_col=choice_col)
        denom = wins + losses
        rate_str = f"{wins / denom:.3f}" if denom else "n/a"
        buf.write(f"  {name}: {rate_str}  n={denom}  ties={ties}\n")
    matrix: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for _, row in df[df[choice_col].isin(["A", "B"])].iterrows():
        provider_a, provider_b = row["provider_a"], row["provider_b"]
        if exclude_ground_truth and ground_truth_name in (provider_a, provider_b):
            continue
        if provider_a == ground_truth_name or provider_b == ground_truth_name:
            continue
        pair = tuple(sorted([provider_a, provider_b]))
        winner = provider_a if row[choice_col] == "A" else provider_b
        matrix[pair][winner] += 1
        matrix[pair]["total"] += 1
    buf.write("\nPairwise ASR vs ASR (win rate):\n")
    for pair in sorted(matrix.keys()):
        counts = matrix[pair]
        total = counts["total"]
        left, right = pair
        left_wins = counts[left]
        right_wins = counts[right]
        left_rate = left_wins / total if total else float("nan")
        right_rate = right_wins / total if total else float("nan")
        buf.write(
            f"  {left} vs {right}  n={total}  "
            f"{left}: {left_rate:.3f}  {right}: {right_rate:.3f}\n"
        )
    return buf.getvalue()


def summarize(
    df: pd.DataFrame,
    *,
    choice_col: str = "choice",
    ground_truth_name: str = "ground_truth",
    exclude_ground_truth: bool = False,
    title: str | None = None,
) -> None:
    print(summarize_to_text(
        df,
        choice_col=choice_col,
        ground_truth_name=ground_truth_name,
        exclude_ground_truth=exclude_ground_truth,
        title=title,
    ), end="")


def summarize_multi_dimension(
    df: pd.DataFrame,
    *,
    ground_truth_name: str = "ground_truth",
    exclude_ground_truth: bool = False,
) -> dict[str, str]:
    multi = df[df["rating_mode"] == "multi_dimension"]
    if multi.empty:
        multi = df
    summaries: dict[str, str] = {}
    for dim in DIMENSIONS:
        col = DIMENSION_CHOICE_COLS[dim]
        summaries[dim] = summarize_to_text(
            multi,
            choice_col=col,
            ground_truth_name=ground_truth_name,
            exclude_ground_truth=exclude_ground_truth,
            title=DIMENSION_LABELS[dim],
        )
    return summaries


def summarize_completion_html(
    output_json: Path | str,
    ground_truth_name: str = "ground_truth",
    rating_mode: str = "overall",
) -> str:
    path = Path(output_json).expanduser().resolve()
    df = load_preferences(path)
    mode = rating_mode if rating_mode in RATING_MODES else detect_rating_mode(df)
    summarize_cmd = summarize_cli_command(path, ground_truth_name)
    parts = [
        "<h3>Review complete</h3>",
        f"<p>Saved responses to <code>{html.escape(str(path))}</code></p>",
    ]
    if mode == "multi_dimension":
        summaries = summarize_multi_dimension(df, ground_truth_name=ground_truth_name)
        parts.append("<p><strong>Summary by dimension:</strong></p>")
        for dim in DIMENSIONS:
            label = DIMENSION_LABELS[dim]
            text = summaries[dim]
            print(f"\n{label}\n{text}", end="")
            parts.append(f"<h4>{html.escape(label)}</h4>")
            parts.append(f"<pre>{html.escape(text)}</pre>")
    else:
        parts.append("<p><strong>Summarize after rating (separate, optional):</strong></p>")
        parts.append(f"<pre>{html.escape(summarize_cmd)}</pre>")
        print(f"\nSummarize after rating (separate, optional):\n{summarize_cmd}")
    return "".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-json",
        type=Path,
        default=Path("results/ab_prefs/preferences_simulated.json"),
    )
    parser.add_argument("--ground-truth-name", type=str, default="ground_truth")
    parser.add_argument("--exclude-ground-truth", action="store_true")
    parser.add_argument(
        "--rating-mode",
        choices=["auto", *RATING_MODES],
        default="auto",
        help="auto: detect from JSON; overall: single choice column; multi_dimension: per-dimension cols",
    )
    parser.add_argument(
        "--dimension",
        choices=[*DIMENSIONS, "all"],
        default="all",
        help="For multi_dimension files: which dimension(s) to summarize",
    )
    args = parser.parse_args()
    df = load_preferences(args.input_json)
    mode = detect_rating_mode(df) if args.rating_mode == "auto" else args.rating_mode
    if mode == "multi_dimension":
        if args.dimension == "all":
            for dim in DIMENSIONS:
                col = DIMENSION_CHOICE_COLS[dim]
                summarize(
                    df[df["rating_mode"].isin(["", "multi_dimension"])],
                    choice_col=col,
                    ground_truth_name=args.ground_truth_name,
                    exclude_ground_truth=args.exclude_ground_truth,
                    title=DIMENSION_LABELS[dim],
                )
                print()
        else:
            col = DIMENSION_CHOICE_COLS[args.dimension]
            summarize(
                df[df["rating_mode"].isin(["", "multi_dimension"])],
                choice_col=col,
                ground_truth_name=args.ground_truth_name,
                exclude_ground_truth=args.exclude_ground_truth,
                title=DIMENSION_LABELS[args.dimension],
            )
    else:
        summarize(
            df,
            choice_col="choice",
            ground_truth_name=args.ground_truth_name,
            exclude_ground_truth=args.exclude_ground_truth,
        )


if __name__ == "__main__":
    main()
