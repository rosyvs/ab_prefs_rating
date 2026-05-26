"""Write synthetic preference JSON for testing summaries (e.g. 30 session items)."""
from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timedelta, timezone
from itertools import combinations
from pathlib import Path
from uuid import uuid4

PROVIDERS_ASR = ("aai_up3_extraprompted", "deepgram_nova3", "azure_mai")
GROUND_TRUTH = "ground_truth"
CHOICES_WEIGHTED = (
    ("A", 0.38),
    ("B", 0.34),
    ("tie", 0.18),
    ("skip", 0.10),
)
SAMPLE_GT = [
    "You guys have some awesome ideas.",
    "Open your books to page twelve.",
    "What do you notice about the character?",
    "SCRUB",
    "Mm-hmm, that's right.",
    "Turn and talk with your partner.",
]
SAMPLE_HYP = {
    "aai_up3_extraprompted": [
        "You guys had some awesome ideas.",
        "Open your books to page 12.",
        "What do you notice about the character?",
        "Carmella.",
        "Mm-hmm that's right.",
        "Turn and talk with your partner.",
    ],
    "deepgram_nova3": [
        "You guys have awesome ideas.",
        "Open your books to page twelve.",
        "What do you notice about the character?",
        "uh",
        "Mm-hmm. That's right.",
        "Turn and talk to your partner.",
    ],
    "azure_mai": [
        "You guys have some awesome ideas so",
        "Open books page twelve.",
        "What you notice about character?",
        "SCRUB",
        "Mm hmm that's right",
        "Turn talk with partner.",
    ],
}


def weighted_choice(rng: random.Random, options: tuple[tuple[str, float], ...]) -> str:
    labels, weights = zip(*options)
    return rng.choices(labels, weights=weights, k=1)[0]


def per_pair_counts(session_items: int, n_pairs: int) -> list[int]:
    base, rem = divmod(session_items, n_pairs)
    return [base + (1 if i < rem else 0) for i in range(n_pairs)]


def simulate_session(
    session_items: int = 30,
    seed: int = 42,
    strategy: str = "random",
) -> list[dict]:
    rng = random.Random(seed)
    session_id = uuid4().hex[:12]
    recording_ids = ["263682", "275140", "271136", "280348", "249387"]
    pairs = [
        (a, b)
        for a, b in combinations([GROUND_TRUTH] + list(PROVIDERS_ASR), 2)
    ]
    sizes = per_pair_counts(session_items, len(pairs))
    t0 = datetime.now(timezone.utc) - timedelta(minutes=session_items)
    records: list[dict] = []
    seg_counter = 0
    for pair_index, (provider_a, provider_b) in enumerate(pairs):
        for _ in range(sizes[pair_index]):
            seg_counter += 1
            rec = rng.choice(recording_ids)
            gt = rng.choice(SAMPLE_GT)
            hyp_a = gt if provider_a == GROUND_TRUTH else rng.choice(SAMPLE_HYP[provider_a])
            hyp_b = gt if provider_b == GROUND_TRUTH else rng.choice(SAMPLE_HYP[provider_b])
            start = float(rng.randint(100, 2000))
            end = start + rng.uniform(3.0, 12.0)
            choice = weighted_choice(rng, CHOICES_WEIGHTED)
            records.append(
                {
                    "session_id": session_id,
                    "timestamp_utc": (t0 + timedelta(seconds=seg_counter * 8)).isoformat(),
                    "strategy": strategy,
                    "recording_id": rec,
                    "segment_index": seg_counter * 3,
                    "start_seconds": start,
                    "end_seconds": end,
                    "provider_a": provider_a,
                    "provider_b": provider_b,
                    "choice": choice,
                    "note": "",
                    "ground_truth_text": gt,
                    "transcript_a": hyp_a,
                    "transcript_b": hyp_b,
                }
            )
    rng.shuffle(records)
    return records[:session_items]


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate A/B preference JSON")
    parser.add_argument("--session-items", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strategy", type=str, default="random")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/ab_prefs/preferences_simulated.json"),
    )
    args = parser.parse_args()
    out = args.output_json.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"responses": simulate_session(args.session_items, args.seed, args.strategy)}
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(payload['responses'])} records to {out}")


if __name__ == "__main__":
    main()
